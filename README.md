# spikyeggroll

Training spiking neural networks with the EGGROLL evolution strategy in JAX.

## Results

**93.7% on MNIST** with a 784-128-128-10 feedforward SNN (118K parameters), trained purely with evolution strategies — no backpropagation, no surrogate gradients.

| Method | MNIST Accuracy |
|--------|---------------|
| SG-BPTT (baseline) | ~99.5% |
| **EGGROLL (this work)** | **93.7%** |

Best configuration: pop_size=10000, rank=3, sigma=0.007, lr=0.001, 4000 epochs (~60 min on RTX 4080).

## Architecture

```
Input [B, 25, 784]  (Poisson rate-coded MNIST, T=25 timesteps)
  → Linear(784, 128) → LIF(β=0.95, θ=1.0, soft reset)
  → Linear(128, 128) → LIF
  → Linear(128, 10)  → LIF
  → spike count readout → argmax → classification
```

ok nowFitness: negative cross-entropy on spike counts. All weight matrices optimized via low-rank (rank-3) EGGROLL perturbations with AdamW.

## Quick start

```bash
# Setup
cd ..
git clone https://github.com/ESHyperscale/HyperscaleES
cd spikyeggrolls
uv venv
uv pip install -e ".[cuda13]"
uv pip install -e ../HyperscaleES/

# Optional GPU installs (choose one based on your machine/runtime)
# uv pip install -e ".[cuda12]"
# uv pip install -e ".[cuda13]"

# Train (best config, ~60 min on RTX 4080)
.venv/bin/python -m spikyeggroll.train \
  --pop_size 10000 --rank 3 --sigma 0.007 --lr 0.001 --epochs 4000

# Quick test (~45s)
.venv/bin/python -m spikyeggroll.train \
  --pop_size 1024 --rank 2 --epochs 50

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## Environment Profiles

The repo now supports shared launchers for both local GPU machines and Runpod Pods:

```bash
# Local
make bootstrap-local
make doctor-local
make smoke-local

# Runpod
make bootstrap-runpod
make doctor-runpod
make smoke-runpod
```

Both paths run the same Python training entry point through environment profiles in
`env/local.env` and `env/runpod.env`.

The default Runpod bootstrap path now targets NVIDIA CUDA 13 environments, which
matches current high-end Runpod GPUs better. Use `.[cuda12]` explicitly if you need
an older fallback. For a non-Runpod local setup, override
`SPIKYEGGROLL_INSTALL_TARGET` if you need a different JAX install target.
By default, local bootstrap expects `HyperscaleES` at `../HyperscaleES` while
Runpod bootstrap expects `/workspace/HyperscaleES`; override `HYPERSCALEES_DIR`
if your checkout is elsewhere.

Long-running jobs now emit:

- stdout logs in `LOG_DIR/<run_name>.stdout.log`
- structured metrics in `LOG_DIR/<run_name>.metrics.jsonl`
- a final summary in `LOG_DIR/<run_name>.summary.json`
- checkpoints in `CHECKPOINT_DIR/<run_name>-{last,best,interrupt}.pkl`

Evaluation note: test accuracy is computed in fixed-size chunks using the training
batch size; when `10000 % batch_size != 0`, the final partial chunk is dropped to
avoid extra JIT recompiles.

## Defaults and precedence

Effective runtime settings are resolved in this order:

1. explicit environment variable overrides (for example `POP_SIZE=4096 make tune-runpod`)
2. preset defaults in `scripts/run_train.sh` (`smoke`, `tune`, `full`)
3. CLI defaults in `spikyeggroll/train.py` when running `python -m spikyeggroll.train` directly
4. `SNNConfig` defaults in `spikyeggroll/configs.py` when `train()` is called programmatically

Preset defaults in `scripts/run_train.sh`:

| Preset | pop_size | rank | sigma | lr | epochs | batch_size | chunk_size |
|--------|----------|------|-------|----|--------|------------|------------|
| smoke  | 512      | 2    | 0.02  | 0.005 | 30   | 128        | 256        |
| tune   | 2048     | 2    | 0.01  | 0.005 | 100  | 256        | 512        |
| full   | 10000    | 3    | 0.007 | 0.001 | 4000 | 256        | 1024       |

## Key findings

- **Population size matters most**: pop=1024 → 75%, pop=10000 → 92%. Large populations solve the dead output neuron problem.
- **Low rank is sufficient**: rank=2-3 outperforms rank=8-16 at large pop sizes. Higher rank needs proportionally more population to estimate gradients.
- **1/5th success rule** for adaptive sigma works well short-term but needs a cap for long training (sigma drifts upward).
- **Centered rank fitness shaping** is scale-free and outlier-robust; comparable to z-scoring.
- **No surrogate gradients needed**: EGGROLL's low-rank perturbations provide enough signal to train a 3-layer SNN from random init.

See [docs/baseline-validation.md](docs/baseline-validation.md) for full experiment details, sweep results, and ablation studies.

## Dependencies

- JAX (with GPU support)
- [HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) — EGGROLL evolution strategy framework (clone as sibling path `../HyperscaleES`, or set `HYPERSCALEES_DIR`)
- torch + torchvision (for MNIST download)
- optax

## Runpod

See [RUNPOD.md](RUNPOD.md) for a Pod setup guide, bootstrap script, and smoke/tuning commands.
