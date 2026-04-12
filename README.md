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

Fitness: negative cross-entropy on spike counts. All weight matrices optimized via low-rank (rank-3) EGGROLL perturbations with AdamW.

## Quick start

```bash
# Setup
uv venv
uv pip install -e ".[cuda13]"

# Optional: other JAX install targets (match your driver / GPU)
# uv pip install -e ".[cuda12]"
# uv pip install -e ".[cpu]"

# Train (best config, ~60 min on RTX 4080)
.venv/bin/python -m spikyeggroll.train \
  --pop_size 10000 --rank 3 --sigma 0.007 --lr 0.001 --epochs 4000

# Quick test (~45s)
.venv/bin/python -m spikyeggroll.train \
  --pop_size 1024 --rank 2 --epochs 50

# CIFAR-10 deep scaling smoke (spiking ResNet18-style model)
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 --epochs 50 \
  --timesteps 16 --batch_size 32 --chunk_size 96 --updates_per_epoch 64 \
  --sigma_max 0.012 --fitness_shaping centered_rank --use_batched_update \
  --dtype bfloat16 --augment --num_test_eval_samples 1024

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

# CIFAR-10 + spiking_resnet18 through launcher
DATASET=cifar10 MODEL_NAME=spiking_resnet18 make smoke-runpod
```

Both paths run the same Python training entry point through environment profiles in
`env/local.env` and `env/runpod.env`.

Local bootstrap defaults to the JAX `cuda13` extra via `env/local.env`, and
`REQUIRE_ACCELERATOR=1` so `make doctor-local` fails if JAX only sees CPU (training
is intended to run on GPU). For CPU-only machines, set `REQUIRE_ACCELERATOR=0` or
install with `.[cpu]`. Runpod also defaults to `cuda13` via `env/runpod.env`. Override
`SPIKYEGGROLL_INSTALL_TARGET` if you need a different JAX wheel.

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
3. explicit CLI flags in `spikyeggroll/train.py` when running `python -m spikyeggroll.train` directly
4. dataset-derived defaults for `n_inputs`, `in_channels`, and `image_size`
5. `SNNConfig` defaults in `spikyeggroll/configs.py`

When no CLI flag is provided, direct `python -m spikyeggroll.train` execution now
inherits baseline defaults from `SNNConfig` instead of a second, partially
duplicated set of argparse defaults.

Preset defaults in `scripts/run_train.sh`:

| Preset | pop_size | rank | sigma | lr | epochs | batch_size | chunk_size |
|--------|----------|------|-------|----|--------|------------|------------|
| smoke  | 512      | 2    | 0.02  | 0.005 | 30   | 128        | 256        |
| tune   | 2048     | 2    | 0.01  | 0.005 | 100  | 256        | 512        |
| full   | 10000    | 3    | 0.007 | 0.001 | 4000 | 256        | 1024       |

For CIFAR-10 deep runs, set `DATASET=cifar10 MODEL_NAME=spiking_resnet18`.
The launcher now applies the throughput-oriented CIFAR defaults used by the
current 5090 experiments:
`TIMESTEPS=16`, `BATCH_SIZE=32`, `CHUNK_SIZE=96`, `UPDATES_PER_EPOCH=64`,
`FITNESS_SHAPING=centered_rank`, `USE_BATCHED_UPDATE=true`,
`DTYPE=bfloat16`, `SIGMA_MAX=0.012`, and `NUM_TEST_EVAL_SAMPLES=1024`.

See [docs/cifar-experiments-log.md](docs/cifar-experiments-log.md) for the
current CIFAR results and failure modes.

Population-scaling sweep example (6 settings, 3 seeds each):

```bash
for POP in 256 512 1024 2048 4096 8192; do
  for SEED in 0 1 2; do
    DATASET=cifar10 MODEL_NAME=spiking_resnet18 \
    POP_SIZE=$POP RANK=2 SIGMA=0.006 LR=0.0015 EPOCHS=50 TIMESTEPS=16 \
    BATCH_SIZE=32 CHUNK_SIZE=96 UPDATES_PER_EPOCH=64 \
    FITNESS_SHAPING=centered_rank USE_BATCHED_UPDATE=true \
    DTYPE=bfloat16 SIGMA_MAX=0.012 NUM_TEST_EVAL_SAMPLES=1024 \
    RUN_NAME="cifar10-resnet18-pop${POP}-s${SEED}" \
    make tune-local -- --seed $SEED
  done
done
```

## Key findings

- **Population size matters most**: pop=1024 → 75%, pop=10000 → 92%. Large populations solve the dead output neuron problem.
- **Low rank is sufficient**: rank=2-3 outperforms rank=8-16 at large pop sizes. Higher rank needs proportionally more population to estimate gradients.
- **1/5th success rule** for adaptive sigma works well short-term but needs a cap for long training (sigma drifts upward).
- **Centered rank fitness shaping** is scale-free and outlier-robust; comparable to z-scoring.
- **No surrogate gradients needed**: EGGROLL's low-rank perturbations provide enough signal to train a 3-layer SNN from random init.

See [docs/baseline-validation.md](docs/baseline-validation.md) for full experiment details, sweep results, and ablation studies.

## Dependencies

- JAX (with GPU support)
- Vendored `hyperscalees` package for the EGGROLL evolution strategy framework
- torch + torchvision (for MNIST download)
- optax

## Vendored Code

This repository vendors the `hyperscalees` package from
`ESHyperscale/HyperscaleES` for the subset of modules used by the SNN training
path. The upstream license text is included at
`third_party/HYPERSCALEES_LICENSE.txt`.

## Runpod

See [RUNPOD.md](RUNPOD.md) for a Pod setup guide, bootstrap script, and smoke/tuning commands.
