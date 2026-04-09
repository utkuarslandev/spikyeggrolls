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

Fitness: negative cross-entropy on spike counts. All weight matrices optimized via low-rank (rank-3) EGGROLL perturbations with Adam.

## Quick start

```bash
# Setup
git clone https://github.com/ESHyperscale/HyperscaleES
uv venv && uv pip install -e . && uv pip install -e HyperscaleES/

# Train (best config, ~60 min on RTX 4080)
.venv/bin/python -m spikyeggroll.train \
  --pop_size 10000 --rank 3 --sigma 0.007 --lr 0.001 --epochs 4000

# Quick test (~45s)
.venv/bin/python -m spikyeggroll.train \
  --pop_size 1024 --rank 2 --epochs 50

# Run tests
.venv/bin/python -m pytest tests/ -v
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
- [HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) — EGGROLL evolution strategy framework (clone into repo root)
- torchvision (for MNIST download)
- optax
