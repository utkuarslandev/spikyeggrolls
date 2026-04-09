# Project: spikyeggroll

Spiking neural networks trained with EGGROLL evolution strategies in JAX.

## Environment

- Use `uv` for virtual environment management (`uv venv`, `uv pip install`, etc.)
- Run Python via `.venv/bin/python`
- HyperscaleES is a local source dependency in `HyperscaleES/` — install with `uv pip install -e HyperscaleES/` or add `HyperscaleES/src` to PYTHONPATH

## Quick start

```bash
# Train MNIST with best config (91.9% in ~6 min on RTX 4080)
.venv/bin/python -m spikyeggroll.train --pop_size 10000 --rank 3 --sigma 0.007 --lr 0.005 --epochs 400

# Quick test run
.venv/bin/python -m spikyeggroll.train --pop_size 1024 --rank 2 --epochs 50

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## Architecture

784-128-128-10 feedforward SNN with LIF neurons, soft reset, Poisson rate coding (T=25).
See `docs/baseline-validation.md` for full experiment details.

## Preferences

- Do NOT pipe command output through grep — the user wants to see live output during long training runs
- Run training commands directly without filtering
- Only commit when something meaningful is achieved, not partial/broken states
