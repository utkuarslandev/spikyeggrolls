# Project: spikyeggroll

General guidance for coding agents working in this repository.

## Purpose

- This repo trains spiking neural networks in JAX with the EGGROLL evolution strategy.
- The main code lives under `spikyeggroll/`.
- The vendored `hyperscalees/` package provides the EGGROLL primitives used by this repo.

## Environment

- Use `uv` for environment and package management.
- Run Python via `.venv/bin/python`.
- Install the project with `uv pip install -e ".[cuda12]"`, `.[cuda13]`, or `.[cpu]` depending on the machine.
- `hyperscalees` is vendored in this repository. Do not assume a sibling `../HyperscaleES` checkout exists or is needed.
- Local defaults live in `env/local.env`.
- Runpod defaults live in `env/runpod.env`.

## Common Commands

```bash
# Bootstrap
make bootstrap-local
make bootstrap-runpod

# Environment checks
make doctor-local
make doctor-runpod

# Training presets
make smoke-local
make tune-local
make full-local

# Direct training
.venv/bin/python -m spikyeggroll.train --pop_size 1024 --rank 2 --epochs 50

# Tests
.venv/bin/python -m pytest tests/ -v
```

## Repo Structure

- `spikyeggroll/train.py`: main training entry point
- `spikyeggroll/configs.py`: runtime config dataclass
- `spikyeggroll/models/`: SNN model definitions
- `spikyeggroll/data/`: MNIST and CIFAR-10 loading and spike encoding
- `hyperscalees/`: vendored EGGROLL dependency used by training
- `scripts/`: bootstrap, doctor, and launcher scripts
- `docs/`: experiment notes and validation docs

## Working Norms

- Prefer small, focused changes over broad refactors.
- Keep launcher scripts, packaging, and docs in sync when changing setup or install behavior.
- Preserve the existing local and Runpod workflows unless the task explicitly changes them.
- Avoid introducing new heavyweight dependencies unless they are necessary for the training path.
- If editing vendored `hyperscalees` code, keep changes minimal and scoped to what this repo actually uses.

## Preferences

- Do not pipe long-running command output through `grep` or similar filters.
- Run training commands directly so logs remain visible.
- Do not commit partial or broken states.
- Prefer validation that matches the scope of the change: import checks for packaging changes, targeted tests for code changes, smoke runs for workflow changes.
