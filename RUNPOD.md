# Runpod

This repo uses a shared environment-profile workflow for both local GPU machines and
Runpod Pods. The Pod-specific profile lives in `env/runpod.env`.

## Recommended Pod

- Cloud: `Community Cloud` for cheap iteration, `Secure Cloud` only if you need network volumes
- GPU: `RTX 4090`, `A40`, or `L40S`
- Template: `Runpod PyTorch`
- Container disk: `20-30 GB`
- Volume disk: `50-100 GB`
- Connection: `SSH` or `VSCode/Cursor`

Keep the repo, virtualenv, MNIST dataset, and JAX compile cache under `/workspace`
so Pod restarts preserve useful state.

## One-time bootstrap

From the Pod terminal:

```bash
cd /workspace
git clone <your-spikyeggrolls-repo-url> spikyeggrolls
cd spikyeggrolls
make bootstrap-runpod
```

By default the bootstrap script clones `HyperscaleES` into `/workspace/HyperscaleES`,
creates `.venv`, installs this repo with the `cuda12` extra, installs
`HyperscaleES`, and then runs a hard-failing doctor check that requires a non-CPU
JAX device.

## Run commands

Smoke test:

```bash
cd /workspace/spikyeggrolls
make smoke-runpod
```

Medium tuning run:

```bash
cd /workspace/spikyeggrolls
make tune-runpod
```

Long run:

```bash
cd /workspace/spikyeggrolls
make full-runpod
```

All run scripts write logs under `/workspace/logs/spikyeggroll/`.
Each run also writes structured metrics and checkpoints using the shared run name.

## Environment

The `runpod` profile sets these defaults:

```bash
export JAX_COMPILATION_CACHE_DIR=/workspace/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export REQUIRE_ACCELERATOR=1
```

These defaults reduce repeated JIT compile cost and make GPU memory behavior less
aggressive during iteration.

## Common overrides

Check the environment before running:

```bash
cd /workspace/spikyeggrolls
make doctor-runpod
```

Override the sibling `HyperscaleES` checkout:

```bash
HYPERSCALEES_DIR=/workspace/custom/HyperscaleES make bootstrap-runpod
```

Override training parameters for a run:

```bash
POP_SIZE=4096 RANK=3 EPOCHS=200 CHUNK_SIZE=1024 make tune-runpod
```

Resume from the most recent checkpoint:

```bash
RUN_NAME=my-run RESUME_FROM=/workspace/checkpoints/spikyeggroll/my-run-last.pkl make tune-runpod
```

## Notes

- `HyperscaleES` is still an external dependency. The bootstrap path assumes a
  sibling checkout at `/workspace/HyperscaleES`.
- `torch` and `torchvision` are required because MNIST loading is implemented via
  `torchvision.datasets.MNIST`.
- The Runpod bootstrap now fails if JAX only sees CPU devices. That is intentional.
- The default full run is expensive. Use `make smoke-runpod` first to verify the Pod,
  CUDA, and dataset path before scaling up.
