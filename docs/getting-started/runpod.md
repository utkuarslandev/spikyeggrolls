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

By default the bootstrap script creates `.venv`, tries the GPU wheel targets in
order (`cuda13` then `cuda12`), and then runs a hard-failing doctor check that
requires a non-CPU JAX device.

The install targets are currently pinned to JAX `0.9.2`. That pin is deliberate:
the CIFAR selective-perturbation ResNet path has been stable there, while replaying
the same runs under JAX `0.10.0` reproduced an XLA layout/reshape crash.

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

## Session checklist (cost + speed)

Use this loop for daily iteration:

1. Start the pod in Runpod.
2. Connect over SSH.
3. Sync code and validate environment.
4. Run smoke or tune.
5. Stop or terminate the pod when done.

Example command chain after SSH:

```bash
cd /workspace/spikyeggrolls
git pull --ff-only
make doctor-runpod
make smoke-runpod
```

For long runs, use `tmux` so disconnects do not kill training:

```bash
tmux new -s train
cd /workspace/spikyeggrolls
make tune-runpod
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t train`.

Cost controls that matter most:

- Stop the pod when no training is active.
- Use a cheaper GPU tier for smoke tests; move up only for long runs.
- Keep repo, venv, data, and JAX cache under `/workspace` to avoid repeated bootstrap/JIT cost.

## Environment

The `runpod` profile sets these defaults:

```bash
export JAX_COMPILATION_CACHE_DIR=/workspace/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export REQUIRE_ACCELERATOR=1
```

These defaults keep the JAX compilation cache warm while preferring preallocated
HBM on dedicated GPUs. Override `XLA_PYTHON_CLIENT_PREALLOCATE=false` when you
need the older iterative memory behavior for debugging or tight-fit experiments.

## Common overrides

Check the environment before running:

```bash
cd /workspace/spikyeggrolls
make doctor-runpod
```

Force the older JAX CUDA 12 wheels if a Pod image or driver stack needs them:

```bash
SPIKYEGGROLL_INSTALL_TARGET='.[cuda12]' make bootstrap-runpod
```

Override training parameters for a run:

```bash
POP_SIZE=4096 RANK=3 EPOCHS=200 CHUNK_SIZE=1024 make tune-runpod
```

Resume from the most recent checkpoint:

```bash
RUN_NAME=my-run RESUME_FROM=/workspace/checkpoints/spikyeggroll/my-run-last.pkl make tune-runpod
```

## Faster SSH workflow from your laptop

### Option A: SSH config alias

Add an alias to `~/.ssh/config` and update only the port when the pod restarts:

```sshconfig
Host runpod-spiky
    HostName <runpod-public-ip-or-hostname>
    User <runpod-ssh-user>
    Port <runpod-ssh-port>
    IdentityFile <path-to-private-key>
    StrictHostKeyChecking accept-new
```

Then connect with:

```bash
ssh runpod-spiky
```

### Option B: Make targets over SSH

This repo includes `scripts/pod_exec.sh` and helper targets:

```bash
cp env/runpod.ssh.env.example env/runpod.ssh.env
# edit host/port/key for your current pod

make pod-pull
make pod-doctor
make pod-smoke
```

Available remote targets:

- `make pod-pull`
- `make pod-doctor`
- `make pod-smoke`
- `make pod-tune`
- `make pod-full`

`env/runpod.ssh.env` is gitignored. Keep SSH endpoint details there or in shell env vars.

## Notes

- `hyperscalees` is vendored in this repository, so no sibling checkout is required.
- `torch` and `torchvision` are required because MNIST loading is implemented via
  `torchvision.datasets.MNIST`.
- Runpod now defaults to auto-detecting the JAX GPU wheel target (`cuda13` then `cuda12`).
  Use `SPIKYEGGROLL_INSTALL_TARGET='.[cuda12]'` or `'.[cuda13]'` to force one.
- The current `cuda12`, `cuda13`, and `cpu` install targets all pin JAX to `0.9.2`.
- The Runpod bootstrap now fails if JAX only sees CPU devices. That is intentional.
- The default full run is expensive. Use `make smoke-runpod` first to verify the Pod,
  CUDA, and dataset path before scaling up.
