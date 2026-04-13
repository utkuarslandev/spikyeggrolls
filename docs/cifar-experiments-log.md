# CIFAR-10 / Spiking ResNet Experiment Log

This document tracks the CIFAR-10 `spiking_resnet18` runs performed after the
from-scratch convolutional ResNet path replaced the earlier residual MLP path.

## Summary

- The current conv SNN path no longer collapses immediately.
- CIFAR learning is now above chance on the stronger 5090 runs, but still weak.
- Best observed test accuracy so far: **21.18%** on the 5090 at epoch 80 with
  `pop_size=4096`, `rank=2`, `timesteps=16`, `batch_size=32`,
  `chunk_size=96`, `resnet_channels_base=32`.
- The main failure mode has shifted from "dead network" to "slow, unstable
  learning with sigma drift."
- The implementation roadmap for fixing the remaining CIFAR issues now lives in
  [docs/cifar-resnet-phased-implementation-plan.md](docs/cifar-resnet-phased-implementation-plan.md).

## Run History

### 1. 4080 smoke config from pulled GitHub repo: OOM

Hardware:
- RTX 4080 16 GB

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 256 --rank 2 --sigma 0.01 --lr 0.002 --epochs 3 \
  --timesteps 8 --batch_size 64 --chunk_size 128
```

Result:
- Failed with GPU OOM.
- JAX attempted to allocate about `30.26 GiB`.

Takeaway:
- README-style CIFAR smoke settings do not fit a 16 GB 4080 for the current
  conv ResNet path.

### 2. 4080 reduced smoke: runs, but at chance

Hardware:
- RTX 4080 16 GB

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 32 --rank 2 --sigma 0.01 --lr 0.002 \
  --epochs 1 --timesteps 4 --batch_size 8 --chunk_size 8 \
  --num_test_eval_samples 64 --log_interval 1 --test_interval 1 \
  --checkpoint_interval 0 --run_name ssh-resnet-smoke-small
```

Result:
- Final test accuracy: `0.1094`
- `better: 13/32`
- `std: 0.07162`

Takeaway:
- The model runs end-to-end and produces nonzero ES signal.
- Accuracy is still effectively chance.

### 3. 4080 5-epoch learn check: not dead, not learning

Hardware:
- RTX 4080 16 GB

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 32 --rank 2 --sigma 0.01 --lr 0.002 \
  --epochs 5 --timesteps 4 --batch_size 8 --chunk_size 8 \
  --num_test_eval_samples 256 --log_interval 1 --test_interval 1 \
  --checkpoint_interval 0 --run_name ssh-resnet-learncheck
```

Result:
- Test accuracy history: about `11.33% -> 8.98% -> 9.77% -> 10.16% -> 9.77%`

Takeaway:
- No collapse.
- No meaningful CIFAR learning.

### 4. 5090 short foreground smoke: still chance-level

Hardware:
- RTX 5090 32 GB

Run name:
- `cifar-5090-resnet`

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 64 --rank 2 --sigma 0.01 --lr 0.002 \
  --epochs 3 --timesteps 4 --batch_size 16 --chunk_size 16 \
  --num_test_eval_samples 256 --log_interval 1 --test_interval 1 \
  --checkpoint_interval 0 --run_name cifar-5090-resnet
```

Result:
- Best/final test accuracy: `11.33%`

Takeaway:
- More VRAM alone does not fix learning.

### 5. 5090 first long background run: active but mostly chance

Hardware:
- RTX 5090 32 GB

Run name:
- `cifar-5090-fullresnet-bg`

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 128 --rank 2 --sigma 0.01 --lr 0.002 \
  --epochs 100 --timesteps 8 --batch_size 16 --chunk_size 16 \
  --log_interval 1 --test_interval 10 --checkpoint_interval 10 \
  --run_name cifar-5090-fullresnet-bg
```

Recorded metrics:
- Best test accuracy: `11.48%` at epoch 50
- Last recorded test accuracy: `11.44%` at epoch 90

Takeaway:
- The conv SNN path is active and non-degenerate.
- `pop=128`, `rank=2`, `T=8` is still far too weak for CIFAR.

### 6. 5090 population-scaled screening runs

#### 6a. `pop=1024`, `rank=1`, `T=16`, width 32

Run name:
- `cifar5090-r1-p1024-t16-c32-20260411`

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 1024 --rank 1 --sigma 0.006 --lr 0.0015 \
  --epochs 300 --timesteps 16 --batch_size 16 --chunk_size 64 \
  --augment --resnet_channels_base 32 --sigma_warmup_epochs 20 \
  --test_interval 10 --checkpoint_interval 25 \
  --run_name cifar5090-r1-p1024-t16-c32-20260411
```

Observed before replacement:
- First test accuracy: `12.06%` at epoch 0

Takeaway:
- Directionally better than the earlier small-pop runs, but too early to trust.

#### 6b. `pop=2048`, `rank=1`, `T=16`, width 32

Run name:
- `cifar5090-r1-p2048-t16-c32-20260411`

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 2048 --rank 1 --sigma 0.006 --lr 0.0015 \
  --epochs 300 --timesteps 16 --batch_size 16 --chunk_size 64 \
  --augment --resnet_channels_base 32 --sigma_warmup_epochs 20 \
  --test_interval 10 --checkpoint_interval 25 \
  --run_name cifar5090-r1-p2048-t16-c32-20260411
```

Observed before replacement:
- First test accuracy: `11.54%` at epoch 0

Takeaway:
- More population alone did not immediately unlock strong learning.

### 7. 5090 aggressive high-pop run, intermediate batch

Hardware:
- RTX 5090 32 GB

Run name:
- `cifar5090-r2-p4096-t16-c32-b24-k96-20260411`

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 300 --timesteps 16 --batch_size 24 --chunk_size 96 \
  --augment --resnet_channels_base 32 --sigma_warmup_epochs 20 \
  --test_interval 10 --checkpoint_interval 25 \
  --run_name cifar5090-r2-p4096-t16-c32-b24-k96-20260411
```

Observed before replacement:
- Fit in memory and reached about `17 GiB` GPU usage.
- Replaced before a useful test series was collected.

Takeaway:
- The 5090 has plenty of headroom for a much more aggressive ES run.

### 8. 5090 current best run so far

Hardware:
- RTX 5090 32 GB

Run name:
- `cifar5090-r2-p4096-t16-c32-b32-k96-20260411`

Command:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 300 --timesteps 16 --batch_size 32 --chunk_size 96 \
  --augment --resnet_channels_base 32 --sigma_warmup_epochs 20 \
  --test_interval 10 --checkpoint_interval 25 \
  --run_name cifar5090-r2-p4096-t16-c32-b32-k96-20260411
```

Metrics snapshot as of April 11, 2026:
- Latest epoch seen: `109`
- Best test accuracy: `21.18%` at epoch 80
- Last recorded test accuracy: `19.00%` at epoch 100
- Latest sigma: `0.0213`
- Latest raw score std: `0.0240`
- Current GPU usage during training: about `17.0 / 32.6 GiB`
- GPU utilization: `99%`

Takeaway:
- This is the first run clearly above chance.
- The model is learning something, but still poorly by CIFAR standards.
- Performance has become unstable after sigma drifted upward.

## Findings

### What improved

- The from-scratch conv ResNet path is no longer dead on arrival.
- CIFAR test accuracy now rises materially above chance on the stronger 5090 run.
- `augment=True`, higher population, and `timesteps=16` help more than the old
  small-population `T=4-8` runs.

### What is still broken or weak

- Learning is still very poor relative to CIFAR expectations.
- The current training loop defines one "epoch" as one minibatch ES update, so
  the run has seen far fewer effective training passes than the epoch count
  suggests.
- Adaptive sigma has no upper cap and drifts upward during long runs.
- The training loop still uses z-score fitness shaping, not centered-rank
  shaping from the stronger MNIST experiments.
- The current conv path uses group norm for eval stability, but this diverges
  from the BN/BNTT-style normalization used in stronger from-scratch CIFAR SNN
  work.

### Hardware / throughput findings

- The 5090 run is already compute-bound at `99%` GPU utilization.
- The spare headroom is memory headroom, not idle GPU time.
- Increasing batch size or chunk size can improve memory use and sometimes wall
  clock efficiency, but it does not automatically produce faster learning.
- The current training loop pays overhead for chunked population evaluation in a
  Python loop and still uses the original per-parameter update replay path.

## Current Conclusion

The current CIFAR `spiking_resnet18` implementation is no longer collapsing, and
it can learn above chance with enough population and a smaller width. However,
the full system is still bottlenecked by training-regime issues:

- one minibatch update per logged epoch
- uncapped sigma adaptation
- z-score shaping instead of centered-rank shaping
- group norm instead of BN/BNTT-style normalization
- low ES sample efficiency for a CIFAR-scale conv model

The next steps should focus on the training rule and throughput, not just larger
VRAM occupancy.

## April 12 Addendum — Startup Stall Investigation and New Heavy 5090 Run

### 9. Heavy 5090 startup-stall investigation across hosts

#### 9a. Host that appeared to hang before training

Hardware:
- RTX 5090 32 GB

Host:
- `216.249.100.66:21657`

Configs attempted:
- `float32`, `batch_size=64`, `chunk_size=128`
- `bfloat16`, `batch_size=48`, `chunk_size=128`
- `bfloat16`, `batch_size=48`, `chunk_size=96`

Observed behavior:
- Process stayed alive for hours.
- GPU stayed at about `97-99%` utilization.
- GPU memory stayed pinned around `29.4 / 32.6 GiB`.
- Log advanced only with JAX/XLA warnings.
- No normal training progress appeared in the expected logs.

Takeaway:
- This looked like a startup/compile stall, not a normal slow first epoch.
- The behavior was severe enough to justify explicit startup tracing.

#### 9b. Instrumentation added for startup debugging

Changes added to `spikyeggroll/train.py` and pushed to GitHub:
- timestamped startup markers around:
  - model init
  - noiser init
  - dataset load
  - JIT wrapper creation
  - prefetch warmup
  - start-metric write
  - first eval forward
  - first population scoring step
  - first update
- `jax.block_until_ready(...)` on traced milestones
- gated device-memory snapshots during startup
- bounded startup trace capture
- optional live profiler server

Relevant commits:
- `fc80b53` `Add startup tracing for CIFAR training hangs`
- `b802030` `Add gated startup memory profiling`
- `db3e1c6` `Add startup trace capture controls`

#### 9c. Important logging-path discovery

For direct `python -m spikyeggroll.train` runs:
- stdout logs were being redirected to `/workspace/logs/spikyeggroll/...`
- metrics and checkpoints were still being written to repo-relative defaults:
  - `/workspace/spikyeggrolls/logs/spikyeggroll/...`
  - `/workspace/spikyeggrolls/checkpoints/spikyeggroll/...`

Takeaway:
- Some early “missing metrics” conclusions were path mismatches rather than
  proof that the run had not progressed.

### 10. Heavy traced run on a second 5090 host: startup clears

Hardware:
- RTX 5090 32 GB

Host:
- `64.228.13.219:61195`

Run name:
- `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412`

Command shape:
```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 300 --timesteps 16 --batch_size 48 --chunk_size 96 \
  --augment --resnet_channels_base 32 --sigma_warmup_epochs 20 \
  --test_interval 5 --checkpoint_interval 10 --log_interval 1 \
  --dtype bfloat16 \
  --run_name cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412
```

Tracing/profiling env:
- `SPIKYEGGROLL_TRACE_STARTUP=1`
- `SPIKYEGGROLL_PROFILE_STARTUP=1`
- `SPIKYEGGROLL_PROFILE_MAX_SNAPSHOTS=16`
- `SPIKYEGGROLL_PROFILE_TRACE=1`
- `SPIKYEGGROLL_PROFILE_SERVER_PORT=9999`

Observed startup milestones:
- `train() begin` at `16:35:39`
- `model init complete` at `16:35:43`
- `dataset loaded` at `16:35:47`
- `start metric written` at `16:35:48`
- first `jit_forward_eval` ready at `16:35:49`
- first `population scores` ready at `16:36:15`
- first `jit_update` ready at `16:36:18`
- bounded startup trace stopped at `16:37:02`

Approximate startup timings:
- model init: `~3 s`
- dataset load/download path: `~4 s`
- prefetch warmup: `~1 s`
- first eval forward: `~1 s`
- first population scoring step: `~26 s`
- first update: `~3 s`

Profiler state:
- live profiler server listening on `:9999`
- bounded startup trace directory created
- startup memory snapshots saved

Takeaway:
- The heavy CIFAR config is not fundamentally deadlocking.
- The earlier “stall” was at least partly host/runtime specific.
- The dominant startup cost is first population scoring, not dataset load or
  prefetch.

### 11. Ongoing heavy 5090 run with traced startup

Hardware:
- RTX 5090 32 GB

Host:
- `64.228.13.219:61195`

Run name:
- `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412`

Latest checked state:
- `epoch 55`
- `global_update 560`
- GPU utilization: `99%`
- GPU memory: about `29.46 / 32.6 GiB`
- average update time: `~21.74 s`
- average epoch time: `~217.4 s`

Test accuracy history:
- `epoch 0`: `10.01%`
- `epoch 5`: `12.65%`
- `epoch 10`: `16.26%`
- `epoch 15`: `19.70%`
- `epoch 20`: `22.33%`
- `epoch 25`: `20.04%`
- `epoch 30`: `20.69%`
- `epoch 35`: `20.85%`
- `epoch 40`: `21.01%`
- `epoch 45`: `21.23%`
- `epoch 50`: `21.39%`
- `epoch 55`: `22.03%`

Best observed on this run:
- `22.33%` at `epoch 20`

Comparison to prior best:
- This slightly exceeds the earlier documented `21.18%` best from
  `cifar5090-r2-p4096-t16-c32-b32-k96-20260411`.

### 12. Sigma / exploration behavior on the new heavy run

Observed sigma collapse:
- `epoch 20`: `0.00492`
- `epoch 21`: `0.00404`
- `epoch 22`: `0.00331`
- `epoch 23`: `0.00272`
- `epoch 24`: `0.00223`
- `epoch 25`: `0.00183`
- `epoch 27`: `0.00123`
- `epoch 55`: `0.00100` (floor)

Observed exploration shrinkage:
- `raw_score_std` fell from about `0.00775` near epoch 20 to
  `0.00024` by epoch 55

Takeaway:
- The run continues to improve slowly even after sigma collapses.
- Exploration is now effectively clamped, so the current sigma schedule remains
  a likely limiter for later-stage learning.

### 13. Effective dataset exposure reminder

At `global_update=560` with `batch_size=48`:
- total sampled training examples: about `26,880`
- relative to CIFAR-10 train size (`50,000`): about `0.54` effective full passes

Important interpretation:
- logged “epoch” in this training loop is not a full dataset epoch
- `epoch 55` here means `560` minibatch ES updates, not `55` complete passes

## Updated Conclusion

The startup-stall investigation changed the diagnosis:

- the heavy CIFAR config can run and train on at least one 5090 host
- the earlier “hang” should be treated as host/runtime specific until proven
  otherwise
- the main startup cost is the first population-scoring path
- the current best documented CIFAR result is now **22.33%**
- the most obvious remaining optimizer issue in long runs is sigma collapsing to
  the floor and starving exploration
