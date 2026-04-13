# CIFAR ResNet Handoff

This document summarizes the current CIFAR-10 `spiking_resnet18` state, the
profiling findings from the latest 5090 runs, the main limiting factors, and the
recommended next direction for architecture, training, and hyperparameter work.

It is intended as a handoff artifact for the next implementation cycle.

## Current State

- The project is past the "dead CIFAR model" phase.
- The current `spiking_resnet18` path runs end-to-end on CIFAR-10 and learns
  above chance.
- The best documented CIFAR result so far is **22.33% test accuracy** on:
  - `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412`
- The most recent strong run is on:
  - host `64.228.13.219:61195`
  - RTX 5090 32 GB

The current system is no longer blocked by "it will not run." It is blocked by
slow ES population scoring, weak sample efficiency, and optimizer behavior that
collapses exploration too early.

## Best Current Run

Run:
- `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412`

Config:
- `pop_size=4096`
- `rank=2`
- `sigma=0.006`
- `lr=0.0015`
- `timesteps=16`
- `batch_size=48`
- `chunk_size=96`
- `dtype=bfloat16`
- `augment=True`
- `resnet_channels_base=32`
- `updates_per_epoch=10`
- `fitness_shaping=centered_rank`
- `use_batched_update=True`

Latest checked state:
- `epoch 55`
- `global_update 560`
- GPU util: `99%`
- GPU memory: `~29.46 / 32.6 GiB`
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

Interpretation:
- The run is learning.
- The curve is not monotonic, but it is clearly above chance.
- This slightly exceeds the previously documented `21.18%` best run.

## Effective Data Exposure

Important reminder:
- one logged "epoch" in this loop is not a full data epoch
- one logged epoch is `updates_per_epoch` minibatch ES updates

At `global_update=560` and `batch_size=48`:
- total sampled training examples: about `26,880`
- CIFAR-10 train set size: `50,000`
- effective full train-set exposure: about **0.54 passes**

So `epoch 55` here does not mean 55 full passes over CIFAR. It means the run has
still seen less than one dataset pass worth of sampled images.

## Startup / Profiling Findings

### Host-specific behavior

We observed two distinct behaviors across 5090 hosts:

1. A bad host appeared to "hang" before normal training logs.
2. A good host ran the same heavy configuration successfully.

This means the earlier startup stall should not be treated as a universal
code-level deadlock.

### Instrumentation added

`spikyeggroll/train.py` now supports:
- timestamped startup markers
- explicit `jax.block_until_ready(...)` at traced milestones
- capped device-memory snapshots
- bounded startup trace capture
- optional profiler server

Useful CLI flags now:
- `--profile_mode startup|steady_state|full`
- `--profile_server_port 9999`
- `--profile_max_snapshots 16`
- `--profile_warmup_updates <n>`
- `--profile_updates_window <n>`
- `--profile_eval_once`
- `--profile_trace_dir <dir>` if trace artifacts should live outside `log_dir`

Default artifact layout:
- metrics and summaries under `<log_dir>/`
- memory profiles under `<log_dir>/profiles/<run_name>/`
- traces under `<log_dir>/traces/<run_name>/...`
- `--profile_trace_dir` only relocates the trace artifacts

Relevant commits:
- `fc80b53` `Add startup tracing for CIFAR training hangs`
- `b802030` `Add gated startup memory profiling`
- `db3e1c6` `Add startup trace capture controls`

### Startup timing on the good host

Observed startup milestones:
- `train() begin` at `16:35:39`
- `model init complete` at `16:35:43`
- `dataset loaded` at `16:35:47`
- `start metric written` at `16:35:48`
- first `jit_forward_eval` ready at `16:35:49`
- first `population scores` ready at `16:36:15`
- first `jit_update` ready at `16:36:18`

Approximate startup timings:
- model init: `~3 s`
- dataset load/download path: `~4 s`
- prefetch warmup: `~1 s`
- first eval forward: `~1 s`
- first population score: `~26 s`
- first update: `~3 s`

Main startup conclusion:
- the expensive startup region is the first population-scoring path
- dataset load and prefetch are not the bottleneck

### Logging-path gotcha

For direct `python -m spikyeggroll.train` runs:
- stdout logs were redirected to `/workspace/logs/spikyeggroll/...`
- metrics and checkpoints were still being written to repo-relative defaults:
  - `/workspace/spikyeggrolls/logs/spikyeggroll/...`
  - `/workspace/spikyeggrolls/checkpoints/spikyeggroll/...`

This created temporary confusion because some runs were progressing even when the
initially checked metrics path looked empty.

## Current Bottlenecks

### 1. Population scoring cost

The dominant runtime cost is population scoring, not data loading.

Evidence:
- first population score took about `26 s`
- steady-state update time is about `21.7 s`
- GPU is already compute-bound at `~99%`

This means hardware headroom alone will not solve the main problem.

### 2. Low ES sample efficiency

Even after hours of training:
- effective data exposure is still low
- useful progress per wall-clock hour is limited

This is the core downside of using pure ES on a CIFAR-scale conv SNN.

### 3. Sigma collapse

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
- `raw_score_std` fell from about `0.00775` near epoch 20 to about `0.00024`
  by epoch 55

Interpretation:
- the run still learns slowly after sigma collapse
- but exploration is now effectively clamped
- the current sigma schedule is a likely limiter for later-stage improvement

### 4. Architecture/training mismatch

The current system is still expensive to optimize because it uses a fully
spiking conv pipeline from pixel-level processing onward.

That is a poor match for pure ES on CIFAR:
- early spatial feature extraction is hard
- conv cost is multiplied by population
- we spend too much budget making the hardest part spiking

### 5. Literature mismatch

The repo is still not aligned with the strongest CIFAR SNN recipes:
- current norm is `group`, not BN/BNTT-style normalization
- residual shortcut behavior still needs scrutiny relative to SEW-style
  identity-preserving blocks
- eval/training conventions remain more pragmatic than literature-aligned

## Overall Assessment

### What has improved

- The current CIFAR path no longer collapses at initialization.
- The model learns above chance.
- Profiling now makes startup behavior visible instead of guesswork.
- The heavy configuration is known to run on at least one 5090 host.

### What remains limiting

The main limiting factor is:

**pure ES sample efficiency on a CIFAR conv SNN, made worse by sigma collapsing
too early**

Compressed further:
- population scoring is expensive
- effective dataset exposure is low
- exploration shrinks too aggressively
- the architecture is too expensive in fully spiking form for this optimizer

## Recommended Overall Direction

### 1. Do deeper steady-state profiling next

The current profiling is enough to identify the startup boundary. The next step
should profile steady-state updates, not just startup.

Use real JAX profiling traces:
- `jax.profiler.start_server(port)`
- `jax.profiler.start_trace(log_dir)`
- `jax.profiler.stop_trace()`

Compare:
- one bounded startup trace
- one steady-state trace over 3-5 updates after compile
- one run with test evaluation enabled
- one run with test evaluation skipped

Goal:
- measure how much time is in:
  - `jit_forward_eval`
  - `score_population_chunked`
  - `jit_update`
  - test evaluation
  - host gaps

Decision rule:
- if `score_population_chunked` dominates everything, architecture and ES-forward
  cost are the main bottlenecks
- if host gaps are large, runtime staging is still too weak

### 2. Fix sigma adaptation before scaling population further

Do not treat larger population as the immediate next answer.

The current live run already shows sigma collapsing to the floor, which means
the optimizer is starving exploration.

Next changes should include:
- slower sigma decay
- configurable growth/decay factors
- a higher effective floor for CIFAR runs
- possibly a different target success strategy

This is the clearest live-run limiter.

### 3. Move toward a hybrid ANN/SNN architecture

The most promising architecture direction is:

**regular conv front-end + spiking middle/back-end**

Recommended first hybrid variant:
- regular conv stem
- regular residual stages 1-2
- spiking residual stages 3-4
- spiking readout/classifier

Why:
- early CIFAR feature extraction is the hardest and most expensive part for
  pure spiking convs
- under ES, that cost is multiplied by population
- a hybrid front-end should reduce cost and improve feature quality

Second-best variant:
- regular conv backbone
- spiking head only

This is a better direction than insisting on a fully spiking pixel-to-logit
pipeline if the goal is better CIFAR learning under ES.

### 4. Improve the training recipe

Priority changes:
- keep centered-rank shaping
- make evaluation deterministic for static CIFAR images
- revisit normalization

If the goal is accuracy, the next normalization work should be:
- BatchNorm path first
- BNTT-style temporal BN if needed after that

Residual correctness should also be revisited:
- identity-capable shortcuts matter for deep SNNs
- SEW-style residual behavior should remain the reference point

### 5. Tune after optimizer/architecture fixes, not before

For the current pure-spiking setup:
- keep `pop_size=4096`, `rank=2`, `T=16`, `batch=48`, `chunk=96` as the current
  reference point
- do not scale population further until sigma behavior is fixed

Once optimizer behavior is corrected, compare:
- `rank=1` vs `rank=2`
- `T=16` vs `T=25`
- `batch=48` vs `batch=64` only if throughput still scales
- hybrid front-end before larger pure-spiking models

For a hybrid model, tune in this order:
1. ANN/SNN split point
2. sigma schedule
3. population size
4. timesteps
5. normalization mode

## Practical Summary

If the goal is **better CIFAR training**, the best overall next path is:

1. keep the good 5090 host as the reference machine
2. do deeper steady-state profiling
3. fix sigma adaptation
4. try a hybrid ANN-front / spiking-back architecture
5. add BN/BNTT-style normalization experiments
6. only then revisit larger populations

Short version:
- profiling says the heavy cost is population scoring
- optimizer says exploration is collapsing
- architecture says pure spiking conv early layers are a bad ES target
- the most promising next move is:
  - **hybrid ANN front-end**
  - **better sigma schedule**
  - **better normalization**
