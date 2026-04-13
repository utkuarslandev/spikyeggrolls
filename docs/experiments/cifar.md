# CIFAR-10 Experiments

This is the primary source of truth for the CIFAR `spiking_resnet18` work.

Use it for:
- current best and active runs
- completed comparisons
- confirmed runtime and training findings
- next experiments

For chronological notes, see [../daily-notes.md](../daily-notes.md). For the
current implementation roadmap, see
[../roadmaps/cifar-resnet-roadmap.md](../roadmaps/cifar-resnet-roadmap.md).

## Current Snapshot

- Current best result:
  - run: `cifar5090-phase3-baseline-batch-kernel-20260413`
  - host: `216.249.100.66:21650`
  - result:
    - final test accuracy: `28.27%`
    - best checkpoint test accuracy: `26.98%` at `epoch 25`
    - `30` logged epochs, `300` updates, `1783.9s`
- Current active long run:
  - run: `cifar5090-2h-batch-kernel-selective-sigmafix`
  - host: `216.249.100.66:21650`
  - status at last check:
    - `epoch 6`
    - `global_update 70`
    - `epoch 5` test accuracy: `21.53%`
    - sigma still holding at `0.006`
    - early-selective throughput improved with profiling off
- Current default baseline configuration:
  - `pop_size=4096`
  - `rank=2`
  - `timesteps=16`
  - `batch_size=48`
  - `chunk_size=96`
  - `dtype=bfloat16`
  - `resnet_norm=batch`
  - `conv_es_mode=kernel_lora`
  - selective stage perturbation enabled

## Best Runs

| Role | Run | Key Config | Result |
|------|-----|------------|--------|
| Current best overall | `cifar5090-phase3-baseline-batch-kernel-20260413` | `batch + kernel_lora + selective perturbation` | final `28.27%`, checkpoint best `26.98%` |
| Best pre-Phase-3 reference | `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412` | `batch=48`, `chunk=96`, `bf16` | best `22.33%` |
| BNTT comparison | `cifar5090-phase3-bntt-kernel` | `bntt + kernel_lora + selective perturbation` | final `28.17%`, checkpoint best `26.39%` |
| Matrix-LoRA attempt | `cifar5090-phase3-batch-matrix` | `batch + matrix_lora + selective perturbation` | killed at `epoch 8`; severe runtime regression |

## Implementation Status

### Implemented

- Phase 1:
  - true identity/projection shortcuts
  - banded sigma adaptation
  - plain BatchNorm with running stats
- Phase 2:
  - selective stage perturbation
  - stage-group ES masking
  - deterministic prefix caching
  - selective suffix-only noisy forward
- Phase 3:
  - `resnet_norm=bntt`
  - `conv_es_mode=matrix_lora`

### Locally validated

- Phase 1, 2, and 3 correctness changes passed the targeted local test suites.

### Remotely benchmarked

- `batch + kernel_lora`
- `bntt + kernel_lora`
- `batch + matrix_lora`

### Not yet worth benchmarking

- `bntt + matrix_lora`
  - blocked until `matrix_lora` itself becomes competitive

## Confirmed Findings

### Throughput

- The dominant steady-state cost is still population scoring.
- Selective perturbation is a first-order throughput lever, not a cosmetic optimization.
- Prefix caching cost is small relative to population scoring in selective phases.
- Eval cost is measurable but not the main bottleneck.

Measured on `cifar5090-phase3-baseline-batch-kernel-20260413`:
- early selective (`stage3 + head`, `after_stage2`):
  - `active_param_fraction ≈ 0.274`
  - `population_score_mean_s ≈ 2.35-2.38s`
  - `avg_update_s ≈ 3.6-3.8s`
- mid selective (`stage2 + stage3 + head`, `after_stage1`):
  - steady-state `population_score_mean_s ≈ 4.89s`
  - steady-state `avg_update_s ≈ 4.93-4.95s`
- full refresh:
  - `population_score_mean_s ≈ 15.64-19.25s`
  - `avg_update_s ≈ 15.69-20.77s`

Runtime ordering is now established:
- early selective < mid selective << full-model refresh

### Training

- The CIFAR conv SNN path is no longer dead or chance-only.
- The Phase 3 baseline materially improved over the pre-Phase-3 best (`22.33%` -> `28.27%`).
- `bntt` does not clearly beat `batch` in the current ES regime.
- Sigma collapse remains a real late-training limiter.

### Systems

- The current `matrix_lora` implementation is not a speed optimization.
- The likely loser is the patch-extraction delta path, not VRAM capacity.

## Completed Comparisons

### `batch + kernel_lora`

Run:
- `cifar5090-phase3-baseline-batch-kernel-20260413`

Result:
- `30` epochs
- `300` updates
- `1783.9s`
- final test `28.27%`
- checkpoint best `26.98%` at `epoch 25`

Observed test checkpoints:
- `epoch 5`: `20.73%`
- `epoch 10`: `23.81%`
- `epoch 15`: `25.89%`
- `epoch 20`: `26.19%`
- `epoch 25`: `26.98%`
- final summary: `28.27%`

Optimizer behavior:
- sigma held at `0.006` through most of training
- late decay resumed:
  - `epoch 20`: `0.00543`
  - `epoch 25`: `0.00335`
  - `epoch 28-29`: hit `sigma_min = 0.0025`

### `bntt + kernel_lora`

Run:
- `cifar5090-phase3-bntt-kernel`

Result:
- `30` epochs
- `300` updates
- `1760.3s`
- final test `28.17%`
- checkpoint best `26.39%` at `epoch 25`

Observed test checkpoints:
- `epoch 10`: `23.91%`
- `epoch 15`: `24.80%`
- `epoch 20`: `25.60%`
- `epoch 25`: `26.39%`
- final summary: `28.17%`

Comparison against the baseline:
- early selective:
  - baseline `population_score_mean_s ≈ 2.37s`
  - `bntt ≈ 2.42s`
- mid selective:
  - baseline `≈ 4.89s`
  - `bntt ≈ 4.94s`
- full refresh:
  - baseline `≈ 15.64-19.25s`
  - `bntt ≈ 15.59-17.13s`

Takeaway:
- essentially a wash
- slightly faster overall, slightly worse on both checkpoint-best and final test

## Failed Comparisons

### `batch + matrix_lora`

Run:
- `cifar5090-phase3-batch-matrix`

Progress before stop:
- killed at `epoch 8`
- first test at `epoch 5`: `20.93%`
- GPU stayed compute-bound
- GPU memory stayed stable around `24.65 / 32.6 GiB`

Comparison against the baseline:
- early selective:
  - baseline `population_score_mean_s ≈ 2.37s`
  - `matrix_lora ≈ 7.47s`
- full refresh:
  - baseline `≈ 15.64-19.25s`
  - `matrix_lora ≈ 191.4s`
- full-refresh total update mean:
  - baseline `≈ 15.69-20.77s`
  - `matrix_lora ≈ 193.1s`

Takeaway:
- about `3x` slower in early selective
- about `10x+` slower in full refresh
- not worth layering `bntt` on top of the current implementation

## Active Run

Current long run:
- `cifar5090-2h-batch-kernel-selective-sigmafix`
- same proven baseline core with a less collapse-prone sigma schedule:
  - `sigma_warmup_epochs=40`
  - `sigma_min=0.0035`
  - `sigma_target_success=0.12`
  - `sigma_success_tolerance=0.05`
  - `sigma_decay=0.995`
  - `sigma_growth=1.01`
  - `profile_mode=off`

Latest observed state:
- `epoch 6`
- `global_update 70`
- `epoch 5` test accuracy: `21.53%`
- sigma still holding at `0.006`
- early-selective throughput improved with profiling off:
  - `avg_update_s ≈ 2.76s`
  - `updates_per_s ≈ 0.36`

## Next Steps

Priority order:
1. let the active 2-hour sigma-tuned run finish
2. compare equal-wall-clock accuracy against the current `28.27%` baseline
3. keep `batch + kernel_lora` as the default baseline unless the 2-hour run wins
4. if `matrix_lora` is revisited, narrow it first:
   - selective phases only
   - late-stage convs only
   - ideally `1x1` convs first

## References

- Chronological notes:
  - [../daily-notes.md](../daily-notes.md)
- Live roadmap:
  - [../roadmaps/cifar-resnet-roadmap.md](../roadmaps/cifar-resnet-roadmap.md)
- Archived handoff:
  - [../archive/cifar-handoff.md](../archive/cifar-handoff.md)
- Archived pure-spiking ideation:
  - [../archive/pure-spiking-cifar-optimization-notes.md](../archive/pure-spiking-cifar-optimization-notes.md)
- Environment and pod setup:
  - [../getting-started/runpod.md](../getting-started/runpod.md)
