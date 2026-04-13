# CIFAR-10 / Spiking ResNet Current Status

This file is the rolling current-status document for the CIFAR
`spiking_resnet18` work.

Use it for:
- the current baseline
- the current best result
- what is implemented and validated
- what we believe so far
- what to run next

For raw chronology, commands, and day-by-day notes, see
[docs/daily-notes.md](docs/daily-notes.md).

## Current Snapshot

- Current active baseline:
  - `cifar5090-phase3-baseline-batch-kernel-20260413`
  - host: `216.249.100.66:21650`
  - source: fresh clone of `utkuarslandev/spikyeggrolls`
  - config:
    - `pop_size=4096`
    - `rank=2`
    - `timesteps=16`
    - `batch_size=48`
    - `chunk_size=96`
    - `dtype=bfloat16`
    - `resnet_norm=batch`
    - `conv_es_mode=kernel_lora`
    - selective stage perturbation enabled
  - finished:
    - `30` logged epochs
    - `300` updates
    - `1783.9s` wall-clock
    - final test accuracy: `28.27%`

- Current best historical CIFAR result:
  - `28.27%` final test accuracy
  - run: `cifar5090-phase3-baseline-batch-kernel-20260413`
  - same run also recorded checkpoint best `26.98%` at `epoch 25`

- Best current Phase 3 baseline test seen so far:
  - final test: `28.27%`
  - checkpoint best: `26.98%` at `epoch 25`
  - run: `cifar5090-phase3-baseline-batch-kernel-20260413`

- Current main bottlenecks:
  - population scoring still dominates wall clock
  - pure-spiking CIFAR learning is still weak relative to compute spent
  - sigma still decays to the floor late in training even on the stronger Phase 3 baseline
  - `bntt` and `matrix_lora` are implemented but not yet benchmarked on the 5090 against the finished baseline

## Active Baselines

| Role | Run | Key Config | Best / Current |
|------|-----|------------|----------------|
| Best historical accuracy | `cifar5090-phase3-baseline-batch-kernel-20260413` | `batch + kernel_lora + selective perturbation` | `28.27%` final test, `26.98%` checkpoint best |
| Pre-Phase 3 reference | `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412` | `batch=48`, `chunk=96`, `bf16`, pre-Phase 3 | `22.33%` best test |
| Current Phase 3 baseline | `cifar5090-phase3-baseline-batch-kernel-20260413` | `batch + kernel_lora + selective perturbation` | `30` epochs, `300` updates, `1783.9s`, final `28.27%` |

## Implementation Status

### Phase 1

Implemented:
- true identity/projection shortcuts
- banded sigma adaptation
- plain BatchNorm with running stats

Validated:
- local targeted and broader test suites passed

### Phase 2

Implemented:
- selective stage perturbation
- stage-group ES masking
- deterministic prefix caching
- selective suffix-only noisy forward

Validated:
- local tests passed
- remote 5090 runtime now clearly shows the selective schedule changing wall-clock cost as intended

### Phase 3

Implemented:
- `resnet_norm=bntt`
- `conv_es_mode=matrix_lora`

Validated:
- local correctness tests passed

Not yet benchmarked remotely:
- `bntt + kernel_lora`
- `batch + matrix_lora`
- `bntt + matrix_lora`

## What We Know

### Training behavior

- The CIFAR conv SNN path is no longer dead or chance-only.
- The model learns above chance on strong 5090 runs.
- The current best historical result is still only `22.33%`, so learning quality remains poor by CIFAR standards.

### Throughput behavior

- On the 5090, the system is generally compute-bound once training is underway.
- Selective perturbation is now a first-order throughput lever, not a cosmetic optimization.

Measured on `cifar5090-phase3-baseline-batch-kernel-20260413`:
- early selective (`stage3 + head`, `cache_split=after_stage2`):
  - `active_param_fraction ≈ 0.274`
  - `population_score_mean_s ≈ 2.35-2.38s`
  - `avg_update_s ≈ 3.6-3.8s`
- full refresh:
  - `active_param_fraction = 1.0`
  - `population_score_mean_s ≈ 15.64-19.25s`
  - `avg_update_s ≈ 15.69-20.77s`
- mid selective (`stage2 + stage3 + head`, `cache_split=after_stage1`):
  - `active_param_fraction ≈ 0.516`
  - early transition epoch: `population_score_mean_s ≈ 6.75s`, `avg_update_s ≈ 7.75s`
  - steady state: `population_score_mean_s ≈ 4.89s`, `avg_update_s ≈ 4.93-4.95s`

This cost ordering is now established:
- early selective < mid selective << full-model refresh

### Finished baseline result

Completed on `216.249.100.66:21650` from a fresh clone of `utkuarslandev/spikyeggrolls`
at commit `94d9afc`.

Final summary:
- `30` logged epochs
- `300` updates
- `1783.9s` total wall-clock, about `29.7 min`
- final test accuracy: `28.27%`
- checkpoint best test accuracy: `26.98%` at `epoch 25`

Observed test checkpoints:
- `epoch 5`: `20.73%`
- `epoch 10`: `23.81%`
- `epoch 15`: `25.89%`
- `epoch 20`: `26.19%`
- `epoch 25`: `26.98%`
- final summary: `28.27%`

Optimizer behavior:
- `sigma` held at `0.006` through the early and most of the mid-selective phase
- late-training decay resumed:
  - `epoch 20`: `0.00543`
  - `epoch 25`: `0.00335`
  - `epoch 28-29`: hit `sigma_min = 0.0025`

Interpretation:
- selective perturbation is operationally validated on the 5090
- the Phase 3 baseline now materially exceeds the earlier `22.33%` pre-Phase 3 result
- sigma collapse remains a real late-training limiter even on the improved baseline

### Profiling conclusions so far

- The dominant steady-state cost is still population scoring.
- Prefix caching cost is small relative to population scoring in the selective phases.
- Eval cost is measurable but not the main throughput problem.

## Open Questions

- Does `bntt` improve equal-wall-clock CIFAR accuracy relative to `batch`?
- Does `matrix_lora` reduce `population_score_mean_s` on the 5090 without causing a memory problem?
- Does `bntt + matrix_lora` beat the new baseline on both:
  - wall-clock efficiency
  - early test accuracy

## Next Runs

Run in this order:

1. `bntt + kernel_lora`
2. `batch + matrix_lora`
3. `bntt + matrix_lora`

Reference config:

```bash
--dataset cifar10 --model_name spiking_resnet18 \
--pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
--epochs 30 --updates_per_epoch 10 \
--timesteps 16 --batch_size 48 --chunk_size 96 \
--augment --dtype bfloat16 \
--resnet_channels_base 32 \
--sigma_warmup_epochs 20 \
--test_interval 5 --checkpoint_interval 10 --log_interval 1 \
--num_test_eval_samples 1024 \
--profile_mode steady_state \
--profile_warmup_updates 5 \
--profile_updates_window 3 \
--profile_eval_once \
--profile_server_port 9999
```

Variants to compare:

- Baseline:
  - `--resnet_norm batch`
  - `--conv_es_mode kernel_lora`
- BNTT only:
  - `--resnet_norm bntt`
  - `--conv_es_mode kernel_lora`
- Matrix-LoRA only:
  - `--resnet_norm batch`
  - `--conv_es_mode matrix_lora`
- Combined:
  - `--resnet_norm bntt`
  - `--conv_es_mode matrix_lora`

## Decision Rules

Track:
- `population_score_mean_s`
- `timing_population_score_frac`
- `avg_update_s`
- peak GPU memory
- `test_acc` at equal wall-clock points
- best test accuracy
- `sigma`
- `raw_score_std`

Success criteria:
- `bntt` is worth keeping if it improves or at least does not hurt early learning at equal wall clock
- `matrix_lora` is worth keeping if it materially lowers population-score cost without breaking memory behavior
- `bntt + matrix_lora` becomes the new baseline only if it improves the overall speed/learning tradeoff, not just one side of it

## Reference Docs

- Chronological notebook:
  - [docs/daily-notes.md](docs/daily-notes.md)
- Long-range implementation roadmap:
  - [docs/cifar-resnet-phased-implementation-plan.md](docs/cifar-resnet-phased-implementation-plan.md)
