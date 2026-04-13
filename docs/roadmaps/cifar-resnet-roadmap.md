# CIFAR ResNet Roadmap

This is the live roadmap for improving the CIFAR `spiking_resnet18` path under
the current pure-EGGROLL, single-GPU setup.

For current results and run-by-run comparisons, see
[../experiments/cifar.md](../experiments/cifar.md).

## Current State

- Best completed result so far:
  - `cifar5090-phase3-baseline-batch-kernel-20260413`
  - final test accuracy `28.27%`
- Current strongest baseline:
  - `batch + kernel_lora + selective stage perturbation`
  - `pop_size=4096`, `rank=2`, `timesteps=16`, `batch_size=48`, `chunk_size=96`, `dtype=bfloat16`
- Main remaining bottlenecks:
  - population scoring still dominates wall clock
  - sigma still collapses late in training
  - `bntt` does not clearly beat `batch`
  - current full-model `matrix_lora` is a severe runtime regression

## Implemented Milestones

### Phase 1 complete

- true identity/projection residual shortcuts
- banded sigma adaptation
- plain BatchNorm with running stats

### Phase 2 complete

- selective stage perturbation
- stage-group ES masking
- deterministic prefix caching
- selective suffix-only noisy forward

### Phase 3 partially complete

Implemented:
- `resnet_norm=bntt`
- `conv_es_mode=matrix_lora`

Benchmarked:
- `bntt + kernel_lora`
  - essentially a wash against the `batch` baseline
- `batch + matrix_lora`
  - severe runtime regression

## Remaining Priorities

### 1. Validate the long-run sigma-tuned baseline

Target:
- determine whether the 2-hour `batch + kernel_lora` run with slower sigma decay
  materially beats the 30-epoch baseline at equal wall-clock or final accuracy

Decision gate:
- if it wins, make the sigma-tuned config the new baseline
- if it does not, keep the current baseline and revisit sigma policy again

### 2. Improve exploration without hurting the working runtime path

Most promising near-term work:
- continue refining the sigma schedule on top of `batch + kernel_lora`
- preserve the current selective perturbation schedule unless a better schedule
  is justified by benchmark data

Constraints:
- do not sacrifice the current `kernel_lora` throughput win for speculative optimizer changes

### 3. Salvage `matrix_lora` narrowly or stop pursuing it

Do not retry:
- full-model `matrix_lora`
- `bntt + matrix_lora` on the current implementation

If revisited, scope it down first:
- selective phases only
- late-stage convs only
- ideally `1x1` convs first

Benchmark gate:
- a narrowed `matrix_lora` variant must beat `kernel_lora` on `population_score_mean_s`
  before it is considered a valid candidate again

### 4. Revisit normalization only if it beats the baseline in wall-clock terms

Current conclusion:
- `bntt` is not a clear improvement in the current ES regime

Future work here only makes sense if:
- a revised training regime changes the tradeoff, or
- a different timestep/optimizer schedule makes `bntt` more useful

## Known Regressions / Dead Ends

### Full-model `matrix_lora`

Observed on `cifar5090-phase3-batch-matrix`:
- early selective `population_score_mean_s ≈ 7.47s` vs baseline `≈ 2.37s`
- full refresh `≈ 191.4s` vs baseline `≈ 15.64-19.25s`

Status:
- rejected as a baseline candidate in its current form

### `bntt` as a default replacement for `batch`

Observed on `cifar5090-phase3-bntt-kernel`:
- runtime essentially unchanged
- final accuracy slightly worse than `batch`

Status:
- keep `batch` as the default normalization baseline

## Benchmark Gates

A change becomes the new default only if it beats the current baseline on the
metrics that matter together, not separately.

Required comparison metrics:
- final test accuracy
- checkpoint-best test accuracy
- equal-wall-clock test accuracy
- `population_score_mean_s`
- `avg_update_s`
- sigma trajectory

Default promotion rule:
- no change becomes the new baseline unless it is at least neutral on throughput
  and clearly better on learning, or clearly better on throughput without losing
  meaningful accuracy

## Active Direction

Current recommended direction:
1. keep `batch + kernel_lora + selective perturbation` as the baseline
2. finish evaluating the active 2-hour sigma-tuned run
3. refine sigma behavior before attempting another major systems rewrite
4. only revisit `matrix_lora` with a narrowed, stage-limited implementation
