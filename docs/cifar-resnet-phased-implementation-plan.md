# CIFAR ResNet Phased Implementation Plan

This document is the primary implementation roadmap for improving the
`spiking_resnet18` CIFAR-10 path under the current pure-EGGROLL, single-GPU
training setup.

## Executive Summary

Current best observed CIFAR result:
- **21.18%** test accuracy on run
  `cifar5090-r2-p4096-t16-c32-b32-k96-20260411`

Current main bottlenecks:
- the residual block still does not behave like a true SEW-style identity-capable
  residual block
- the model is still locked to group norm, which diverges from the strongest
  from-scratch CIFAR SNN recipes
- conv perturbations are structured in update space but still execute as dense
  noisy kernels in forward passes
- the training loop is improved, but still not fully staged end to end

Why this work is phased:
- correctness and literature alignment should land before more throughput tuning
- learning quality problems and hardware-efficiency problems are related but not
  the same
- each phase should have its own tests, acceptance criteria, and benchmark gate

## Current State

What is already in place:
- a real CIFAR-sized convolutional `spiking_resnet18` path exists and runs
- CIFAR encoding preserves image structure over time
- centered-rank fitness shaping, sigma clamp support, batched EGGROLL updates,
  `updates_per_epoch`, and `bfloat16` support are already implemented
- the current best 5090 run is above chance and no longer collapses at
  initialization

What is still unresolved in the code:
- the residual shortcut path still passes through a LIF neuron, even for
  identity-preserving blocks
- the model currently only supports `resnet_norm="group"`
- deterministic evaluation is not yet separated from Poisson-coded training
- conv perturbations still materialize dense noisy kernels for forward execution
- the training loop still contains remaining Python-side orchestration and is
  not fully staged end to end

Baseline run to beat:
- `cifar5090-r2-p4096-t16-c32-b32-k96-20260411`
- best test accuracy: `21.18%`
- last recorded test accuracy: `19.00%`

Core metrics to track in every phase:
- best test accuracy
- last test accuracy
- sigma trajectory
- raw score std
- updates/sec
- average update time
- GPU memory used

Comparison rule:
- a phase is only complete if it either fixes a correctness invariant or clears
  its benchmark target
- learning-focused phases must compare accuracy at equal wall-clock budget, not
  only equal update count

## Phase 1 — Residual Correctness and Evaluation Integrity

### Goal

Make the residual block behavior align with identity-capable residual learning
and clean up evaluation semantics so accuracy and diagnostics reflect the model
itself rather than input sampling noise.

### Code Areas Likely Affected

- `spikyeggroll/models/spiking_resnet.py`
- `spikyeggroll/train.py`
- `spikyeggroll/eval.py`

### Exact Behavior Changes

- Rework the residual block so the shortcut is:
  - pure identity when input/output shapes match
  - projection-only when stride or channels change
- Remove shortcut membrane state for identity blocks.
- Keep spiking state only at actual neuron sites on the main branch.
- Make the residual merge explicit and additive by default.
- Add deterministic CIFAR evaluation:
  - keep Poisson encoding for training
  - add a deterministic static-image eval path for test/eval/debug
- Make CIFAR eval use the deterministic path consistently in:
  - `eval_test()`
  - `evaluate(...)`
  - debug probes
- Keep the current readout options, but document exact expected semantics for
  firing-rate vs membrane readout.

### Invariants That Must Not Change

- `model_name="spiking_resnet18"` remains the entrypoint
- projection shortcuts still apply when stride/channel changes
- the conv ResNet path stays compatible with the current EGGROLL parameter tree
- CIFAR training remains pure EGGROLL, not surrogate-gradient-based

### Tests to Add or Update

- identity blocks have no shortcut neuron state
- projection blocks still transform shape correctly
- residual merge outputs remain finite and shape-stable
- deterministic eval is batch-order-invariant and batch-companion-invariant
- deterministic eval and training eval paths are explicitly distinct

### Acceptance Criteria

- shortcut behavior matches the intended identity/projection split
- eval outputs are deterministic for fixed params and fixed images
- no regression in existing conv-forward tests
- debug stats remain finite and interpretable

### Risks and Likely Regressions

- changing block state shape can break checkpoint compatibility for in-flight
  CIFAR runs
- existing tests that assume shortcut LIF behavior will need updates
- deterministic eval may shift reported accuracy relative to previous Poisson
  test numbers

### Metrics to Compare Before Phase 2

- deterministic vs current eval accuracy on the same checkpoint
- output activity distribution before/after block correction
- short 5090 smoke comparison against the current baseline run

## Phase 2 — Literature-Aligned Learning Improvements

### Goal

Improve CIFAR learning quality by moving the training recipe closer to what the
from-scratch residual SNN literature actually uses.

### Code Areas Likely Affected

- `spikyeggroll/models/spiking_resnet.py`
- `spikyeggroll/configs.py`
- `spikyeggroll/train.py`

### Exact Behavior Changes

- Add a normalization mode beyond group norm:
  - preferred first step: `resnet_norm="batch"`
  - optional later extension: explicit BNTT-style mode
- Keep `group` available as a comparison path.
- Add an explicit readout mode field for CIFAR:
  - `firing_rate`
  - `membrane`
- Make firing-rate the CIFAR default unless a comparison run overrides it.
- Document and run the key CIFAR ablations:
  - `group` vs `batch`
  - `firing_rate` vs `membrane`
  - `timesteps=16` vs `timesteps=25`

### Invariants That Must Not Change

- no surrogate-gradient path is introduced
- current centered-rank shaping and sigma clamp remain available
- CIFAR launcher workflow remains compatible with existing scripts

### Tests to Add or Update

- BN path initializes, trains, and evaluates without batch-order leakage
- readout modes return finite logits and preserve shape
- CIFAR conv forward still works in `float32` and `bfloat16`

### Acceptance Criteria

- at least one normalization/readout configuration beats the current 21.18%
  baseline at equal wall-clock budget, or clearly improves learning stability
- no eval instability regressions are introduced

### Risks and Likely Regressions

- batch-norm state management may complicate EGGROLL parameter handling
- temporal BN variants can increase implementation complexity quickly
- accuracy gains may be sensitive to timesteps and sigma schedule

### Metrics to Compare Before Phase 3

- best test accuracy and last test accuracy
- sigma trajectory and raw score std
- output activity depth profile
- accuracy at equal wall-clock budget relative to the current baseline

## Phase 3 — EGGROLL Conv Efficiency Alignment

### Goal

Bring the conv perturbation path closer to the EGGROLL paper’s efficiency model
instead of treating conv perturbations as structured updates applied through
fully materialized dense noisy kernels.

### Code Areas Likely Affected

- `hyperscalees/noiser/eggroll.py`
- `hyperscalees/models/common.py`
- `spikyeggroll/models/spiking_resnet.py`

### Exact Behavior Changes

- Add an explicit conv ES mode selector:
  - `kernel_lora` for the current behavior
  - `matrix_lora` for the new path
- Implement a conv perturbation strategy that keeps low-rank perturbations in a
  forward-friendly matrix form, e.g. via unfold/im2col or another equivalent
  mapping.
- Preserve exact antithetic perturbation semantics and update logic.
- Keep the current conv mode as a fallback and comparison baseline.

### Invariants That Must Not Change

- antithetic sampling semantics must remain unchanged
- parameter updates must stay numerically consistent with the selected ES mode
- conv ResNet outputs must remain shape-compatible with existing training code

### Tests to Add or Update

- `kernel_lora` vs `matrix_lora` noisy-forward equivalence on a small
  deterministic case
- update semantics match for fixed seeds and scores
- no pytree/shape regressions in conv ES updates

### Acceptance Criteria

- throughput improves relative to the current conv path at equal model size
- peak memory remains below 32 GB on the 5090 benchmark runs
- learning quality does not regress materially relative to Phase 2

### Risks and Likely Regressions

- matrix-style conv perturbation may be significantly more invasive than the
  current kernel path
- memory usage may increase if the unfold path is not carefully structured
- numerical differences may appear even when update semantics are preserved

### Metrics to Compare Before Phase 4

- updates/sec
- average update time
- GPU memory used
- best test accuracy at equal wall-clock budget

## Phase 4 — Full Training-Loop Staging and Runtime Cleanup

### Goal

Remove remaining Python-side orchestration from the hot path and make the CIFAR
training loop cleaner, more stageable, and easier to benchmark.

### Code Areas Likely Affected

- `spikyeggroll/train.py`
- launcher/env defaults only if profiling shows a need

### Exact Behavior Changes

- Move the inner `updates_per_epoch` loop fully into JAX control flow.
- Replace remaining host-managed prefetch with a simpler staged batch-key
  pipeline or remove it if it has negligible benefit.
- Hoist eval JIT helpers entirely out of the call path.
- Keep batched updates and donation in place.
- Profile preallocation and memory-fraction defaults on the 5090 once the loop
  is fully staged.

### Invariants That Must Not Change

- metrics and checkpoints must preserve `global_update`
- existing CLI/runtime semantics for `updates_per_epoch` remain intact
- no change to CIFAR model topology in this phase

### Tests to Add or Update

- staged update loop matches prior behavior on a fixed-seed small run
- checkpoints/resume still preserve update counts and pending state
- eval path still reports deterministic accuracy

### Acceptance Criteria

- throughput improves relative to the current staged-but-partially-Python loop
- compile behavior is stable across repeated runs
- no checkpoint/resume regressions

### Risks and Likely Regressions

- aggressive staging can make debugging harder
- scan/lax refactors may change compilation boundaries in unexpected ways
- host/device overlap gains may be smaller than expected on a single 5090

### Metrics to Compare Before Phase 5

- compile latency
- updates/sec
- average update time
- GPU utilization and memory usage

## Phase 5 — Validation and Experiment Matrix

### Goal

Validate the phased changes against the current baseline and define stop/go
rules so future work stays disciplined.

### Exact Re-run Matrix on the 5090

Baseline to beat:
- `cifar5090-r2-p4096-t16-c32-b32-k96-20260411`
- best test accuracy `21.18%`

Recommended matrix:
- Phase 1 regression run:
  - same config as baseline, but with corrected residual block and deterministic eval
- Phase 2 learning runs:
  - `group` vs `batch`
  - `firing_rate` vs `membrane`
  - `T=16` vs `T=25`
- Phase 3 throughput runs:
  - `kernel_lora` vs `matrix_lora`
  - same model width and population
- Phase 4 runtime runs:
  - current loop vs fully staged loop
  - same config and seed where possible

### Stop/Go Rules

- Phase 1 is done when correctness invariants pass and deterministic eval is
  established.
- Phase 2 is done when at least one recipe beats the current baseline at equal
  wall-clock budget or clearly stabilizes learning.
- Phase 3 is done when throughput improves without a material learning
  regression.
- Phase 4 is done when runtime overhead is reduced without breaking checkpoints
  or determinism.
- If a phase fails its main benchmark target, stop and document the regression
  before proceeding.

## Cross-References

- Current CIFAR results and baseline runs:
  [docs/cifar-experiments-log.md](docs/cifar-experiments-log.md)
- Ongoing chronological notes:
  [docs/daily-notes.md](docs/daily-notes.md)

This phased plan is the primary source of truth for upcoming CIFAR ResNet
implementation work.
