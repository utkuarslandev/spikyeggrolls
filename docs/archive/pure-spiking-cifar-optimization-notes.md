# Archived: Pure-Spiking CIFAR Optimization Notes

This document is archived.

Current sources of truth:
- live experiment status: [../experiments/cifar.md](../experiments/cifar.md)
- live implementation roadmap: [../roadmaps/cifar-resnet-roadmap.md](../roadmaps/cifar-resnet-roadmap.md)

## Summary

These notes capture the current recommendation for improving CIFAR `spiking_resnet18` training while staying strictly pure spiking.

The two goals are:

- reduce `score_population_chunked` cost
- improve CIFAR learning speed

This document intentionally excludes hybrid ANN/SNN front-end ideas. Those may still be valid overall, but they are out of scope here because the target is a pure-spiking model.

## Current Read on the Bottleneck

From the 5090 steady-state profiling run:

- `population_score_s` dominates update time
- `forward_eval_s` is tiny in comparison
- `update_s` is tiny in steady state
- test evaluation is not the main runtime problem

The practical conclusion is:

- the main systems bottleneck is population scoring, not replay update or eval
- the main optimization bottleneck is weak exploration after sigma collapses

Code-side, the most important observation is that the conv path does not currently get the full EGGROLL-style forward-speed benefit that matrix layers get.

- matrix layers use low-rank perturbations directly in the forward path
- conv layers still materialize dense perturbed kernels and then run normal convolutions

That means the current pure-spiking conv ResNet is expensive exactly where ES multiplies cost most: noisy population forward.

## Critique of Candidate Ideas

### Hybrid ANN front-end

This is the best speed/accuracy compromise overall, but it breaks the pure-spiking constraint.

Verdict:

- reject for now

### Conv `matrix_lora` / im2col-style EGGROLL forward

This is the cleanest long-term systems idea for pure spiking.

Why it is attractive:

- directly targets `score_population_chunked`
- keeps the architecture pure spiking
- aligns conv perturbations more closely with the EGGROLL paper’s forward-efficiency idea

Main risks:

- high implementation complexity
- im2col-style lowering can shift the bottleneck from compute to memory if done naively
- likely requires careful profiling and shape management to avoid making compile or memory behavior worse

Verdict:

- strong long-term systems direction
- not the first change to ship

### Stage-wise / block-coordinate ES

Keep the whole model pure spiking, but only perturb a subset of stages on most updates.

Example:

- most updates: perturb stage 4 + classifier
- some updates: perturb stages 3-4 + classifier
- occasional updates: perturb the whole network

Why it is attractive:

- lowers noisy forward cost per update
- reduces optimization noise in fragile early layers
- does not violate the pure-spiking constraint
- lower implementation risk than a full conv-forward redesign

Limits:

- the full network still runs every forward pass
- speedup will not match a fully shared/cached front-end
- needs careful scheduling so early layers do not become permanently under-trained

Verdict:

- best near-term pure-spiking optimization idea

### Smaller pure-spiking backbone

A smaller model is a serious option under ES.

Why it may help:

- lower population-scoring cost
- larger effective exploration budget per unit wall-clock
- may improve optimization stability relative to a larger, under-explored model

Risk:

- lower ceiling on final accuracy

Current assessment:

- this risk is acceptable because current accuracy is still far below any plausible model-capacity ceiling

Verdict:

- practical, high-ROI fallback if stage-wise ES and optimizer fixes are not enough

### Bigger batch / chunk / population

This is not the main lever anymore.

Why:

- the GPU is already compute-bound on the good 5090 runs
- profiling shows `population_score` is the dominant cost
- blindly scaling population or chunk size does not fix the underlying inefficiency

Verdict:

- secondary tuning lever, not a primary optimization strategy

## Best Pure-Spiking Direction

The best chance of improving both runtime and learning, while staying pure spiking, is:

1. perturb less of the network per update
2. keep exploration alive longer
3. make the pure-spiking residual network easier to train

That translates into the following ranked plan.

## Ranked Recommendations

### 1. Implement stage-wise selective perturbation

This is the highest-priority pure-spiking optimization.

Recommended behavior:

- add config controlling which stages are ES-active
- allow stage schedules over training
- default schedule:
  - warm-up: classifier + last stage only
  - mid-run: last two stages
  - periodic full-model updates

Expected benefit:

- reduces noisy population-scoring cost
- improves update quality by focusing exploration where it matters most

### 2. Fix sigma adaptation

This is the clearest live-run optimizer failure.

Observed issue:

- sigma starts reasonable
- learning improves
- sigma collapses to the floor
- raw score spread collapses with it

Expected change:

- slower decay
- higher effective floor for CIFAR
- more conservative adaptation so exploration does not die too early

Expected benefit:

- better learning per expensive update

### 3. Fix residual shortcut correctness

Pure-spiking CIFAR ResNets need every training advantage available.

Key target:

- identity-capable residual shortcuts
- avoid shortcut behavior that undermines the residual path

Expected benefit:

- faster and more stable learning
- better alignment with SEW-style residual principles

### 4. Add stronger normalization for the spiking ResNet

Current group norm is operationally stable, but it is not the strongest literature-aligned choice for deep from-scratch CIFAR SNN training.

Likely target:

- BatchNorm first
- BNTT-style normalization later if needed

Expected benefit:

- fewer updates needed to reach a useful CIFAR regime

### 5. Shrink the pure-spiking model if needed

If runtime is still too expensive after the changes above:

- reduce `resnet_channels_base`
- consider reducing active depth in the early experiments

Expected benefit:

- better ES budget allocation
- cheaper `score_population_chunked`

### 6. Pursue conv `matrix_lora` after the easier wins

This remains the long-term systems fix.

It should come after:

- stage-wise ES
- sigma schedule improvements
- residual/normalization fixes

Reason:

- those earlier changes are lower risk and more likely to improve wall-clock learning quickly

## Recommended Order of Work

If the goal is the best pure-spiking path with the current codebase, the recommended order is:

1. stage-wise selective perturbation
2. sigma schedule redesign
3. residual shortcut correction
4. stronger normalization path
5. model-size reduction if needed
6. conv `matrix_lora` / lowered conv-forward redesign

## Overall Conclusion

For a pure-spiking CIFAR model under ES, the most important idea is not larger hardware settings and not architectural hybridization.

The best near-term strategy is:

- perturb fewer spiking stages per update
- prevent exploration from collapsing
- make the residual SNN easier to optimize

The best long-term systems strategy is:

- redesign the conv noisy-forward path so conv layers benefit from EGGROLL-style low-rank forward efficiency, not only low-rank updates
