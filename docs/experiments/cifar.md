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

- Best completed result:
  - run: `cifar5090-phase3-baseline-batch-kernel-20260413`
  - host: `216.249.100.66:21650`
  - result:
    - final test accuracy: `28.27%`
    - best checkpoint test accuracy: `26.98%` at `epoch 25`
    - `30` logged epochs, `300` updates, `1783.9s`
- Best observed long-run result so far:
  - run: `cifar5090-2h-batch-kernel-selective-sigmafix`
  - host: `216.249.100.66:21650`
  - latest confirmed state before the host stopped accepting SSH:
    - `epoch 108`
    - `global_update 1090`
    - best observed test accuracy: `30.36%` at `epoch 100`
    - later dip to `28.27%` at `epoch 105`
    - sigma climbed to and pinned at `0.012`
  - interpretation:
    - slower sigma decay helped push past `30%`
    - the late sigma policy overshot and became too exploratory
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
- Current stable JAX target for CIFAR selective runs:
  - `jax==0.9.2`
  - `jax 0.10.0` reproduced an XLA layout/reshape crash in the selective
    `full_model_refresh` path on multiple hosts and historical commits
- Current training/runtime caution:
  - medium selective runs are stable on `jax 0.9.2`
  - very large startup-heavy configs can still spend tens of minutes in compile
    before epoch `0`, especially when combining large population, large chunks,
    BNTT, and augmentation-heavy recipes

## Best Runs

| Role | Run | Key Config | Result |
|------|-----|------------|--------|
| Best observed to date | `cifar5090-2h-batch-kernel-selective-sigmafix` | baseline core + slower sigma decay | observed peak `30.36%` at `epoch 100`; final artifact not recovered |
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
- Sigma control is still the main optimizer lever:
  - the shorter baseline decayed to the floor late
  - the 2-hour sigma-tuned run avoided early collapse and reached `30.36%`
  - the same 2-hour run later overshot to `sigma_max=0.012` and destabilized

### Systems

- The current `matrix_lora` implementation is not a speed optimization.
- The likely loser is the patch-extraction delta path, not VRAM capacity.
- `jax 0.10.0` is not currently safe for the CIFAR selective ResNet path:
  - current `main`, `824ae2f`, and `9414039` all reproduced the same
    `ShapeUtil::ReshapeIsBitcast` failure at the selective `full_model_refresh`
    boundary when replayed under `jax 0.10.0`
  - the same medium selective run completed successfully under `jax 0.9.2`
- The selective `full_model_refresh` phase should use unbatched updates:
  - removing that override (`6616533`) reintroduced the epoch-8 crash pattern
  - current `main` restores the phase-specific fallback to
    `use_batched_update=False` for `full_model_refresh`

## 2026-04-18 Runtime Validation

### JAX version finding

Remote replay on `38.65.239.55:10633` established:

- `jax 0.10.0`:
  - historical commits that had previously worked (`9414039`, `824ae2f`) still
    crashed in the same selective path with:
    - `INTERNAL: RET_CHECK failure`
    - `layout_normalization.cc`
    - `ShapeUtil::ReshapeIsBitcast`
- `jax 0.9.2`:
  - a medium selective run at `9414039` completed through epoch `8` and epoch
    `9` without hitting the XLA crash

Takeaway:
- treat `jax 0.10.0` as a regression for this training path
- pin CIFAR runs to `jax 0.9.2` until there is an upstream fix or a confirmed
  local workaround

### Isolated host validation

An isolated clone and venv on `64.228.13.219:61169` was set up under:

- repo: `/workspace/repos/spikyeggrolls-cifar-baseline`
- data: `/workspace/data/spikyeggrolls-cifar-baseline`
- logs: `/workspace/logs/spikyeggrolls-cifar-baseline`
- checkpoints: `/workspace/checkpoints/spikyeggrolls-cifar-baseline`
- JAX cache: `/workspace/caches/jax-spikyeggrolls-cifar-baseline`

`make bootstrap-runpod` and `make doctor-runpod` both passed there with:

- `jax=0.9.2`
- `torch=2.11.0+cu130`
- `torchvision=0.26.0+cu130`
- `jax_devices=[CudaDevice(id=0)]`

### Smoke validation

Small CIFAR smoke:

- run: `cifar-smoke-small-20260418-host64`
- config:
  - `pop_size=32`
  - `timesteps=4`
  - `batch_size=8`
  - `chunk_size=8`
  - `resnet_channels_base=32`
- result:
  - final test accuracy `11.72%`
  - wall-clock `36.1s`

Takeaway:
- the isolated host/runtime is valid
- the CIFAR ResNet path boots, trains, evaluates, and writes checkpoints there

### Demo ladder results

Fixed demo base:

- `pop_size=128`
- `timesteps=8`
- `batch_size=16`
- `chunk_size=16`
- `resnet_channels_base=32`
- selective perturbation enabled
- `10` epochs, `1` update per epoch, `256` test eval samples

Completed results:

| Run | Change | Final test | Notes |
|-----|--------|------------|-------|
| `cifar-demo-01-base` | baseline | `11.33%` | stable |
| `cifar-demo-02-cutmix` | `augment + cutmix` | `8.20%` | worse than base |
| `cifar-demo-03-bntt` | `resnet_norm=bntt` | `12.50%` | best demo result |
| `cifar-demo-04-direct` | `direct_coding` | `12.11%` | second-best demo result |
| `cifar-demo-06-sigma` | adaptive sigma schedule | `11.33%` | no gain over base |
| `cifar-demo-07-cutmix-bntt` | `cutmix + bntt` | `10.94%` | worse than `bntt` alone |
| `cifar-demo-08-cutmix-sigma` | `cutmix + sigma` | `8.59%` | worse than base |

Failed demo:

- `cifar-demo-05-learnable`
  - change: `learnable_neuron_params`
  - failed before training with dtype mismatch:
    - `lax.conv_general_dilated requires arguments to have the same dtypes`
    - `float32` vs `bfloat16`

Takeaways:

- promote:
  - `bntt`
  - `direct_coding`
- do not promote yet:
  - `cutmix`
  - sigma schedule
  - `learnable_neuron_params` (needs a dtype fix first)

### Combined “final big run” behavior

Requested combined recipe:

- `pop_size=8000`
- `epochs=400`
- `timesteps=8`
- `batch_size=48`
- `chunk_size=128`
- `resnet_channels_base=32`
- `resnet_norm=bntt`
- `augment + cutmix`
- `direct_coding`
- sigma schedule enabled
- selective perturbation enabled

Run:

- `cifar-final-pop8k-bntt-direct-cutmix-sigma`

Observed behavior:

- alive after `~28 minutes`
- GPU remained fully busy (`~99-100%`, `~24.6 GiB`)
- stdout was still dominated by Triton/XLA fusion compilation
- no epoch output yet

Takeaway:

- this config is not obviously dead, but the time-to-first-step is too high for
  practical iteration
- use it only after proving a lower-pressure rung such as:
  - `pop_size=4096`, same rest of config
  - or `pop_size=8000` with `batch_size=16`, `chunk_size=16`

### Small-model / higher-population scale-up

Remote validation on `210.164.16.102:13146` used an isolated clone and runtime
under:

- repo: `/workspace/repos/spikyeggrolls-stage1`
- data: `/workspace/data/spikyeggrolls-stage1`
- logs: `/workspace/logs/spikyeggrolls-stage1`
- checkpoints: `/workspace/checkpoints/spikyeggrolls-stage1`
- JAX cache: `/workspace/caches/jax-spikyeggrolls-stage1`

Both runs below used:

- `jax 0.9.2`
- `resnet_channels_base=16`
- `timesteps=8`
- `batch_size=48`
- `chunk_size=128`
- `resnet_norm=bntt`
- `direct_coding`
- sigma schedule enabled
- selective perturbation enabled

#### Stage 1: `pop_size=4096`

Run:

- `cifar-stage1-ch16-pop4096-bntt-direct`

Observed behavior:

- cleared startup normally
- steady-state throughput around `1.0 upd/s`
- stable throughout training
- meaningful learning, then clear rollover

Observed test checkpoints:

- `epoch 0`: `8.93%`
- `epoch 10`: `13.69%`
- `epoch 20`: `16.27%`
- `epoch 30`: `18.35%`
- `epoch 40`: `20.24%`
- `epoch 50`: `20.34%`
- `epoch 60`: `21.23%`
- `epoch 70`: `21.13%`
- `epoch 80`: `21.33%`
- `epoch 90`: `21.13%`
- `epoch 100`: `21.13%`
- `epoch 110`: `19.84%`
- `epoch 120`: `19.84%`
- `epoch 130`: `19.25%`
- `epoch 140`: `18.95%`
- `epoch 150`: `19.44%`
- `epoch 160`: `18.95%`

Takeaway:

- viable and practical
- best useful checkpoint: `21.33%` at `epoch 80`
- running past `epoch 80-100` was not worthwhile; the run had already peaked

#### Stage 2: `pop_size=8000`

Run:

- `cifar-stage2-ch16-pop8000-bntt-direct`

Observed behavior:

- much heavier startup/compile than Stage 1
- eventually entered training successfully
- slower steady-state throughput, around `0.55-0.73 upd/s`
- clearly better best accuracy than Stage 1
- later rollover again after the best checkpoint

Observed test checkpoints:

- `epoch 0`: `9.03%`
- `epoch 10`: `14.29%`
- `epoch 20`: `19.94%`
- `epoch 30`: `20.04%`
- `epoch 40`: `22.02%`
- `epoch 50`: `21.92%`
- `epoch 60`: `21.63%`
- `epoch 70`: `22.42%`
- `epoch 80`: `23.81%`
- `epoch 90`: `21.23%`
- `epoch 100`: `20.73%`

Sigma behavior:

- by `epoch 80`, sigma was still moderate at about `0.00499`
- after that it climbed hard:
  - `epoch 90`: `0.00457`
  - `epoch 95`: `0.00811`
  - `epoch 100`: `0.00985`
  - `epoch 102+`: pinned at `sigma_max = 0.01200`

Takeaway:

- `pop_size=8000` beat `pop_size=4096`
- best useful checkpoint: `23.81%` at `epoch 80`
- the extra population bought about `+2.5` absolute points over Stage 1 peak
- the current sigma schedule becomes too exploratory late and likely contributes
  to the post-peak decline

Recommendation from the scale-up comparison:

- keep the Stage 2 core recipe as the strongest current small-model direction:
  - `resnet_channels_base=16`
  - `pop_size=8000`
  - `resnet_norm=bntt`
  - `direct_coding`
  - `timesteps=8`
  - `batch_size=48`
  - `chunk_size=128`
- but revise the sigma policy before the next long run:
  - lower `sigma_max`
  - reduce `sigma_growth`
  - avoid late-stage drift to the exploration ceiling

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

Long-run sigma experiment:
- `cifar5090-2h-batch-kernel-selective-sigmafix`
- same proven baseline core with a less collapse-prone sigma schedule:
  - `sigma_warmup_epochs=40`
  - `sigma_min=0.0035`
  - `sigma_target_success=0.12`
  - `sigma_success_tolerance=0.05`
  - `sigma_decay=0.995`
  - `sigma_growth=1.01`
  - `profile_mode=off`

Latest confirmed state:
- `epoch 108`
- `global_update 1090`
- observed test checkpoints:
  - `epoch 0`: `13.00%`
  - `epoch 5`: `21.53%`
  - `epoch 10`: `23.71%`
  - `epoch 15`: `26.59%`
  - `epoch 20`: `25.69%`
  - `epoch 25`: `25.99%`
  - `epoch 90`: `29.66%`
  - `epoch 95`: `29.86%`
  - `epoch 100`: `30.36%`
  - `epoch 105`: `28.27%`
- latest optimizer behavior:
  - sigma climbed to `0.012` and pinned there
  - repeated `sigma_action=grow` preceded the late accuracy dip
- operational note:
  - the host later stopped accepting SSH, so the final artifact was not recovered

## Recommended Run Ladder

If the goal is to maximize both accuracy and throughput, do not jump straight to a
single long run. Use this short elimination ladder and choose the winner by
equal-wall-clock accuracy.

### Fixed baseline core

Keep these fixed unless a run explicitly changes one of them:

- `dataset=cifar10`
- `model_name=spiking_resnet18`
- `pop_size=4096`
- `rank=2`
- `lr=0.0015`
- `updates_per_epoch=10`
- `batch_size=48`
- `chunk_size=96`
- `dtype=bfloat16`
- `resnet_channels_base=32`
- `resnet_norm=batch`
- `conv_es_mode=kernel_lora`
- `selective_stage_perturbation`
- `stage_perturbation_schedule=head_last_then_last2`
- `stage_perturbation_early_fraction=0.30`
- `stage_perturbation_full_epoch_interval=8`
- `test_interval=5`
- `checkpoint_interval=10`
- `num_test_eval_samples=1024`
- `profile_mode=off`

### Short-run comparison matrix

Use `epochs=30` for all four runs below.

| Order | Run Name | Changes vs control | Goal | Expected tradeoff |
|------|-----------|--------------------|------|-------------------|
| 1 | `cifar-control` | `timesteps=16`, `augment`, current baseline sigma schedule | Re-establish current reference on the target host | Safest known behavior |
| 2 | `cifar-sigma` | tighter long-run sigma policy only | Test whether better exploration control improves wall-clock accuracy | Highest-confidence improvement path |
| 3 | `cifar-sigma-cutmix` | `cifar-sigma` + `cutmix`, `cutmix_alpha=1.0` | Test generalization gain from stronger augmentation | Best raw-accuracy upside among new knobs |
| 4 | `cifar-t12-sigma-cutmix` | `cifar-sigma-cutmix` + `timesteps=12` | Test whether lower T wins on accuracy per hour | Best throughput-adjusted candidate |

Optional fifth run only after the first four:

| Order | Run Name | Changes vs control | Goal | Expected tradeoff |
|------|-----------|--------------------|------|-------------------|
| 5 | `cifar-t20-sigma-cutmix` | `cifar-sigma-cutmix` + `timesteps=20` | Test whether more temporal integration beats its cost | Accuracy-biased, slower than `T=16` |

### Recommended sigma policy for the test ladder

Use this policy for `cifar-sigma`, `cifar-sigma-cutmix`, `cifar-t12-sigma-cutmix`, and `cifar-t20-sigma-cutmix`:

- `sigma=0.006`
- `sigma_warmup_epochs=40`
- `sigma_min=0.0035`
- `sigma_max=0.010`
- `sigma_target_success=0.14`
- `sigma_success_tolerance=0.04`
- `sigma_growth=1.005`
- `sigma_decay=0.995`
- `sigma_ema_decay=0.90`

Rationale:

- the prior 2-hour run showed that slower sigma decay can push CIFAR past `30%`
- the same run also showed that `sigma_max=0.012` and faster late growth can overshoot
- this tighter upper policy keeps the long-horizon idea while reducing late runaway risk

### Suggested commands

Control:

```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 30 --updates_per_epoch 10 \
  --timesteps 16 --batch_size 48 --chunk_size 96 \
  --augment --dtype bfloat16 \
  --resnet_channels_base 32 \
  --resnet_norm batch \
  --conv_es_mode kernel_lora \
  --selective_stage_perturbation \
  --stage_perturbation_schedule head_last_then_last2 \
  --stage_perturbation_early_fraction 0.30 \
  --stage_perturbation_full_epoch_interval 8 \
  --sigma_warmup_epochs 20 \
  --test_interval 5 --checkpoint_interval 10 --log_interval 1 \
  --num_test_eval_samples 1024 \
  --profile_mode off \
  --run_name cifar-control
```

Sigma-only:

```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 30 --updates_per_epoch 10 \
  --timesteps 16 --batch_size 48 --chunk_size 96 \
  --augment --dtype bfloat16 \
  --resnet_channels_base 32 \
  --resnet_norm batch \
  --conv_es_mode kernel_lora \
  --selective_stage_perturbation \
  --stage_perturbation_schedule head_last_then_last2 \
  --stage_perturbation_early_fraction 0.30 \
  --stage_perturbation_full_epoch_interval 8 \
  --sigma_warmup_epochs 40 \
  --sigma_min 0.0035 --sigma_max 0.010 \
  --sigma_target_success 0.14 --sigma_success_tolerance 0.04 \
  --sigma_growth 1.005 --sigma_decay 0.995 --sigma_ema_decay 0.90 \
  --test_interval 5 --checkpoint_interval 10 --log_interval 1 \
  --num_test_eval_samples 1024 \
  --profile_mode off \
  --run_name cifar-sigma
```

Sigma + CutMix:

```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 30 --updates_per_epoch 10 \
  --timesteps 16 --batch_size 48 --chunk_size 96 \
  --augment --cutmix --cutmix_alpha 1.0 --dtype bfloat16 \
  --resnet_channels_base 32 \
  --resnet_norm batch \
  --conv_es_mode kernel_lora \
  --selective_stage_perturbation \
  --stage_perturbation_schedule head_last_then_last2 \
  --stage_perturbation_early_fraction 0.30 \
  --stage_perturbation_full_epoch_interval 8 \
  --sigma_warmup_epochs 40 \
  --sigma_min 0.0035 --sigma_max 0.010 \
  --sigma_target_success 0.14 --sigma_success_tolerance 0.04 \
  --sigma_growth 1.005 --sigma_decay 0.995 --sigma_ema_decay 0.90 \
  --test_interval 5 --checkpoint_interval 10 --log_interval 1 \
  --num_test_eval_samples 1024 \
  --profile_mode off \
  --run_name cifar-sigma-cutmix
```

Timestep-reduced throughput candidate:

```bash
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 --model_name spiking_resnet18 \
  --pop_size 4096 --rank 2 --sigma 0.006 --lr 0.0015 \
  --epochs 30 --updates_per_epoch 10 \
  --timesteps 12 --batch_size 48 --chunk_size 96 \
  --augment --cutmix --cutmix_alpha 1.0 --dtype bfloat16 \
  --resnet_channels_base 32 \
  --resnet_norm batch \
  --conv_es_mode kernel_lora \
  --selective_stage_perturbation \
  --stage_perturbation_schedule head_last_then_last2 \
  --stage_perturbation_early_fraction 0.30 \
  --stage_perturbation_full_epoch_interval 8 \
  --sigma_warmup_epochs 40 \
  --sigma_min 0.0035 --sigma_max 0.010 \
  --sigma_target_success 0.14 --sigma_success_tolerance 0.04 \
  --sigma_growth 1.005 --sigma_decay 0.995 --sigma_ema_decay 0.90 \
  --test_interval 5 --checkpoint_interval 10 --log_interval 1 \
  --num_test_eval_samples 1024 \
  --profile_mode off \
  --run_name cifar-t12-sigma-cutmix
```

### Decision rule

Compare each short run on:

- test accuracy at epochs `5`, `10`, `15`, `20`, `25`, `30`
- wall-clock runtime
- best test accuracy
- sigma trajectory

Rank the candidates by:

1. best test accuracy at equal wall-clock budget
2. sigma stability
3. final test accuracy
4. total runtime

### Recommended long-run candidates

Most likely winners by objective:

- best raw-accuracy shot: `T=16 + sigma + cutmix`
- best accuracy per hour: `T=12 + sigma + cutmix`
- safest improved baseline: `T=16 + sigma` without CutMix

## Next Steps

Priority order:
1. run the short experiment ladder above and choose the winner by equal-wall-clock accuracy
2. recover the final 2-hour run artifact if the host becomes reachable again
3. keep retuning the sigma upper-control policy if late overshoot persists:
   - lower `sigma_max`
   - weaken late growth
   - widen hold behavior near the target band
3. keep `batch + kernel_lora` as the default short-run baseline
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
