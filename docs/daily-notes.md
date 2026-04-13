# Daily Experiment Notes

This file is the chronological lab notebook.

Use it for:
- day-by-day work logs
- raw commands and observations
- implementation milestones as they happened

For the current CIFAR/ResNet state, active baseline, and next actions, see
[docs/cifar-experiments-log.md](docs/cifar-experiments-log.md).

## March 31 — Project Bootstrap & Initial Benchmarks

### Pop_size throughput sweep (deterministic SCN, RTX 4080)
```
.venv/bin/python scripts/run_experiments.py --sweep_pop
.venv/bin/python scripts/run_experiments.py --sweep_pop --pop_sizes 1024 2048 4096 8192 16384 32768 65536 100000
```
| pop_size | ep/s |
|----------|------|
| 32–128 | ~34 |
| 256 | 31.8 |
| 1024 | 28.2 |
| 4096 | 18.3 |
| 16384 | 9.5 |
| 65536 | 3.4 |
| 100000 | 1.0 |

### Homeostatic threshold — initial tests (all failed)
```
train_scn(SCNEggrollConfig(num_epochs=20, pop_size=1000, homeo_lr=0.001))
# Result: fitness -0.0, RMSE 0.0000, 0 spikes (silent)

train_scn(SCNEggrollConfig(num_epochs=30, pop_size=1000, homeo_lr=0.001, signal_type='sine', random_init=True))
# Result: fitness -0.4999, RMSE 0.7071, 0 spikes (constant half-error)

train_scn(SCNEggrollConfig(num_epochs=50, pop_size=1000, homeo_lr=0.01, signal_type='sine'))
```
Takeaway: homeostasis alone couldn't rescue silent neurons at this stage.

---

## April 1–2 — Deterministic SCN Training (Big Session)

### M=1 sine, analytical init
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 300 --homeo_lr 0.001 --pop_size 4096
# RMSE 0.1797, 598 spikes
```

### M=1 sine, random init
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 300 --homeo_lr 0.001 --pop_size 4096 --random_init
# Initial RMSE 21.95 → after Wishart Omega fix → RMSE 0.4213, 97 spikes
```

### M=4 sine, baseline (rank=1, sigma=0.02)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 50 --homeo_lr 0.01 --pop_size 10000 --M 4 --random_init
# RMSE 0.6090, 153 spikes, fitness -1.49
```

### M=4 sine, higher sigma + rank
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 50 --homeo_lr 0.01 --pop_size 10000 --M 4 --random_init \
  --sigma 0.1 --rank 4
# RMSE 0.5403, 169 spikes, fitness -1.17
```

### M=4 sine, longer run (300ep)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 300 --homeo_lr 0.01 --pop_size 10000 --M 4 --random_init \
  --sigma 0.1 --rank 4
# RMSE 0.6011, 189 spikes, best fitness -1.01 then regressed to -1.45
```

### M=4 sine, low homeo_lr, 1000ep
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 1000 --homeo_lr 0.001 --pop_size 10000 --M 4 --random_init \
  --sigma 0.1 --rank 1 --mu 0
# RMSE 0.2846, 836 spikes, best fitness -0.050
```

### M=4 sine, F=D^T fixed (evolve only T+Omega)
```
.venv/bin/python -m spikyeggroll.train_scn --evolve_T 1 --evolve_F 0 --evolve_Omega 1 \
  --signal sine --greedy --epochs 100 --homeo_lr 0.001 --pop_size 10000 \
  --M 4 --random_init --sigma 0.1 --rank 1 --mu 0
# RMSE 0.40, ~360 spikes, fitness -0.646
# KEY FINDING: F=D^T fixed learns 10x faster than evolving F from random
```

### Sigma sweep (M=4, random init, rank=4, mu=0, 100 epochs)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 100 --homeo_lr 0.01 --pop_size 100000 --M 4 --random_init \
  --sigma $SIGMA --rank 4 --mu 0
```
| sigma | Best Fit | Final Fit | RMSE | Spikes |
|-------|----------|-----------|------|--------|
| 0.001 | -1.575 | -1.721 | 0.656 | 178 |
| 0.003 | -1.512 | -1.565 | 0.626 | 129 |
| 0.01 | -1.505 | -1.508 | 0.614 | 175 |
| 0.03 | -1.397 | -1.487 | 0.610 | 177 |
| **0.1** | **-1.005** | **-1.057** | **0.514** | **196** |
| 0.3 | -1.584 | -1.640 | 0.640 | 141 |
| 1.0 | -1.822 | -1.928 | 0.694 | 24 |

### D gradient learning (M=4, N=50)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 100 --homeo_lr 0.001 --pop_size 10000 --M 4 --random_init \
  --sigma 0.1 --rank 1 --mu 0.02 --d_lr 0.01 --d_update_interval 10
# RMSE 0.1828, 522 spikes
```

### D gradient, N=300, 1000ep
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 1000 --homeo_lr 0.001 --pop_size 10000 --M 4 --N 300 --random_init \
  --sigma 0.1 --rank 1 --mu 0.02 --d_lr 0.01 --d_update_interval 10
# RMSE 0.1032, 631 spikes — best deterministic result
```

### D gradient, N=300, no homeostasis
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal sine --greedy \
  --epochs 1000 --homeo_lr 0 --pop_size 10000 --M 4 --N 300 --random_init \
  --sigma 0.1 --rank 1 --mu 0.02 --d_lr 0.01 --d_update_interval 10
# RMSE 1.5571, 5000 spikes — homeostasis matters at scale
```

---

## April 2 — Stochastic LIF + Langevin Filtering (Phase 1)

### First stochastic run (analytical init, M=1)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal langevin \
  --epochs 100 --homeo_lr 0 --pop_size 10000 --M 1 --N 50 \
  --sigma 0.1 --rank 1 --mu 0.02 --d_lr 0.01 --d_update_interval 10 \
  --signal_noise 0.3 --signal_damping 2.0 --escape_noise --escape_beta 50 \
  --membrane_noise 0.1
# RMSE 0.2315, 697 spikes — first working stochastic run
```

### With Poisson KL loss (variational free energy)
```
# Same params as above, after switching to Poisson KL loss
# RMSE 0.0635, 530 spikes — big improvement from KL regularization
```

### Decoder scale sweep (escape_beta / decoder_scale interaction, 5ep smoke tests)
```
--escape_beta 1  --decoder_scale 0.1  → RMSE 0.0696, 26 spikes
--escape_beta 50 --decoder_scale 1.0  → RMSE 0.0541, 4 spikes
--escape_beta 50 --decoder_scale 0.01 → RMSE 0.0662, 19 spikes
```

### Many divergent runs (decoder_scale=0.1 problematic)
```
--decoder_scale 0.1 --d_lr 0.01  → RMSE inf (multiple attempts)
--decoder_scale 0.1 --d_lr 0.001 → RMSE inf
--decoder_scale 0.2 --readout_gain 1.0 → RMSE 145461 (first), then 0.0711 with d_lr=0
```

### Readout_gain=1.0, fixed D (working config found)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal langevin \
  --epochs 30 --homeo_lr 0 --pop_size 10000 --M 1 --N 50 \
  --sigma 0.01 --rank 1 --mu 0 --d_lr 0 --d_update_interval 10 \
  --signal_noise 0.3 --signal_damping 2.0 --escape_noise --escape_beta 50 \
  --membrane_noise 0.1 --decoder_scale 0.2 --readout_gain 1.0
# RMSE 0.0711, 14 spikes
```

### D gradient with explicit JIT argument (after fix)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal langevin \
  --epochs 30 --homeo_lr 0 --pop_size 10000 --M 1 --N 50 \
  --sigma 0.01 --rank 1 --mu 0 --d_lr 0.001 --d_update_interval 10 \
  --signal_noise 0.3 --signal_damping 2.0 --escape_noise --escape_beta 50 \
  --membrane_noise 0.1 --decoder_scale 0.2 --readout_gain 1.0
# RMSE 0.0880, 37 spikes (after JIT fix)
```

### Stochastic, random init, signal_noise sweep (M=1)
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal langevin \
  --epochs 200 --homeo_lr 0 --pop_size 10000 --M 1 --N 50 \
  --sigma 0.01 --rank 1 --mu 0 --d_lr 0.001 --d_update_interval 10 \
  --signal_noise $NOISE --signal_damping 2.0 --escape_noise --escape_beta 50 \
  --membrane_noise 0.1 --decoder_scale 0.2 --readout_gain 1.0 --random_init
```
| signal_noise | RMSE | Spikes |
|-------------|------|--------|
| 1.0 | **0.0887** | 252 |
| 9.0 | 0.1478 | 1512 |

### Stochastic, random init, M=4
```
.venv/bin/python -m spikyeggroll.train_scn --exp 3 --signal langevin \
  --epochs 200 --homeo_lr 0 --pop_size 10000 --M 4 --N 50 \
  --sigma 0.01 --rank 1 --mu 0 --d_lr 0.001 --d_update_interval 10 \
  --signal_noise 9.0 --signal_damping 2.0 --escape_noise --escape_beta 50 \
  --membrane_noise 0.1 --decoder_scale 0.2 --readout_gain 1.0 --random_init
# RMSE 0.3138, 4151 spikes

# Same with --d_lr 0 (frozen D):
# RMSE 0.2906, 4496 spikes
```

### Posterior score verification
```
.venv/bin/python scripts/check_posterior_score.py
# V_i = d_i^T * e correlation: 0.9998
# SCN RMSE: 0.3249
# Kalman RMSE: 2.2651
# SCN significantly outperforms Kalman baseline
```

---

## April 3 — Pop_size Sweep (4D Stochastic Langevin)

### Pop_size sweep (random init, 4D Langevin)
```
for POP in 64 128 256 512 1024 2048 4096 8192 16384; do
  .venv/bin/python -m spikyeggroll.train_scn \
    --signal langevin --M 4 --random_init --escape_noise \
    --signal_noise 9 --epochs 200 --N 50 --pop_size $POP
done
```
| pop_size | Final RMSE | Spikes |
|----------|-----------|--------|
| 64 | 3.535 | 202 |
| 128 | 3.526 | 218 |
| 256 | 3.526 | 199 |
| 512 | 3.470 | 204 |
| 1024 | 3.464 | 211 |
| 2048 | 3.443 | 223 |
| 4096 | 3.458 | 232 |
| 8192 | 3.440 | 245 |
| 16384 | 3.438 | 249 |

Note: all starting from Initial RMSE ~17.63. Missing key hyperparams (exp=1 only evolved T, wrong sigma/decoder_scale/readout_gain). Diminishing returns past pop_size ~1024.

### Pop_size sweep with full config (exp=3, sigma=0.01, decoder_scale=0.2, readout_gain=1.0, d_lr=0.001, membrane_noise=0.1, signal_damping=2.0, homeo_lr=0)
```
for POP in 16384 8192 4096 2048 1024 512 256 128 64; do
  .venv/bin/python -m spikyeggroll.train_scn --exp 3 \
    --signal langevin --M 4 --random_init --escape_noise \
    --signal_noise 9 --epochs 200 --N 50 --pop_size $POP
done
```
| pop_size | Final RMSE | Spikes |
|----------|-----------|--------|
| 16384 | 0.288 | 3968 |
| 8192 | 0.291 | 4139 |
| 4096 | 0.324 | 5514 |
| 2048 | 0.340 | 5142 |
| 1024 | 0.375 | 5000 |
| 512 | 0.543 | 5479 |

---

## April 13 — Phase 2/3 CIFAR ResNet Work, Validation, and New 5090 Baseline

### Phase 2 selective-stage perturbation implementation

Implemented:
- stage-group taxonomy for `spiking_resnet18`
- selective perturbation schedule:
  - early: `stage3 + head`
  - mid: `stage2 + stage3 + head`
  - periodic full refresh
- active ES-map masking by stage group
- deterministic prefix caching with noisy suffix-only forward
- per-epoch metrics:
  - `perturbation_phase`
  - `active_stage_groups`
  - `cache_split`
  - `active_param_fraction`

Validation:
```bash
.venv/bin/python -m pytest tests/test_eggroll.py tests/test_lif.py tests/test_cifar10.py tests/test_model.py tests/test_runtime.py -q
# 52 passed
```

Takeaway:
- selective perturbation is implemented and locally validated
- prefix/suffix cached forward matches full forward in tests
- selective updates only touch the intended active stages

### Phase 3 implementation: BNTT + conv `matrix_lora`

Implemented:
- new `resnet_norm="bntt"` mode with:
  - per-timestep affine weights
  - per-timestep running mean/variance
  - timestep-aware scan/prefix/suffix forward handling
- running-stat evaluation path extended from `batch` to `batch|bntt`
- new `conv_es_mode`:
  - `kernel_lora` (existing dense noisy-kernel path)
  - `matrix_lora` (dense base conv + patch-space low-rank delta)

Validation:
- `tests/test_runtime.py` passed
- `tests/test_model.py` passed
- `tests/test_cifar10.py` passed

Takeaway:
- the repo now has both the literature-aligned BNTT path and the systems-facing
  `matrix_lora` conv forward
- local correctness is good, but 5090 performance/learning impact still needs
  hardware comparison

### Fresh remote baseline from GitHub clone

Host:
- `216.249.100.66:21650`

Source:
- fresh clone of `https://github.com/utkuarslandev/spikyeggrolls.git`
- cloned commit reported as `94d9afc`

Bootstrap:
```bash
make bootstrap-runpod
```

Baseline command:
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
  --profile_mode steady_state \
  --profile_warmup_updates 5 \
  --profile_updates_window 3 \
  --profile_eval_once \
  --profile_server_port 9999 \
  --run_name cifar5090-phase3-baseline-batch-kernel-20260413
```

Observed by epoch 9 / update 100:
- run healthy, no startup stall
- GPU memory: about `24.65 / 32.6 GiB`
- GPU utilization: sustained high utilization; occasional `0%` samples were just
  transient between-kernel snapshots
- first test accuracy:
  - `epoch 5`: `20.73%`

Timing by selective phase:
- early selective (`stage3 + head`, `cache_split=after_stage2`):
  - `active_param_fraction ≈ 0.274`
  - `population_score_mean_s ≈ 2.35-2.38s`
  - `avg_update_s ≈ 3.6-3.8s`
- full-model refresh (`epoch 8`):
  - `active_param_fraction = 1.0`
  - `population_score_mean_s ≈ 19.25s`
  - `avg_update_s ≈ 20.77s`
- mid selective (`epoch 9`, `stage2 + stage3 + head`, `cache_split=after_stage1`):
  - `active_param_fraction ≈ 0.516`
  - `population_score_mean_s ≈ 6.75s`
  - `avg_update_s ≈ 7.75s`

Takeaway:
- selective perturbation is not a cosmetic change; it dramatically changes wall
  clock cost
- the runtime ordering is now explicit:
  - early selective: cheapest
  - mid selective: intermediate
  - full refresh: very expensive
- this baseline is the new reference for comparing:
  - `bntt + kernel_lora`
  - `batch + matrix_lora`
  - `bntt + matrix_lora`
| 256 | 0.865 | 7358 |
| 128 | 0.843 | 4220 |
| 64 | 0.995 | 9355 |

Note: Initial RMSE ~3.13. Matches original 4D result (0.31). Diminishing returns past ~8k. Knee of curve at 1k–4k. Hyperparams (not pop_size) were the bottleneck in the first sweep.

### Hyperparam ablation at pop_size=256 (one param changed from full config)
```
Baseline: full config at pop_size=256
Each row: one param reverted to its old default
```
| Ablation | Final RMSE | Delta | Spikes |
|----------|-----------|-------|--------|
| **baseline (full config)** | **0.611** | — | 4979 |
| sigma=0.02 (was 0.01) | 0.727 | +0.12 | 5737 |
| decoder_scale=1.0 (was 0.2) | 1.153 | +0.54 | 3719 |
| readout_gain=0.25 (was 1.0) | 0.556 | -0.06 | 18804 |
| d_lr=0 (was 0.001) | 0.762 | +0.15 | 8852 |
| membrane_noise=0 (was 0.1) | 0.632 | +0.02 | 6034 |
| signal_damping=1.0 (was 2.0) | 0.735 | +0.12 | 5665 |
| homeo_lr=0.001 (was 0.0) | 2.296 | +1.69 | 1291 |

Top 3 most impactful (at pop_size=256):
1. **homeo_lr=0.001 destroys training** (+1.69 RMSE) — homeostasis fights ES at this scale
2. **decoder_scale=1.0** (+0.54) — too large causes unstable dynamics
3. **d_lr=0** (+0.15) — D gradient learning helps but isn't critical

Note: readout_gain=0.25 actually scored slightly better RMSE (0.556) but with 4x more spikes (18804 vs 4979), suggesting less efficient coding.

---

## Key Findings Across All Sessions

1. **sigma=0.1** is the ES sweet spot for this task
2. **F=D^T fixed** learns 10x faster than evolving F from random
3. **Wishart Omega init** is necessary for random init (prevents dead/hyperactive neuron splits)
4. **homeo_lr=0.001** is much better than 0.01 — homeostasis destabilizes after ~300 epochs
5. **D gradient learning** with running statistics works well (d_lr=0.01, d_update_interval=10)
6. **Poisson KL loss** significantly improves stochastic training over plain reconstruction loss
7. **decoder_scale=0.2, readout_gain=1.0** is the stable stochastic config (0.1 diverges)
8. **Posterior score connection confirmed**: V_i ≈ d_i^T * e with correlation 0.9998
9. **Pop_size diminishing returns** past ~1024 for stochastic 4D filtering

---

## April 7 — MNIST Baseline Validation

### Architecture
784-128-128-10 feedforward SNN, LIF neurons, soft reset, Poisson rate coding (T=25).
Learnable per-layer thresholds (PARAM), weight matrices (MM_PARAM, low-rank perturbation).

### Hyperparameter sweeps (best config)
```
.venv/bin/python -m spikyeggroll.train --dataset mnist --pop_size 1024 \
  --sigma 0.005 --lr 0.005 --rank 8 --epochs 500
```
| Parameter | Swept | Best |
|-----------|-------|------|
| sigma | 0.001–1.0 | 0.005–0.01 |
| pop_size | 64–16384 | 1024 |
| lr | 0.001–0.03 | 0.005 |
| threshold | 0.1–1.0 | 1.0 (initial, learnable) |
| rank | 1, 2, 4, 8, 16 | **8** |

**Best result: 75.6% test accuracy** (500ep, 47s, RTX 4080)

### Rank was the key lever
| rank | Accuracy |
|------|----------|
| 1 | 66.1% |
| 2 | 59.9% |
| 4 | 61.2% |
| **8** | **75.6%** |
| 16 | 74.7% |

### Per-class accuracy (500ep, rank=8)
```
0: 93.3%  1: 94.7%  6: 89.7%  2: 76.5%  3: 76.7%
7: 75.3%  9: 75.0%  4: 72.7%  5: 20.5%  8: 0.1%
```
Digits 5 and 8 nearly unlearned — output neurons go silent, ES gets zero gradient signal.
Digit 0 is a "sink" absorbing misclassifications from visually similar digits.

### Ablations
- **Learnable thresholds**: broke the frozen-threshold plateau (fitness -1.8 → -0.95)
- **Escape noise** (β=10): faster early convergence (71% at 200ep) but same plateau
- **Membrane readout**: 61% vs 73% spike counts — high-variance logits hurt ES
- **Output bias**: changed which neurons die, didn't fix the problem
- **Logit offset (+1.0)**: diluted signal, 64.4% vs 75.6%
- **Smaller/larger batches**: no improvement
- **Lower threshold**: saturates firing, 47%

### April 8 — Scaling + Diagnostics

**Dead neuron diagnosis**: Output neuron 8 completely dead (0.01 spikes). Hidden layers 1+2 have 0 dead neurons — problem is exclusively at the output layer. W3[8] has the largest norm (1.46) and lowest threshold (0.93) but wrong direction in 128-dim space.

**1/5th success rule**: Adaptive sigma via EMA-smoothed success rate (fraction of perturbations beating baseline). Uses 1.02× multiplier. Sigma stabilizes around 0.007–0.009.

**Centered rank fitness shaping**: Replace z-scored fitness with uniform ranks in [-0.5, 0.5]. Scale-free, outlier-robust. Combined with lr=0.001: **82.7% at 2000ep** (previous best was 80.3%).

**Per-layer perturbation analysis** (sigma=0.006, post-training):
```
linear1     (128,784): 17.2% success, range=0.118
linear2     (128,128): 17.6% success, range=0.095
linear_out  (10,128):  18.6% success, range=0.092
threshold1  (128,):    23.2% success, range=0.042
threshold2  (128,):    20.1% success, range=0.030
threshold_out (10,):   58.8% success, range=0.010
```

**Rank × pop_size grid** (200ep, sigma=0.007, lr=0.005, fixed threshold=1.0):
```
         pop=256  pop=512  pop=1024  pop=2048  pop=4096
rank=1    35.9%    54.3%    63.0%     68.3%     70.4%
rank=2    34.3%    43.3%    45.2%     70.0%     73.6%
rank=4    33.1%    43.0%    62.7%     78.1%     80.3%
rank=8    43.0%    53.7%    61.4%     68.5%     69.9%
rank=16   43.1%    59.3%    72.7%     77.6%     63.9%
```

**Large population breakthrough** (pop=10000, 400ep):
```
.venv/bin/python -m spikyeggroll.train --dataset mnist --pop_size 10000 \
  --sigma 0.007 --lr 0.005 --rank 2 --epochs 400
```
| rank | Accuracy | Dead neurons |
|------|----------|-------------|
| 1 | 75.0% | digit 8 dead |
| 2 | 91.2% | **none** |
| **3** | **91.9%** | **none** |
| 4 | 86.6% | none |

**Best result: 91.9% test accuracy** (rank=3, pop=10k, 400ep, 362s).
Large population solved the dead neuron problem — digit 8 alive at 89.0%.
All 10 classes above 83%.

### Layer 1 base optimization
Precompute `x @ W1.T` once outside vmap (shared across population).
45% speedup for rank=1 (1.1→1.6 ep/s). Moderate for higher ranks since layers 2+3 dominate.

### Memory analysis (pop=10000, batch=256, hidden=128)
```
LIF carry states (V1+V2+V_out+accum):  2.8 GB
Spike intermediates (s1+s2):           2.6 GB
Current intermediates (I1+I2+I_out):   2.7 GB
LoRA noise (rank=2):                   0.1 GB
Peak during forward:                   ~8.3 GB
```
Memory dominated by N×B×H tensors, not LoRA rank. Chunked evaluation needed for rank≥2 at pop=10k on 16GB GPU.

### Open questions
- SG-BPTT baseline needed for comparison (~99.5% expected)
- CIFAR-10 / ResNet-18 experiment deferred
- Per-layer sigma adaptation would help (threshold_out needs 3× more sigma)
- Chunked update step needed for pop>10k (EGGROLL replays all N noise vectors at once)

---

## April 11 — CIFAR-10 / Spiking ResNet18 Remote Runs

### Implementation state

- The old residual MLP path was replaced by a real CIFAR-sized convolutional
  spiking ResNet path.
- CIFAR encoding now preserves image structure over time.
- Conv kernels use the dedicated conv ES primitive instead of full-tensor
  parameter noise.
- Group norm was adopted for eval stability.

### 4080 findings

- The larger pulled-repo smoke config
  `pop=256, rank=2, sigma=0.01, T=8, batch=64, chunk=128`
  OOMed on a 16 GB RTX 4080 with an attempted allocation of about `30.26 GiB`.
- A reduced smoke run with
  `pop=32, rank=2, T=4, batch=8, chunk=8`
  completed and produced about `10.94%` test accuracy.
- A 5-epoch learn check on the same reduced config stayed around `9-11%`.

Takeaway:
- The current conv ResNet path fits and runs on small 4080 configs.
- Small-population CIFAR runs do not learn meaningfully.

### Early 5090 findings

- `pop=64, rank=2, T=4, batch=16, chunk=16` reached `11.33%`.
- `pop=128, rank=2, T=8, batch=16, chunk=16` ran for 100 updates and peaked at
  only `11.48%`.
- Screening runs with `width=32`, `T=16`, `augment=True`, and larger
  populations (`1024`, `2048`) showed healthy ES signal but only weak early test
  accuracy (`~11.5-12.1%` at the first test point).

Takeaway:
- More VRAM alone does not fix CIFAR learning.
- The network is no longer dead, but the original CIFAR config was far too weak.

### Current strongest run

Run:
- `cifar5090-r2-p4096-t16-c32-b32-k96-20260411`

Config:
- `pop_size=4096`
- `rank=2`
- `sigma=0.006`
- `lr=0.0015`
- `timesteps=16`
- `batch_size=32`
- `chunk_size=96`
- `augment=True`
- `resnet_channels_base=32`

Metrics snapshot:
- First epoch time: about `114 s`
- Later average epoch time: about `35 s`
- GPU utilization: `99%`
- GPU memory used: about `17 / 32.6 GiB`
- Best test accuracy so far: **21.18%** at epoch 80
- Last recorded test accuracy: `19.00%` at epoch 100
- Latest sigma seen: about `0.021`

Takeaway:
- This is the first CIFAR run clearly above chance.
- The model is learning something now.
- Performance is still poor and has become unstable as sigma drifted upward.

### Main conclusions

1. The problem is no longer "dead network at initialization." The new conv path
   has nonzero ES signal and can learn above chance.
2. The current training loop is still a poor fit for CIFAR-scale conv SNNs.
   One logged epoch is only one minibatch ES update, so the run has seen far
   fewer effective training passes than the epoch count suggests.
3. Sigma drift is a real issue. The 1/5th rule helps early but needs a cap.
4. The current loop still uses z-score fitness shaping, while the stronger MNIST
   experiments benefited from centered-rank shaping.
5. The model currently uses group norm for stability, but stronger CIFAR SNN
   literature usually relies on BN/BNTT-like normalization.
6. The 5090 run is compute-bound, not memory-bound. Extra VRAM headroom does not
   imply unused GPU compute.

### Next priorities

- Add a sigma ceiling.
- Switch to centered-rank shaping.
- Revisit what "epoch" means for CIFAR training.
- Explore faster update paths and lower precision.
- Treat larger batch/chunk settings as throughput tuning, not as the main fix
  for poor CIFAR accuracy.

### Planning handoff

- The detailed implementation roadmap for the next CIFAR ResNet work now lives
  in [docs/cifar-resnet-phased-implementation-plan.md](docs/cifar-resnet-phased-implementation-plan.md).
- That document is the primary source of truth for phase ordering, acceptance
  criteria, and the 5090 validation matrix.

---

## April 12 — CIFAR-10 / Spiking ResNet18 Startup Tracing and Heavy 5090 Runs

### Code instrumentation added

- Added startup tracing in `spikyeggroll/train.py` with timestamped markers
  around:
  - model init
  - noiser init
  - dataset load
  - JIT wrapper creation
  - prefetch warmup
  - start-metric write
  - first eval forward
  - first population scoring step
  - first update step
- Added `jax.block_until_ready(...)` at the traced milestones so log ordering
  reflects actual device completion rather than queued dispatch.
- Added gated device-memory snapshots during startup.
- Added a bounded startup trace mode and optional profiler server:
  - `SPIKYEGGROLL_TRACE_STARTUP=1`
  - `SPIKYEGGROLL_PROFILE_STARTUP=1`
  - `SPIKYEGGROLL_PROFILE_TRACE=1`
  - `SPIKYEGGROLL_PROFILE_SERVER_PORT=<port>`
- These env vars were part of the original stall-debugging patch. The current
  profiling interface is CLI/config driven via `--profile_mode`,
  `--profile_trace_dir`, `--profile_server_port`, `--profile_max_snapshots`,
  `--profile_warmup_updates`, `--profile_updates_window`, and
  `--profile_eval_once`.
- Commits pushed:
  - `fc80b53` `Add startup tracing for CIFAR training hangs`
  - `b802030` `Add gated startup memory profiling`
  - `db3e1c6` `Add startup trace capture controls`

### Host-specific startup behavior

#### Bad host: `216.249.100.66:21657`

- Multiple heavy runs appeared to "hang" before normal training logs:
  - `float32`, `batch_size=64`, `chunk_size=128`
  - `bfloat16`, `batch_size=48`, `chunk_size=128`
  - `bfloat16`, `batch_size=48`, `chunk_size=96`
- Symptoms:
  - process alive for hours
  - GPU at `97-99%`
  - GPU memory pinned around `29.4 / 32.6 GiB`
  - no metrics file written
  - log advanced only with repeated JAX/XLA warnings
- Takeaway:
  - this was not a normal "slow first epoch"
  - the behavior was host/runtime specific enough to justify instrumentation

#### Good host: `64.228.13.219:61195`

- The same heavy traced config cleared startup and entered normal training:
  - `pop_size=4096`
  - `rank=2`
  - `timesteps=16`
  - `batch_size=48`
  - `chunk_size=96`
  - `dtype=bfloat16`
- The run wrote the startup trace markers all the way through:
  - `start metric written`
  - first `jit_forward_eval`
  - first `score_population_chunked`
  - first `jit_update`
- Important conclusion:
  - the code path is not fundamentally deadlocking
  - the earlier "stall" was at least partly machine-specific

### Startup / compile timing from traced run

Run:
- `cifar5090-r2-p4096-t16-c32-b48-k96-bf16-trace2-20260412`

Observed startup milestones on the good 5090 host:
- `train() begin` at `16:35:39`
- `model init complete` at `16:35:43`
- `dataset loaded` at `16:35:47`
- `start metric written` at `16:35:48`
- first `jit_forward_eval` ready at `16:35:49`
- first `population scores` ready at `16:36:15`
- first `jit_update` ready at `16:36:18`
- bounded startup trace stopped at `16:37:02`

Approximate timings:
- model init: `~3 s`
- dataset load/download path on fresh host: `~4 s`
- prefetch warmup: `~1 s`
- first eval forward: `~1 s`
- first population score: `~26 s`
- first update: `~3 s`

Takeaway:
- the expensive startup region is the first population scoring/update path
- dataset load and prefetch are not the bottleneck

### Metrics path gotcha

- Stdout logs were written under `/workspace/logs/spikyeggroll/...`
- Metrics and checkpoints for direct `python -m spikyeggroll.train` runs were
  still being written under repo-relative defaults:
  - `/workspace/spikyeggrolls/logs/spikyeggroll/...`
  - `/workspace/spikyeggrolls/checkpoints/spikyeggroll/...`
- This caused an initial false impression that the good host had "no metrics"
  when the run was actually progressing.

### Current heavy run results on the good 5090 host

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
- GPU utilization: `99%`
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

Current best:
- best observed test accuracy on this run: `22.33%` at `epoch 20`

### Sigma / exploration behavior

- Sigma shrank steadily during the run:
  - `epoch 20`: `0.00492`
  - `epoch 21`: `0.00404`
  - `epoch 22`: `0.00331`
  - `epoch 23`: `0.00272`
  - `epoch 24`: `0.00223`
  - `epoch 25`: `0.00183`
  - `epoch 27`: `0.00123`
  - `epoch 55`: `0.00100` (floor)
- Raw score std also collapsed:
  - `epoch 20`: `0.00775`
  - `epoch 27`: `0.00049`
  - `epoch 55`: `0.00024`

Takeaway:
- The run still learns slowly even after sigma collapses.
- Exploration is now effectively clamped at the sigma floor.
- This makes the current sigma adaptation rule a likely limiter for later-stage
  progress.

### Effective data exposure

- At `global_update=560` and `batch_size=48`, the run has processed about
  `26,880` sampled training examples.
- Relative to the CIFAR-10 train set (`50,000`), this is only about
  `0.54` effective full passes over the dataset.
- Important reminder:
  - logged "epoch" in this loop is **not** a full data epoch
  - `epoch 55` here means `560` minibatch ES updates, not `55` full CIFAR passes

### Updated conclusions

1. The heavy CIFAR config does run and train on at least one 5090 host.
2. The previous startup "hang" should be treated as a host/runtime issue, not a
   universal codepath deadlock.
3. The most expensive region is first population scoring, not dataset loading or
   prefetch.
4. The current heavy run is the new best-documented CIFAR run and is roughly
   matching the earlier `21.18%` result while using the newer throughput path.
5. Sigma collapse now looks like the main optimizer-level problem on long runs.
