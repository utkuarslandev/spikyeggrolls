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
