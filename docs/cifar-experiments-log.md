# CIFAR Experiments Log

## Summary

The current `cifar10 + spiking_resnet18` path does not begin learning under the
tested EGGROLL settings. The failure mode is not just "low accuracy"; the model
starts in a dead-output regime where the classifier emits all-zero outputs, the
cross-entropy stays fixed at `-log(10)`, and the ES population produces no
fitness spread.

This log records the observed runs, the direct diagnostic probe results, and the
current hypotheses grounded in the code.

## Runs Observed

### Run: `cifar10-resnet18-tune-a`

Configuration:

- `DATASET=cifar10`
- `MODEL_NAME=spiking_resnet18`
- `POP_SIZE=512`
- `RANK=2`
- `SIGMA=0.01`
- `LR=0.002`
- `EPOCHS=100`
- `TIMESTEPS=4`
- `BATCH_SIZE=64`
- `CHUNK_SIZE=128`

Observed behavior:

- `Final test accuracy: 0.1000`
- `fitness` remained `-2.3026`
- `better: 0/512` throughout
- sigma decayed steadily downward

Interpretation:

- `-2.3026` is `-log(10)`, consistent with equal logits across all 10 classes
- no population member outperformed the base model, so the ES loop had no useful
  ranking signal

### Run: `cifar10-resnet18-tune-2048`

Configuration:

- `DATASET=cifar10`
- `MODEL_NAME=spiking_resnet18`
- `POP_SIZE=2048`
- `RANK=2`
- `SIGMA=0.01`
- `LR=0.002`
- `EPOCHS=100`
- `TIMESTEPS=4`
- `BATCH_SIZE=64`
- `CHUNK_SIZE=256`

Observed behavior:

- `Final test accuracy: 0.1000`
- `fitness` remained `-2.3026`
- `better: 0/2048` throughout
- increasing population size alone did not recover learning

Interpretation:

- the failure is not explained by small population size alone
- the CIFAR path is collapsing before EGGROLL can extract a gradient estimate

## Direct Diagnostic Probe

A direct inline probe was run on a fresh pod against the GitHub checkout using a
real CIFAR-10 batch and the current default deep CIFAR architecture:

- `timesteps=4`
- `resnet_width=768`
- `resnet_blocks=8`
- `batch_size=64`
- `pop_size=128` for the probe population stats

### Base Model Output

Measured values:

- `fitness = -2.3025851`
- `acc = 0.109375`
- `output_nonzero_fraction = 0.0`
- `output_mean = 0.0`
- `output_max = 0.0`
- `output_class_variance_mean = 0.0`

Observed predictions:

- the first 16 predictions were all class `0`
- sample output row sums were all `0.0`

Interpretation:

- the base model emits all-zero output spike counts at initialization
- cross-entropy is flat because every class logit is identical

### Population Behavior

Measured values:

- `raw_score_mean = -2.3025851`
- `raw_score_std = 0.0`
- `raw_score_min = -2.3025851`
- `raw_score_max = -2.3025851`
- `n_better_than_base = 0`
- `population_output_variance_mean = 0.0`

Interpretation:

- every perturbed population member also emits all-zero outputs
- EGGROLL receives exactly zero fitness spread
- the optimizer cannot start learning from this state

### Spike Activity by Depth

Measured spike rates:

- stem spike rates: `0.0585, 0.1452, 0.1662, 0.1684`
- early residual blocks still show some activity
- last residual block spike rates: `0.0, 0.0, 0.0, 0.0`
- output spike rates: `0.0, 0.0, 0.0, 0.0`

Interpretation:

- the input is not dead
- activity dies inside the residual stack
- the classifier never receives a live spike signal

## Code-Level Findings

### 1. The CIFAR path is much deeper than the MNIST path

Relevant code:

- [spikyeggroll/models/spiking_resnet.py](../spikyeggroll/models/spiking_resnet.py)
- [spikyeggroll/models/snn.py](../spikyeggroll/models/snn.py)

Facts:

- MNIST uses a shallow 3-layer spiking MLP
- CIFAR uses a stem, `8` residual blocks, and an output layer
- each CIFAR residual block applies multiple `lif_step(...)` thresholding
  operations

Conclusion:

- CIFAR has a much harsher dynamical path than MNIST

### 2. CIFAR defaults to only 4 timesteps

Relevant code:

- [scripts/run_train.sh](../scripts/run_train.sh)

Facts:

- for `DATASET=cifar10` and `MODEL_NAME=spiking_resnet18`, the launcher applies:
  - `TIMESTEPS=4`
  - `BATCH_SIZE=64`
  - `CHUNK_SIZE=128`

Conclusion:

- the default deep CIFAR model gets very few temporal opportunities to accumulate
  membrane and maintain spike activity

### 3. The model is bias-free and thresholded at 1.0 throughout

Relevant code:

- [spikyeggroll/models/spiking_resnet.py](../spikyeggroll/models/spiking_resnet.py)
- [hyperscalees/models/common.py](../hyperscalees/models/common.py)

Facts:

- all `Linear` layers are created with `use_bias=False`
- threshold is fixed from config and shared across the stack
- the default threshold is `1.0`

Conclusion:

- a deep stack of hard-threshold units with no bias makes dead-output behavior at
  initialization plausible

### 4. The residual merge is not an identity-style skip path

Relevant code:

- [spikyeggroll/models/spiking_resnet.py](../spikyeggroll/models/spiking_resnet.py)

Key line:

```python
v3, s = lif_step(v_r[i], s + s2, beta, threshold)
```

Conclusion:

- the skip connection is immediately re-thresholded through another LIF state
- this does not preserve analog signal like a standard residual connection would
- it likely contributes to signal death across depth

### 5. The sigma schedule reinforces collapse once it starts

Relevant code:

- [spikyeggroll/train.py](../spikyeggroll/train.py)

Facts:

- success rate is based on `raw_scores > val_fitness`
- when no perturbation beats the base model, `n_better = 0`
- if `ema_success < 0.2`, sigma is divided by `1.02`

Conclusion:

- after entering a flat-logit regime, the code reduces exploration every epoch
- this makes recovery from dead outputs less likely

### 6. The CIFAR input path is minimal

Relevant code:

- [spikyeggroll/data/cifar10.py](../spikyeggroll/data/cifar10.py)

Facts:

- images are only normalized to `[0, 1]`
- they are Poisson-encoded directly
- `augment` is currently ignored

Conclusion:

- the input pipeline is intentionally simple and may be too weak for the current
  deep hard-spiking architecture

### 7. Tests do not cover this failure mode

Relevant code:

- [tests/test_cifar10.py](../tests/test_cifar10.py)

Facts:

- current CIFAR tests only cover shape and finiteness
- there is no regression test for:
  - nonzero output activity
  - output variance at init
  - population fitness spread
  - short-run CIFAR fitness improvement

Conclusion:

- this collapse can exist while the test suite still passes

## Current Working Hypothesis

The current CIFAR model starts in a dead-output regime because the architecture
is too deep and too threshold-heavy for the default `T=4` setup. Activity is
present in the stem and early blocks, but it collapses by the later residual
blocks, leaving the output layer with zero spikes. Once outputs are flat,
EGGROLL receives no ranking signal and the sigma adaptation rule shrinks
exploration further, locking the run into failure.

## Recommended Next Experiments

These are the next experiments to run before attempting another large CIFAR job:

1. Reduce depth sharply:
   - `resnet_blocks=2` or `4`
2. Increase timesteps:
   - `timesteps=8` or `12`
3. Reduce width while debugging:
   - `resnet_width=128` or `256`
4. Re-run the diagnostic probe before full training:
   - confirm nonzero output activity
   - confirm nonzero population score variance
5. Consider a sigma floor or disabling sigma shrink during early CIFAR training
6. Consider testing `--membrane_readout` specifically for CIFAR bootstrap

Suggested first debug configuration:

```bash
DATASET=cifar10 \
MODEL_NAME=spiking_resnet18 \
POP_SIZE=512 \
RANK=2 \
SIGMA=0.01 \
LR=0.002 \
EPOCHS=100 \
TIMESTEPS=8 \
BATCH_SIZE=64 \
CHUNK_SIZE=128 \
.venv/bin/python -m spikyeggroll.train \
  --dataset cifar10 \
  --model_name spiking_resnet18 \
  --resnet_width 128 \
  --resnet_blocks 2
```

This is not intended as an optimal CIFAR setup. It is intended to answer a more
basic question first: can the CIFAR path produce live output activity and
non-degenerate ES fitness variation at all?
