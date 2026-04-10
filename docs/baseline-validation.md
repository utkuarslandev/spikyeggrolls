# Baseline Validation: EGGROLL on Static MNIST

Can EGGROLL optimize spiking neural networks at all, and how close does it get to surrogate-gradient BPTT on well-understood benchmarks?

## Architecture

Feedforward SNN (784-128-128-10), LIF neurons, soft reset, rate coding.

### Neuron dynamics

Per layer $\ell$, per timestep $t = 1, \ldots, T$:

$$I_\ell^t = W_\ell \cdot o_{\ell-1}^t$$

$$V_\ell^t = \beta \cdot V_\ell^{t-1} + I_\ell^t$$

$$o_\ell^t = \mathbf{1}[V_\ell^t > \theta_\ell]$$

$$V_\ell^t \leftarrow V_\ell^t - o_\ell^t \cdot \theta_\ell \qquad \text{(soft reset)}$$

where $\beta = 0.95$ (membrane decay, frozen) and $\theta = 1.0$ (single scalar threshold, frozen and shared across layers).

Soft reset preserves the supra-threshold residual $V - \theta$ after a spike. This is smoother than hard reset ($V \leftarrow 0$) and more amenable to ES optimization.

### Input encoding

Poisson rate coding: each pixel $p_i \in [0,1]$ generates i.i.d. spikes $s_i^t \sim \text{Bern}(p_i)$ for $T = 25$ timesteps. Re-encoded stochastically each mini-batch for regularization.

### Output readout

Spike count decoding: $\hat{y} = \arg\max_c \sum_{t=1}^T o_{\text{out},c}^t$.

Alternative (membrane readout: $\sum_t V_{\text{out}}^t$) was tested but performs worse with ES due to high-variance logits. Spike counts are naturally bounded and low-variance.

### Fitness function

Negative cross-entropy: $f = -\text{CE}(\text{softmax}(\mathbf{counts}), y)$.

CE is smooth in logit space — small weight changes shift class probabilities continuously. Accuracy ($\mathbf{1}[\arg\max = y]$) is a step function, unusable for ES gradient estimation.

### Parameters

| Component | Shape | Count | ES type |
|-----------|-------|-------|---------|
| $W_1$ | $128 \times 784$ | 100,352 | MM\_PARAM (low-rank) |
| $W_2$ | $128 \times 128$ | 16,384 | MM\_PARAM (low-rank) |
| $W_3$ | $10 \times 128$ | 1,280 | MM\_PARAM (low-rank) |
| **Total** | | **118,016** | |

### EGGROLL perturbation

Weight matrices perturbed as $\tilde{W} = W + \sigma A B^\top / \sqrt{r}$ where $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{n \times r}$ are i.i.d. Gaussian. For $W_1$ (128×784) with $r = 8$: each perturbation is rank-8, requiring $128 \times 8 + 784 \times 8 = 7296$ scalars instead of 100,352 — a 14× compression.

Only weight matrices are perturbed in this configuration; threshold is frozen at
inference and training time.

Implementation note: the current code path freezes `threshold` in
`spikyeggroll/models/snn.py` via `merge_frozen(...)`, then reuses that same scalar
as `thr1`, `thr2`, and `thr_out` in the forward pass.

## Experiment 1 — Hyperparameter sweeps

Baseline: ~99.5% (SG-BPTT).

### Best configuration found

| Parameter | Value |
|-----------|-------|
| pop_size | 10000 |
| sigma | 0.007 |
| lr | 0.005 |
| rank | 3 |
| threshold | 1.0 (frozen) |
| beta | 0.95 (frozen) |
| batch_size | 256 |
| timesteps | 25 |
| epochs | 400 |
| fitness shaping | z-scored |
| sigma adaptation | 1/5th rule (EMA, 1.02×) |

**Best result: 93.7% test accuracy** (4000 epochs, pop=10000, rank=3, lr=0.001, ~60 min on RTX 4080).
All 10 digit classes above 83%. No dead output neurons.

With lr=0.005: 92.4% at epoch 900 (peaks then degrades due to sigma runaway).
With lr=0.001: 93.7% at epoch 3700 (slower start, stable sigma, higher ceiling).

### Learning curve (best config)

```
epoch  train_acc  test_acc  sigma
    0    7.8%     14.2%    0.007
  100   85.2%     82.9%    0.010
  300   91.4%     90.5%    0.011
  500   89.8%     91.9%    0.011
  900   93.0%     92.4%    0.013  ← peak test
 1500   90.6%     91.6%    0.015
 2500   91.0%     90.9%    0.021  (sigma drift begins)
 3900   89.8%     87.8%    0.036  (sigma runaway degrades test)
```

Test accuracy peaks at epoch ~900 then slowly degrades as the 1/5th sigma
adaptation rule drives sigma upward. Capping sigma at ~0.015 would preserve
the peak.

### Rank × pop_size grid (200 epochs)

```
         pop=256  pop=512  pop=1024  pop=2048  pop=4096
rank=1    35.9%    54.3%    63.0%     68.3%     70.4%
rank=2    34.3%    43.3%    45.2%     70.0%     73.6%
rank=4    33.1%    43.0%    62.7%     78.1%     80.3%
rank=8    43.0%    53.7%    61.4%     68.5%     69.9%
rank=16   43.1%    59.3%    72.7%     77.6%     63.9%
```

At pop=10000 (400 epochs):

| rank | Accuracy |
|------|----------|
| 1 | 75.0% |
| **2** | **91.2%** |
| **3** | **91.9%** |
| 4 | 86.6% |

Large population (10k) was the key breakthrough — it solved the dead output
neuron problem that plagued all smaller populations. With pop=1024, digit 8
had 0% accuracy; with pop=10000, all digits above 83%.

### Sweep results

**Sigma** (pop=256, rank=1, 200ep): sweet spot at 0.005–0.01, dead above 0.05.

**Learning rate** (pop=1024, sigma=0.005, rank=1, 500ep): 0.005 best (71.6%).

**Threshold** (pop=1024, 500ep): 1.0 best. Lower thresholds saturate firing.

### Ablations

**Escape noise** (stochastic LIF): Faster early convergence (71% at 200ep vs 50%
deterministic) but same plateau. Not recommended for ES.

**Membrane readout** (Σ V_out instead of Σ spikes): 61% vs 73% with spike counts.
High-variance logits hurt ES gradient estimation.

**Centered rank fitness shaping**: Scale-free alternative to z-scoring. Similar
performance (~92% with both).

**1/5th success rule**: Adaptive sigma based on fraction of perturbations beating
baseline. Works well short-term but sigma drifts upward over long training.
Needs a cap or decay schedule.

**Layer 1 base optimization**: Precompute `x @ W1.T` once outside vmap (input is
shared across population). 45% speedup for rank=1. Moderate for higher ranks.

### Dead neuron analysis

With pop=1024, output neuron 8 was completely dead (0.01 spikes, 0% accuracy).
Hidden layers had 0 dead neurons — the problem was exclusively at the output layer.
W3[8] had the largest weight norm but wrong direction in 128-dim space. ES with
small populations couldn't find the rotation to fix it.

Pop=10000 solved this: enough perturbation directions to escape the dead neuron
basin. All 10 output neurons active with 9+ spikes on average.

### Memory profile (pop=10000, batch=256, hidden=128)

```
LIF carry states (V1+V2+V_out+accum):  2.8 GB
Spike intermediates (s1+s2):           2.6 GB
Current intermediates (I1+I2+I_out):   2.7 GB
LoRA noise (rank=3):                   0.2 GB
Peak during forward:                   ~8.5 GB
```

Memory dominated by N×B×H tensors, not LoRA rank. 16 GB GPU fits pop=10000
at rank≤3. Chunked evaluation needed for rank≥4.

