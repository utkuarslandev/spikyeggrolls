"""SNN model as an EGGROLL Model subclass.

Architecture: 784-128-128-10 feedforward SNN with LIF neurons.

Neuron dynamics (per layer, per timestep t):
    V^t = β · V^{t-1} + W · s^t_in          (leaky integration)
    o^t = 1[V^t > θ]                          (deterministic spike)
    V^t ← V^t - o^t · θ                       (soft reset)

Soft reset subtracts threshold on spike, preserving supra-threshold residual.
This gives smoother gradients than hard reset (V^t ← 0).

Output readout (two modes):
    spike_counts: Σ_t o^t_out  — standard but discontinuous
    membrane:     Σ_t V^t_out  — leaky integrator, smoother fitness landscape

Fitness: -CE(softmax(readout), y). Cross-entropy is smooth in readout space;
accuracy (argmax == label) is a step function unsuitable for ES gradient estimation.

Escape noise (optional, for SG-BPTT baseline only — not recommended for ES):
    o^t ~ Bern(λ₀ · exp(β_esc · (V^t - θ)))  (stochastic firing)
    Rate clamped to [0, 1] per timestep.
"""

import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import Model, CommonInit, CommonParams
from hyperscalees.models.common import (
    Linear,
    Parameter,
    merge_inits,
    merge_frozen,
    call_submodule,
    PARAM,
    EXCLUDED,
)

from spikyeggroll.configs import SNNConfig


def lif_step(V, I, beta, threshold):
    """Deterministic LIF: V = βV + I, spike if V > θ, soft reset."""
    V = beta * V + I
    spikes = (V > threshold).astype(V.dtype)
    V = V - spikes * threshold
    return V, spikes


def lif_step_escape(V, I, beta, threshold, key, escape_beta=50.0, escape_lambda0=1.0):
    """Stochastic LIF with escape noise.

    Spike probability: p = clamp(λ₀ · exp(β_esc · (V - θ)), 0, 1).
    """
    V = beta * V + I
    log_rate = jnp.clip(escape_beta * (V - threshold), -20.0, 0.0)
    spike_prob = jnp.clip(escape_lambda0 * jnp.exp(log_rate), 0.0, 1.0)
    spikes = (jax.random.uniform(key, V.shape) < spike_prob).astype(V.dtype)
    V = V - spikes * threshold
    return V, spikes


class SNNModel(Model):
    """Feedforward SNN with 2 hidden LIF layers for temporal classification.

    Architecture:
        Input [B, T, n_inputs] → Linear₁ → LIF₁ → Linear₂ → LIF₂ → Linear_out → LIF_out
        Output: spike counts [B, n_classes] or accumulated membrane [B, n_classes]
    """

    @classmethod
    def rand_init(cls, key, cfg: SNNConfig):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        dtype = cfg.dtype

        init = merge_inits(
            linear1=Linear.rand_init(k1, cfg.n_inputs, cfg.hidden_size, False, dtype),
            linear2=Linear.rand_init(k2, cfg.hidden_size, cfg.hidden_size, False, dtype),
            linear_out=Linear.rand_init(k3, cfg.hidden_size, cfg.n_classes, False, dtype),
        )

        init = merge_frozen(
            init,
            beta=cfg.beta,
            threshold=cfg.threshold,
            timesteps=cfg.timesteps,
            membrane_readout=cfg.membrane_readout,
            escape_noise=cfg.escape_noise,
            escape_beta=cfg.escape_beta,
            escape_lambda0=cfg.escape_lambda0,
            stoch_key=jax.random.key(cfg.seed + 2000),
        )

        return init

    @classmethod
    def _forward(cls, common_params: CommonParams, x, l1_base=None):
        """Forward pass over spike train.

        Args:
            common_params: EGGROLL common params
            x: input spikes [B, T, n_inputs]
            l1_base: optional precomputed x @ W1.T [T, B, hidden] to avoid
                     redundant base matmul when vmapped across population

        Returns:
            logits: [B, n_classes] — spike counts or accumulated membrane potential
        """
        B = x.shape[0]
        T = common_params.frozen_params["timesteps"]
        beta = common_params.frozen_params["beta"]
        use_escape = common_params.frozen_params["escape_noise"]
        use_membrane = common_params.frozen_params["membrane_readout"]
        esc_beta = common_params.frozen_params["escape_beta"]
        esc_lambda0 = common_params.frozen_params["escape_lambda0"]

        # Fixed threshold (frozen, shared across all layers)
        threshold = common_params.frozen_params["threshold"]
        thr1 = threshold
        thr2 = threshold
        thr_out = threshold

        hidden_size = common_params.params["linear1"]["weight"].shape[0]
        n_classes = common_params.params["linear_out"]["weight"].shape[0]

        # Derive stochastic key — shared across population per epoch
        rng_key = None
        if use_escape:
            base_key = common_params.frozen_params["stoch_key"]
            if common_params.iterinfo is not None:
                epoch, _ = common_params.iterinfo
                rng_key = jax.random.fold_in(base_key, epoch)
            else:
                rng_key = base_key

        # Layer 1 LoRA delta helper: only compute the perturbation, add to precomputed base
        has_l1_base = l1_base is not None
        if has_l1_base and common_params.iterinfo is not None:
            from hyperscalees.noiser.eggroll import get_lora_update_params
            l1_key = common_params.es_tree_key["linear1"]["weight"]
            l1_param = common_params.params["linear1"]["weight"]
            l1_sigma = common_params.noiser_params["sigma"] / jnp.sqrt(common_params.frozen_noiser_params["rank"])
            l1_A, l1_B = get_lora_update_params(
                common_params.frozen_noiser_params, l1_sigma,
                common_params.iterinfo, l1_param, l1_key)

        # Initial state
        V1 = jnp.zeros((B, hidden_size))
        V2 = jnp.zeros((B, hidden_size))
        V_out = jnp.zeros((B, n_classes))
        accum = jnp.zeros((B, n_classes))

        if use_escape:
            def scan_fn(carry, inp):
                V1, V2, V_out, accum = carry
                x_t, t_idx = inp
                k1, k2, k3 = jax.random.split(jax.random.fold_in(rng_key, t_idx), 3)

                I1 = call_submodule(Linear, "linear1", common_params, x_t)
                V1, s1 = lif_step_escape(V1, I1, beta, thr1, k1, esc_beta, esc_lambda0)

                I2 = call_submodule(Linear, "linear2", common_params, s1)
                V2, s2 = lif_step_escape(V2, I2, beta, thr2, k2, esc_beta, esc_lambda0)

                I_out = call_submodule(Linear, "linear_out", common_params, s2)
                V_out, s_out = lif_step_escape(V_out, I_out, beta, thr_out, k3, esc_beta, esc_lambda0)

                accum = accum + (V_out if use_membrane else s_out)
                return (V1, V2, V_out, accum), None

            x_t = jnp.transpose(x, (1, 0, 2))
            (_, _, _, accum), _ = jax.lax.scan(
                scan_fn, (V1, V2, V_out, accum), (x_t, jnp.arange(T))
            )
        else:
            if has_l1_base and common_params.iterinfo is not None:
                # Optimized: use precomputed base for layer 1, only add LoRA delta
                def scan_fn(carry, inp):
                    V1, V2, V_out, accum = carry
                    x_t, base_t = inp

                    # Layer 1: base (precomputed) + LoRA delta
                    I1 = base_t + x_t @ l1_B @ l1_A.T
                    V1, s1 = lif_step(V1, I1, beta, thr1)

                    I2 = call_submodule(Linear, "linear2", common_params, s1)
                    V2, s2 = lif_step(V2, I2, beta, thr2)

                    I_out = call_submodule(Linear, "linear_out", common_params, s2)
                    V_out, s_out = lif_step(V_out, I_out, beta, thr_out)

                    accum = accum + (V_out if use_membrane else s_out)
                    return (V1, V2, V_out, accum), None

                x_t = jnp.transpose(x, (1, 0, 2))
                (_, _, _, accum), _ = jax.lax.scan(
                    scan_fn, (V1, V2, V_out, accum), (x_t, l1_base)
                )
            else:
                # Standard path (eval or no precomputed base)
                def scan_fn(carry, x_t):
                    V1, V2, V_out, accum = carry

                    I1 = call_submodule(Linear, "linear1", common_params, x_t)
                    V1, s1 = lif_step(V1, I1, beta, thr1)

                    I2 = call_submodule(Linear, "linear2", common_params, s1)
                    V2, s2 = lif_step(V2, I2, beta, thr2)

                    I_out = call_submodule(Linear, "linear_out", common_params, s2)
                    V_out, s_out = lif_step(V_out, I_out, beta, thr_out)

                    accum = accum + (V_out if use_membrane else s_out)
                    return (V1, V2, V_out, accum), None

                x_t = jnp.transpose(x, (1, 0, 2))
                (_, _, _, accum), _ = jax.lax.scan(
                    scan_fn, (V1, V2, V_out, accum), x_t
                )

        return accum
