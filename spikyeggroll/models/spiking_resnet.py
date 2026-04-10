"""Deep residual spiking network for CIFAR-10 scaling experiments.

This model uses residual blocks built from EGGROLL-compatible Linear modules so
it can share the same ES/noiser pipeline as the MNIST model.
"""

import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import Model, CommonParams
from hyperscalees.models.common import (
    Linear,
    merge_inits,
    merge_frozen,
    call_submodule,
)

from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import lif_step


class SpikingResNet18Model(Model):
    """Residual deep spiking MLP with 8 residual blocks.

    Input is expected as flattened CIFAR spikes [B, T, 3072].
    """

    @classmethod
    def rand_init(cls, key, cfg: SNNConfig):
        blocks = int(cfg.resnet_blocks)
        width = int(cfg.resnet_width)
        total_keys = 2 + blocks * 2
        keys = jax.random.split(key, total_keys)
        dtype = cfg.dtype

        all_inits = {
            "linear1": Linear.rand_init(keys[0], cfg.n_inputs, width, False, dtype),
            "linear_out": Linear.rand_init(keys[1], width, cfg.n_classes, False, dtype),
        }
        for i in range(blocks):
            all_inits[f"block{i}_a"] = Linear.rand_init(
                keys[2 + i * 2], width, width, False, dtype
            )
            all_inits[f"block{i}_b"] = Linear.rand_init(
                keys[2 + i * 2 + 1], width, width, False, dtype
            )

        init = merge_inits(**all_inits)

        init = merge_frozen(
            init,
            beta=cfg.beta,
            threshold=cfg.threshold,
            timesteps=cfg.timesteps,
            membrane_readout=cfg.membrane_readout,
            resnet_blocks=blocks,
            resnet_width=width,
        )
        return init

    @classmethod
    def _forward(cls, common_params: CommonParams, x, l1_base=None):
        bsz = x.shape[0]
        timesteps = common_params.frozen_params["timesteps"]
        beta = common_params.frozen_params["beta"]
        threshold = common_params.frozen_params["threshold"]
        use_membrane = common_params.frozen_params["membrane_readout"]
        n_blocks = int(common_params.frozen_params["resnet_blocks"])
        width = int(common_params.frozen_params["resnet_width"])
        n_classes = common_params.params["linear_out"]["weight"].shape[0]

        has_l1_base = l1_base is not None
        if has_l1_base and common_params.iterinfo is not None:
            from hyperscalees.noiser.eggroll import get_lora_update_params

            l1_key = common_params.es_tree_key["linear1"]["weight"]
            l1_param = common_params.params["linear1"]["weight"]
            l1_sigma = common_params.noiser_params["sigma"] / jnp.sqrt(
                common_params.frozen_noiser_params["rank"]
            )
            l1_A, l1_B = get_lora_update_params(
                common_params.frozen_noiser_params,
                l1_sigma,
                common_params.iterinfo,
                l1_param,
                l1_key,
            )

        v_stem = jnp.zeros((bsz, width))
        v_out = jnp.zeros((bsz, n_classes))
        v_block_a = jnp.zeros((n_blocks, bsz, width))
        v_block_b = jnp.zeros((n_blocks, bsz, width))
        v_block_res = jnp.zeros((n_blocks, bsz, width))
        accum = jnp.zeros((bsz, n_classes))

        def _scan_blocks(v_a, v_b, v_r, s):
            for i in range(n_blocks):
                i1 = call_submodule(Linear, f"block{i}_a", common_params, s)
                v1, s1 = lif_step(v_a[i], i1, beta, threshold)
                v_a = v_a.at[i].set(v1)

                i2 = call_submodule(Linear, f"block{i}_b", common_params, s1)
                v2, s2 = lif_step(v_b[i], i2, beta, threshold)
                v_b = v_b.at[i].set(v2)

                v3, s = lif_step(v_r[i], s + s2, beta, threshold)
                v_r = v_r.at[i].set(v3)
            return v_a, v_b, v_r, s

        if has_l1_base and common_params.iterinfo is not None:

            def scan_fn(carry, inp):
                stem_v, v_a, v_b, v_r, out_v, acc = carry
                x_t, base_t = inp

                i_stem = base_t + x_t @ l1_B @ l1_A.T
                stem_v, s = lif_step(stem_v, i_stem, beta, threshold)
                v_a, v_b, v_r, s = _scan_blocks(v_a, v_b, v_r, s)

                i_out = call_submodule(Linear, "linear_out", common_params, s)
                out_v, s_out = lif_step(out_v, i_out, beta, threshold)
                acc = acc + (out_v if use_membrane else s_out)
                return (stem_v, v_a, v_b, v_r, out_v, acc), None

            x_t = jnp.transpose(x, (1, 0, 2))
            (_, _, _, _, _, accum), _ = jax.lax.scan(
                scan_fn, (v_stem, v_block_a, v_block_b, v_block_res, v_out, accum), (x_t, l1_base)
            )
        else:

            def scan_fn(carry, x_t):
                stem_v, v_a, v_b, v_r, out_v, acc = carry

                i_stem = call_submodule(Linear, "linear1", common_params, x_t)
                stem_v, s = lif_step(stem_v, i_stem, beta, threshold)
                v_a, v_b, v_r, s = _scan_blocks(v_a, v_b, v_r, s)

                i_out = call_submodule(Linear, "linear_out", common_params, s)
                out_v, s_out = lif_step(out_v, i_out, beta, threshold)
                acc = acc + (out_v if use_membrane else s_out)
                return (stem_v, v_a, v_b, v_r, out_v, acc), None

            x_t = jnp.transpose(x, (1, 0, 2))
            (_, _, _, _, _, accum), _ = jax.lax.scan(
                scan_fn, (v_stem, v_block_a, v_block_b, v_block_res, v_out, accum), x_t
            )

        return accum
