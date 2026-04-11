"""CIFAR-sized spiking ResNet-18 compatible with the EGGROLL parameter pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import CommonInit, CommonParams, Model
from hyperscalees.models.common import ConvKernel, Linear, Parameter, call_submodule, merge_frozen, merge_inits

from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import lif_step


def _conv_out_dim(size: int, kernel_size: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1


def _conv2d_nchw(x, weight, stride: int, padding: int):
    return jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


class Conv2d(Model):
    @classmethod
    def rand_init(
        cls,
        key,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_bias: bool,
        dtype,
    ):
        k1, k2 = jax.random.split(key)
        init = merge_inits(
            weight=ConvKernel.rand_init(
                k1, in_channels, out_channels, kernel_size, dtype
            )
        )
        if use_bias:
            bias_init = Parameter.rand_init(
                k2, None, None, jnp.zeros((out_channels,), dtype=dtype), dtype
            )
            init = CommonInit(
                init.frozen_params,
                {**init.params, "bias": bias_init.params},
                {**init.scan_map, "bias": bias_init.scan_map},
                {**init.es_map, "bias": bias_init.es_map},
            )
        return merge_frozen(init, stride=stride, padding=padding)

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        stride = int(common_params.frozen_params["stride"])
        padding = int(common_params.frozen_params["padding"])
        weight = call_submodule(ConvKernel, "weight", common_params)
        out = _conv2d_nchw(x, weight, stride=stride, padding=padding)
        if "bias" in common_params.params:
            bias = call_submodule(Parameter, "bias", common_params)
            out = out + bias[None, :, None, None]
        return out


class GroupNorm2d(Model):
    @classmethod
    def rand_init(
        cls, key, channels: int, dtype, num_groups: int = 8, eps: float = 1e-5
    ):
        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_groups={num_groups}"
            )
        k1, k2 = jax.random.split(key)
        init = merge_inits(
            weight=Parameter.rand_init(
                k1, None, None, jnp.ones((channels,), dtype=dtype), dtype
            ),
            bias=Parameter.rand_init(
                k2, None, None, jnp.zeros((channels,), dtype=dtype), dtype
            ),
        )
        return merge_frozen(init, eps=eps, num_groups=num_groups)

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        eps = common_params.frozen_params["eps"]
        num_groups = int(common_params.frozen_params["num_groups"])
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch, num_groups, channels // num_groups, height, width)
        mean = jnp.mean(grouped, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(grouped, axis=(2, 3, 4), keepdims=True)
        normed = (grouped - mean) / jnp.sqrt(var + eps)
        normed = normed.reshape(batch, channels, height, width)
        weight = call_submodule(Parameter, "weight", common_params)[None, :, None, None]
        bias = call_submodule(Parameter, "bias", common_params)[None, :, None, None]
        return normed * weight + bias


class ProjectionShortcut(Model):
    @classmethod
    def rand_init(
        cls,
        key,
        in_channels: int,
        out_channels: int,
        stride: int,
        dtype,
        norm_groups: int,
    ):
        k1, k2 = jax.random.split(key)
        return merge_inits(
            conv=Conv2d.rand_init(
                k1,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                use_bias=False,
                dtype=dtype,
            ),
            norm=GroupNorm2d.rand_init(k2, out_channels, dtype, norm_groups),
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        x = call_submodule(Conv2d, "conv", common_params, x)
        x = call_submodule(GroupNorm2d, "norm", common_params, x)
        return x


class BasicBlock(Model):
    @classmethod
    def rand_init(
        cls,
        key,
        in_channels: int,
        out_channels: int,
        stride: int,
        cfg: SNNConfig,
        stage_threshold: float | None = None,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        init = merge_inits(
            conv1=Conv2d.rand_init(
                k1,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                use_bias=False,
                dtype=cfg.dtype,
            ),
            norm1=GroupNorm2d.rand_init(
                k2, out_channels, cfg.dtype, cfg.resnet_norm_groups
            ),
            conv2=Conv2d.rand_init(
                k3,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bias=False,
                dtype=cfg.dtype,
            ),
            norm2=GroupNorm2d.rand_init(
                k4, out_channels, cfg.dtype, cfg.resnet_norm_groups
            ),
        )

        if stride != 1 or in_channels != out_channels:
            shortcut_init = ProjectionShortcut.rand_init(
                k5,
                in_channels,
                out_channels,
                stride,
                cfg.dtype,
                cfg.resnet_norm_groups,
            )
            init = CommonInit(
                {
                    **(init.frozen_params or {}),
                    "shortcut": shortcut_init.frozen_params,
                },
                {**init.params, "shortcut": shortcut_init.params},
                {**init.scan_map, "shortcut": shortcut_init.scan_map},
                {**init.es_map, "shortcut": shortcut_init.es_map},
            )

        effective_threshold = stage_threshold if stage_threshold is not None else cfg.threshold
        init = merge_frozen(init, beta=cfg.beta, threshold=effective_threshold)
        init.params["norm2"]["weight"] = jnp.zeros_like(init.params["norm2"]["weight"])
        return init

    @classmethod
    def _forward(cls, common_params: CommonParams, x, state, collect_stats: bool = False):
        beta = common_params.frozen_params["beta"]
        threshold = common_params.frozen_params["threshold"]
        v1, v2, v_sc = state  # v_sc: shortcut membrane (SEW pattern)

        # Main branch: conv1 → norm1 → LIF1
        out = call_submodule(Conv2d, "conv1", common_params, x)
        out = call_submodule(GroupNorm2d, "norm1", common_params, out)
        v1, s1 = lif_step(v1, out, beta, threshold)

        # Main branch: conv2 → norm2 → LIF2
        out = call_submodule(Conv2d, "conv2", common_params, s1)
        out = call_submodule(GroupNorm2d, "norm2", common_params, out)
        v2, s2 = lif_step(v2, out, beta, threshold)

        # Shortcut branch: optional projection, then LIF (SEW-ResNet pattern)
        # Both branches are binary before addition: out ∈ {0, 1, 2}
        shortcut_pre = x
        if "shortcut" in common_params.params:
            shortcut_pre = call_submodule(ProjectionShortcut, "shortcut", common_params, x)
        v_sc, s_sc = lif_step(v_sc, shortcut_pre, beta, threshold)

        out = s_sc + s2

        if collect_stats:
            stats = {
                "conv1_rate": jnp.mean(s1),
                "conv2_rate": jnp.mean(s2),
                "shortcut_rate": jnp.mean(s_sc),
                "block_output_nonzero_fraction": jnp.mean(out != 0),
            }
            return out, (v1, v2, v_sc), stats

        return out, (v1, v2, v_sc)


class SpikingResNet18Model(Model):
    """Convolutional CIFAR-sized spiking ResNet-18."""

    STAGE_BLOCKS = (2, 2, 2, 2)

    @classmethod
    def rand_init(cls, key, cfg: SNNConfig):
        if cfg.dataset == "cifar10" and (
            cfg.in_channels != 3 or cfg.image_size != 32 or cfg.n_inputs != 3072
        ):
            raise ValueError(
                "spiking_resnet18 expects CIFAR-10 config defaults: "
                "in_channels=3, image_size=32, n_inputs=3072."
            )
        if cfg.resnet_norm != "group":
            raise ValueError("Only resnet_norm='group' is supported for spiking_resnet18.")
        if len(cfg.resnet_block_counts) != len(cls.STAGE_BLOCKS):
            raise ValueError(
                f"spiking_resnet18 requires exactly {len(cls.STAGE_BLOCKS)} stages, "
                f"got resnet_block_counts={cfg.resnet_block_counts}."
            )

        dtype = cfg.dtype
        base_channels = int(cfg.resnet_channels_base)
        stage_channels = tuple(base_channels * (2**i) for i in range(4))
        stage_blocks = tuple(cfg.resnet_block_counts)

        n_block_keys = sum(stage_blocks)
        keys = jax.random.split(key, 3 + n_block_keys)

        all_inits = {
            "stem_conv": Conv2d.rand_init(
                keys[0],
                cfg.in_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bias=False,
                dtype=dtype,
            ),
            "stem_norm": GroupNorm2d.rand_init(
                keys[1], base_channels, dtype, cfg.resnet_norm_groups
            ),
            "linear_out": Linear.rand_init(
                keys[2], stage_channels[-1], cfg.n_classes, True, dtype
            ),
        }

        key_idx = 3
        in_channels = base_channels
        for stage_idx, (out_channels, block_count) in enumerate(zip(stage_channels, stage_blocks)):
            stage_threshold = cfg.threshold * (2 ** stage_idx) if cfg.resnet_threshold_scale else None
            for block_idx in range(block_count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                name = f"stage{stage_idx}_block{block_idx}"
                all_inits[name] = BasicBlock.rand_init(
                    keys[key_idx], in_channels, out_channels, stride, cfg,
                    stage_threshold=stage_threshold,
                )
                in_channels = out_channels
                key_idx += 1

        init = merge_inits(**all_inits)
        init = merge_frozen(
            init,
            beta=cfg.beta,
            threshold=cfg.threshold,
            timesteps=cfg.timesteps,
            membrane_readout=cfg.membrane_readout,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            resnet_channels_base=base_channels,
            resnet_norm=cfg.resnet_norm,
            resnet_norm_groups=cfg.resnet_norm_groups,
            resnet_threshold_scale=cfg.resnet_threshold_scale,
        )
        return init

    @classmethod
    def _initial_block_states(cls, batch_size: int, image_size: int, stage_channels, stage_blocks):
        states = []
        spatial = image_size
        for stage_idx, (out_channels, block_count) in enumerate(zip(stage_channels, stage_blocks)):
            for block_idx in range(block_count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                spatial = _conv_out_dim(spatial, 3, stride, 1)
                shape = (batch_size, out_channels, spatial, spatial)
                states.append((jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape)))
        return tuple(states)

    @classmethod
    def _scan_step(cls, common_params: CommonParams, carry, x_t):
        beta = common_params.frozen_params["beta"]
        threshold = common_params.frozen_params["threshold"]
        use_membrane = common_params.frozen_params["membrane_readout"]
        stage_blocks = common_params.frozen_params["stage_blocks"]

        stem_v, block_states, classifier_v, acc = carry

        x = call_submodule(Conv2d, "stem_conv", common_params, x_t)
        x = call_submodule(GroupNorm2d, "stem_norm", common_params, x)
        stem_v, x = lif_step(stem_v, x, beta, threshold)

        state_idx = 0
        for stage_idx, block_count in enumerate(stage_blocks):
            for block_idx in range(block_count):
                block_name = f"stage{stage_idx}_block{block_idx}"
                x, new_state = call_submodule(
                    BasicBlock,
                    block_name,
                    common_params,
                    x,
                    block_states[state_idx],
                )
                block_states = block_states[:state_idx] + (new_state,) + block_states[state_idx + 1 :]
                state_idx += 1

        pooled = jnp.mean(x, axis=(2, 3))
        logits = call_submodule(Linear, "linear_out", common_params, pooled)
        classifier_v = beta * classifier_v + logits
        acc = acc + (classifier_v if use_membrane else logits)
        return (stem_v, block_states, classifier_v, acc), None

    @classmethod
    def _forward(cls, common_params: CommonParams, x, l1_base=None):
        del l1_base
        if x.ndim != 5:
            raise ValueError(
                f"spiking_resnet18 expects input shape [B, T, C, H, W], got {x.shape}."
            )

        batch_size, timesteps, _, image_size, _ = x.shape
        stage_channels = common_params.frozen_params["stage_channels"]
        stage_blocks = common_params.frozen_params["stage_blocks"]
        stem_v = jnp.zeros((batch_size, stage_channels[0], image_size, image_size))
        block_states = cls._initial_block_states(
            batch_size, image_size, stage_channels, stage_blocks
        )
        classifier_v = jnp.zeros(
            (batch_size, common_params.params["linear_out"]["weight"].shape[0])
        )
        acc = jnp.zeros_like(classifier_v)
        x_t = jnp.transpose(x, (1, 0, 2, 3, 4))
        (_, _, _, acc), _ = jax.lax.scan(
            lambda carry, step_x: cls._scan_step(common_params, carry, step_x),
            (stem_v, block_states, classifier_v, acc),
            x_t,
        )
        return acc / timesteps

    @classmethod
    def forward_debug(cls, common_params: CommonParams, x):
        if x.ndim != 5:
            raise ValueError(
                f"spiking_resnet18 expects input shape [B, T, C, H, W], got {x.shape}."
            )

        beta = common_params.frozen_params["beta"]
        threshold = common_params.frozen_params["threshold"]
        use_membrane = common_params.frozen_params["membrane_readout"]
        stage_channels = common_params.frozen_params["stage_channels"]
        stage_blocks = common_params.frozen_params["stage_blocks"]

        batch_size, timesteps, _, image_size, _ = x.shape
        stem_v = jnp.zeros((batch_size, stage_channels[0], image_size, image_size))
        block_states = list(
            cls._initial_block_states(batch_size, image_size, stage_channels, stage_blocks)
        )
        classifier_v = jnp.zeros(
            (batch_size, common_params.params["linear_out"]["weight"].shape[0])
        )
        acc = jnp.zeros_like(classifier_v)

        stem_rates = []
        stage_rates = [[] for _ in range(4)]
        block_conv1_rates = []
        block_conv2_rates = []
        classifier_positive_fraction = []
        classifier_mean = []

        x_t = jnp.transpose(x, (1, 0, 2, 3, 4))
        for x_step in x_t:
            x_step = call_submodule(Conv2d, "stem_conv", common_params, x_step)
            x_step = call_submodule(GroupNorm2d, "stem_norm", common_params, x_step)
            stem_v, x_step = lif_step(stem_v, x_step, beta, threshold)
            stem_rates.append(jnp.mean(x_step))

            state_idx = 0
            for stage_idx, block_count in enumerate(stage_blocks):
                for block_idx in range(block_count):
                    block_name = f"stage{stage_idx}_block{block_idx}"
                    x_step, new_state, stats = call_submodule(
                        BasicBlock,
                        block_name,
                        common_params,
                        x_step,
                        block_states[state_idx],
                        True,
                    )
                    block_states[state_idx] = new_state
                    block_conv1_rates.append(float(stats["conv1_rate"]))
                    block_conv2_rates.append(float(stats["conv2_rate"]))
                    state_idx += 1
                stage_rates[stage_idx].append(jnp.mean(x_step != 0))

            pooled = jnp.mean(x_step, axis=(2, 3))
            logits = call_submodule(Linear, "linear_out", common_params, pooled)
            classifier_v = beta * classifier_v + logits
            readout = classifier_v if use_membrane else logits
            acc = acc + readout
            classifier_positive_fraction.append(jnp.mean(readout > 0))
            classifier_mean.append(jnp.mean(readout))

        acc = acc / timesteps
        stats = {
            "stem_spike_rates": [float(v) for v in stem_rates],
            "stage_nonzero_rates": [[float(v) for v in values] for values in stage_rates],
            "block_conv1_spike_rates": block_conv1_rates,
            "block_conv2_spike_rates": block_conv2_rates,
            "classifier_positive_fraction": [float(v) for v in classifier_positive_fraction],
            "classifier_mean": [float(v) for v in classifier_mean],
            "output_nonzero_fraction": float(jnp.mean(acc != 0)),
            "output_mean": float(jnp.mean(acc)),
            "output_max": float(jnp.max(acc)),
            "output_class_variance_mean": float(jnp.mean(jnp.var(acc, axis=-1))),
            "sample_output_row_sums": [float(v) for v in jnp.sum(acc, axis=-1)[:8]],
        }
        return acc, stats
