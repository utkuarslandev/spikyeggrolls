"""CIFAR-sized spiking ResNet-18 compatible with the EGGROLL parameter pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import CommonInit, CommonParams, Model
from hyperscalees.models.common import (
    EXCLUDED,
    ConvKernel,
    Linear,
    Parameter,
    call_submodule,
    merge_frozen,
    merge_inits,
)

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


def _excluded_array(value):
    return CommonInit(None, value, (), EXCLUDED)


def _merge_bn_stats(existing, incoming):
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    merged = {}
    for key in existing.keys() | incoming.keys():
        if key in existing and key in incoming:
            if isinstance(existing[key], dict) and isinstance(incoming[key], dict):
                merged[key] = _merge_bn_stats(existing[key], incoming[key])
            else:
                merged[key] = incoming[key]
        elif key in existing:
            merged[key] = existing[key]
        else:
            merged[key] = incoming[key]
    return merged


def _mean_bn_stats(stats_tree):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), stats_tree)


def _update_running_stats(params, stats_tree, momentum):
    if stats_tree is None:
        return params

    if isinstance(stats_tree, dict) and "mean" in stats_tree and "var" in stats_tree:
        return {
            **params,
            "running_mean": momentum * params["running_mean"]
            + (1.0 - momentum) * stats_tree["mean"].astype(params["running_mean"].dtype),
            "running_var": momentum * params["running_var"]
            + (1.0 - momentum) * stats_tree["var"].astype(params["running_var"].dtype),
        }

    updated = dict(params)
    for key, value in stats_tree.items():
        updated[key] = _update_running_stats(params[key], value, momentum)
    return updated


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


class BatchNorm2d(Model):
    @classmethod
    def rand_init(
        cls, key, channels: int, dtype, momentum: float = 0.9, eps: float = 1e-5
    ):
        k1, k2 = jax.random.split(key)
        weight = Parameter.rand_init(
            k1, None, None, jnp.ones((channels,), dtype=dtype), dtype
        )
        bias = Parameter.rand_init(
            k2, None, None, jnp.zeros((channels,), dtype=dtype), dtype
        )
        running_mean = jnp.zeros((channels,), dtype=dtype)
        running_var = jnp.ones((channels,), dtype=dtype)
        return CommonInit(
            {"momentum": momentum, "eps": eps},
            {
                "weight": weight.params,
                "bias": bias.params,
                "running_mean": running_mean,
                "running_var": running_var,
            },
            {
                "weight": weight.scan_map,
                "bias": bias.scan_map,
                "running_mean": (),
                "running_var": (),
            },
            {
                "weight": weight.es_map,
                "bias": bias.es_map,
                "running_mean": EXCLUDED,
                "running_var": EXCLUDED,
            },
        )

    @classmethod
    def _forward(
        cls,
        common_params: CommonParams,
        x,
        *,
        norm_training: bool = False,
        collect_bn_stats: bool = False,
    ):
        eps = common_params.frozen_params["eps"]
        weight = call_submodule(Parameter, "weight", common_params)[None, :, None, None]
        bias = call_submodule(Parameter, "bias", common_params)[None, :, None, None]
        if norm_training:
            mean = jnp.mean(x, axis=(0, 2, 3))
            var = jnp.var(x, axis=(0, 2, 3))
        else:
            mean = common_params.params["running_mean"]
            var = common_params.params["running_var"]
        normed = (x - mean[None, :, None, None]) / jnp.sqrt(var[None, :, None, None] + eps)
        out = normed * weight + bias
        if collect_bn_stats:
            return out, {"mean": mean, "var": var}
        return out


def _norm_rand_init(key, channels: int, cfg: SNNConfig):
    if cfg.resnet_norm == "group":
        return GroupNorm2d.rand_init(key, channels, cfg.dtype, cfg.resnet_norm_groups)
    if cfg.resnet_norm == "batch":
        return BatchNorm2d.rand_init(
            key,
            channels,
            cfg.dtype,
            momentum=cfg.resnet_bn_momentum,
            eps=cfg.resnet_bn_eps,
        )
    raise ValueError(f"Unsupported resnet_norm='{cfg.resnet_norm}'.")


def _apply_norm(
    common_params: CommonParams,
    name: str,
    x,
    *,
    norm_training: bool,
    collect_bn_stats: bool,
):
    norm_kind = common_params.frozen_params["resnet_norm"]
    if norm_kind == "group":
        return call_submodule(GroupNorm2d, name, common_params, x), None
    if collect_bn_stats:
        return call_submodule(
            BatchNorm2d,
            name,
            common_params,
            x,
            norm_training=norm_training,
            collect_bn_stats=True,
        )
    return call_submodule(
        BatchNorm2d,
        name,
        common_params,
        x,
        norm_training=norm_training,
    ), None


class ProjectionShortcut(Model):
    @classmethod
    def rand_init(
        cls,
        key,
        in_channels: int,
        out_channels: int,
        stride: int,
        cfg: SNNConfig,
    ):
        k1, k2 = jax.random.split(key)
        init = merge_inits(
            conv=Conv2d.rand_init(
                k1,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                use_bias=False,
                dtype=cfg.dtype,
            ),
            norm=_norm_rand_init(k2, out_channels, cfg),
        )
        return merge_frozen(init, resnet_norm=cfg.resnet_norm)

    @classmethod
    def _forward(
        cls,
        common_params: CommonParams,
        x,
        *,
        norm_training: bool = False,
        collect_bn_stats: bool = False,
    ):
        x = call_submodule(Conv2d, "conv", common_params, x)
        x, norm_stats = _apply_norm(
            common_params,
            "norm",
            x,
            norm_training=norm_training,
            collect_bn_stats=collect_bn_stats,
        )
        if collect_bn_stats:
            return x, {"norm": norm_stats}
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
            norm1=_norm_rand_init(k2, out_channels, cfg),
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
            norm2=_norm_rand_init(k4, out_channels, cfg),
        )

        if stride != 1 or in_channels != out_channels:
            shortcut_init = ProjectionShortcut.rand_init(
                k5,
                in_channels,
                out_channels,
                stride,
                cfg,
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
        init = merge_frozen(
            init,
            beta=cfg.beta,
            threshold=effective_threshold,
            resnet_norm=cfg.resnet_norm,
        )
        init.params["norm2"]["weight"] = jnp.zeros_like(init.params["norm2"]["weight"])
        return init

    @classmethod
    def _forward(
        cls,
        common_params: CommonParams,
        x,
        state,
        collect_stats: bool = False,
        norm_training: bool = False,
        collect_bn_stats: bool = False,
    ):
        beta = common_params.frozen_params["beta"]
        threshold = common_params.frozen_params["threshold"]
        v1, v2 = state

        out = call_submodule(Conv2d, "conv1", common_params, x)
        out, norm1_stats = _apply_norm(
            common_params,
            "norm1",
            out,
            norm_training=norm_training,
            collect_bn_stats=collect_bn_stats,
        )
        v1, s1 = lif_step(v1, out, beta, threshold)

        out = call_submodule(Conv2d, "conv2", common_params, s1)
        out, norm2_stats = _apply_norm(
            common_params,
            "norm2",
            out,
            norm_training=norm_training,
            collect_bn_stats=collect_bn_stats,
        )
        v2, s2 = lif_step(v2, out, beta, threshold)

        shortcut = x
        shortcut_stats = None
        if "shortcut" in common_params.params:
            shortcut_result = call_submodule(
                ProjectionShortcut,
                "shortcut",
                common_params,
                x,
                norm_training=norm_training,
                collect_bn_stats=collect_bn_stats,
            )
            if collect_bn_stats:
                shortcut, shortcut_stats = shortcut_result
            else:
                shortcut = shortcut_result

        out = shortcut + s2
        bn_stats = None
        if collect_bn_stats:
            bn_stats = {"norm1": norm1_stats, "norm2": norm2_stats}
            if shortcut_stats is not None:
                bn_stats["shortcut"] = shortcut_stats

        if collect_stats and collect_bn_stats:
            stats = {
                "conv1_rate": jnp.mean(s1),
                "conv2_rate": jnp.mean(s2),
                "shortcut_nonzero_fraction": jnp.mean(shortcut != 0),
                "block_output_nonzero_fraction": jnp.mean(out != 0),
            }
            return out, (v1, v2), stats, bn_stats
        if collect_stats:
            stats = {
                "conv1_rate": jnp.mean(s1),
                "conv2_rate": jnp.mean(s2),
                "shortcut_nonzero_fraction": jnp.mean(shortcut != 0),
                "block_output_nonzero_fraction": jnp.mean(out != 0),
            }
            return out, (v1, v2), stats
        if collect_bn_stats:
            return out, (v1, v2), bn_stats
        return out, (v1, v2)


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
        if cfg.resnet_norm not in {"group", "batch"}:
            raise ValueError(
                "spiking_resnet18 supports resnet_norm in {'group', 'batch'}."
            )
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
            "stem_norm": _norm_rand_init(keys[1], base_channels, cfg),
            "linear_out": Linear.rand_init(
                keys[2], stage_channels[-1], cfg.n_classes, True, dtype
            ),
        }

        key_idx = 3
        in_channels = base_channels
        for stage_idx, (out_channels, block_count) in enumerate(
            zip(stage_channels, stage_blocks)
        ):
            stage_threshold = (
                cfg.threshold * (2**stage_idx) if cfg.resnet_threshold_scale else None
            )
            for block_idx in range(block_count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                name = f"stage{stage_idx}_block{block_idx}"
                all_inits[name] = BasicBlock.rand_init(
                    keys[key_idx],
                    in_channels,
                    out_channels,
                    stride,
                    cfg,
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
            resnet_bn_momentum=cfg.resnet_bn_momentum,
            resnet_bn_eps=cfg.resnet_bn_eps,
            resnet_threshold_scale=cfg.resnet_threshold_scale,
        )
        return init

    @classmethod
    def _initial_block_states(
        cls, batch_size: int, image_size: int, stage_channels, stage_blocks, dtype=jnp.float32
    ):
        states = []
        spatial = image_size
        for stage_idx, (out_channels, block_count) in enumerate(
            zip(stage_channels, stage_blocks)
        ):
            for block_idx in range(block_count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                spatial = _conv_out_dim(spatial, 3, stride, 1)
                shape = (batch_size, out_channels, spatial, spatial)
                states.append(
                    (jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype))
                )
        return tuple(states)

    @classmethod
    def _scan_step(
        cls,
        common_params: CommonParams,
        carry,
        x_t,
        *,
        norm_training: bool,
        collect_bn_stats: bool,
    ):
        beta = common_params.frozen_params["beta"]
        threshold = common_params.frozen_params["threshold"]
        use_membrane = common_params.frozen_params["membrane_readout"]
        stage_blocks = common_params.frozen_params["stage_blocks"]

        stem_v, block_states, classifier_v, acc = carry

        x = call_submodule(Conv2d, "stem_conv", common_params, x_t)
        x, stem_bn_stats = _apply_norm(
            common_params,
            "stem_norm",
            x,
            norm_training=norm_training,
            collect_bn_stats=collect_bn_stats,
        )
        stem_v, x = lif_step(stem_v, x, beta, threshold)

        state_idx = 0
        block_stats = {}
        for stage_idx, block_count in enumerate(stage_blocks):
            for block_idx in range(block_count):
                block_name = f"stage{stage_idx}_block{block_idx}"
                block_result = call_submodule(
                    BasicBlock,
                    block_name,
                    common_params,
                    x,
                    block_states[state_idx],
                    norm_training=norm_training,
                    collect_bn_stats=collect_bn_stats,
                )
                if collect_bn_stats:
                    x, new_state, bn_stats = block_result
                    block_stats[block_name] = bn_stats
                else:
                    x, new_state = block_result
                block_states = (
                    block_states[:state_idx]
                    + (new_state,)
                    + block_states[state_idx + 1 :]
                )
                state_idx += 1

        pooled = jnp.mean(x, axis=(2, 3))
        logits = call_submodule(Linear, "linear_out", common_params, pooled)
        classifier_v = beta * classifier_v + logits
        acc = acc + (classifier_v if use_membrane else logits)
        if collect_bn_stats:
            step_stats = {"stem_norm": stem_bn_stats, **block_stats}
            return (stem_v, block_states, classifier_v, acc), step_stats
        return (stem_v, block_states, classifier_v, acc), None

    @classmethod
    def _forward(
        cls,
        common_params: CommonParams,
        x,
        l1_base=None,
        *,
        norm_training: bool = False,
        collect_bn_stats: bool = False,
    ):
        del l1_base
        if x.ndim != 5:
            raise ValueError(
                f"spiking_resnet18 expects input shape [B, T, C, H, W], got {x.shape}."
            )

        model_dtype = common_params.params["stem_conv"]["weight"].dtype
        batch_size, timesteps, _, image_size, _ = x.shape
        stage_channels = common_params.frozen_params["stage_channels"]
        stage_blocks = common_params.frozen_params["stage_blocks"]
        stem_v = jnp.zeros(
            (batch_size, stage_channels[0], image_size, image_size), dtype=model_dtype
        )
        block_states = cls._initial_block_states(
            batch_size, image_size, stage_channels, stage_blocks, dtype=model_dtype
        )
        classifier_v = jnp.zeros(
            (batch_size, common_params.params["linear_out"]["weight"].shape[0]),
            dtype=model_dtype,
        )
        acc = jnp.zeros_like(classifier_v)
        x_t = jnp.transpose(x.astype(model_dtype), (1, 0, 2, 3, 4))
        (carry, step_stats) = jax.lax.scan(
            lambda current_carry, step_x: cls._scan_step(
                common_params,
                current_carry,
                step_x,
                norm_training=norm_training,
                collect_bn_stats=collect_bn_stats,
            ),
            (stem_v, block_states, classifier_v, acc),
            x_t,
        )
        acc = carry[3] / timesteps
        if collect_bn_stats:
            return acc, _mean_bn_stats(step_stats)
        return acc

    @classmethod
    def forward_train_with_bn_stats(
        cls,
        noiser,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        iterinfo,
        x,
    ):
        return cls._forward(
            CommonParams(
                noiser,
                frozen_noiser_params,
                noiser_params,
                frozen_params,
                params,
                es_tree_key,
                iterinfo,
            ),
            x,
            norm_training=True,
            collect_bn_stats=True,
        )

    @classmethod
    def apply_bn_running_stats(cls, params, bn_stats, momentum: float):
        return _update_running_stats(params, bn_stats, momentum)

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
        model_dtype = common_params.params["stem_conv"]["weight"].dtype
        norm_training = False

        batch_size, timesteps, _, image_size, _ = x.shape
        stem_v = jnp.zeros(
            (batch_size, stage_channels[0], image_size, image_size), dtype=model_dtype
        )
        block_states = list(
            cls._initial_block_states(
                batch_size, image_size, stage_channels, stage_blocks, dtype=model_dtype
            )
        )
        classifier_v = jnp.zeros(
            (batch_size, common_params.params["linear_out"]["weight"].shape[0]),
            dtype=model_dtype,
        )
        acc = jnp.zeros_like(classifier_v)

        stem_rates = []
        stage_rates = [[] for _ in range(4)]
        block_conv1_rates = []
        block_conv2_rates = []
        block_shortcut_nonzero = []
        classifier_positive_fraction = []
        classifier_mean = []

        x_t = jnp.transpose(x.astype(model_dtype), (1, 0, 2, 3, 4))
        for x_step in x_t:
            x_step = call_submodule(Conv2d, "stem_conv", common_params, x_step)
            x_step, _ = _apply_norm(
                common_params,
                "stem_norm",
                x_step,
                norm_training=norm_training,
                collect_bn_stats=False,
            )
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
                        norm_training=norm_training,
                        collect_bn_stats=False,
                    )
                    block_states[state_idx] = new_state
                    block_conv1_rates.append(float(stats["conv1_rate"]))
                    block_conv2_rates.append(float(stats["conv2_rate"]))
                    block_shortcut_nonzero.append(
                        float(stats["shortcut_nonzero_fraction"])
                    )
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
            "block_shortcut_nonzero_fraction": block_shortcut_nonzero,
            "classifier_positive_fraction": [
                float(v) for v in classifier_positive_fraction
            ],
            "classifier_mean": [float(v) for v in classifier_mean],
            "output_nonzero_fraction": float(jnp.mean(acc != 0)),
            "output_mean": float(jnp.mean(acc)),
            "output_max": float(jnp.max(acc)),
            "output_class_variance_mean": float(jnp.mean(jnp.var(acc, axis=-1))),
            "sample_output_row_sums": [float(v) for v in jnp.sum(acc, axis=-1)[:8]],
        }
        return acc, stats
