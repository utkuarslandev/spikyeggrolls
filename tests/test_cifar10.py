"""Tests for CIFAR-10 encoding and spiking ResNet wiring."""

import pytest
import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import CommonParams
from hyperscalees.models.common import ConvKernel, EXCLUDED, simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll
from spikyeggroll.data.cifar10 import augment_batch, encode_batch
from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.spiking_resnet import (
    BasicBlock,
    BatchNorm2d,
    BNTTNorm2d,
    Conv2d,
    SpikingResNet18Model,
)
from spikyeggroll.train import compute_fitness
from spikyeggroll.models.snn import SNNModel


def test_cifar_encode_batch_shape_and_binary():
    key = jax.random.key(0)
    images = jax.random.uniform(key, (8, 32, 32, 3))
    spikes = encode_batch(images, timesteps=4, key=key)
    assert spikes.shape == (8, 4, 3, 32, 32)
    unique = jnp.unique(spikes)
    assert jnp.all((unique == 0.0) | (unique == 1.0))


def test_conv_kernel_supports_eval_and_noisy_forward():
    key = jax.random.key(11)
    k1, k2 = jax.random.split(key)
    _, params, scan_map, _ = ConvKernel.rand_init(k1, 3, 8, 3, "float32")
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, sigma=0.01, lr=0.001, rank=2
    )

    clean = ConvKernel.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        None,
        params,
        es_tree_key,
        None,
    )
    noisy = ConvKernel.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        None,
        params,
        es_tree_key,
        (jnp.int32(0), jnp.int32(0)),
    )

    assert clean.shape == (8, 3, 3, 3)
    assert noisy.shape == clean.shape
    assert jnp.all(jnp.isfinite(noisy))


def _forward_population(model_cls, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, x, l1b):
    return jax.vmap(
        lambda i: model_cls.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            i,
            x,
            l1b,
        )
    )


def _score_population(model_cls, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, x, y, l1b):
    forward_pop = _forward_population(
        model_cls, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, x, l1b
    )
    return lambda iterinfo: jax.vmap(compute_fitness, in_axes=(0, None))(forward_pop(iterinfo), y)


def _init_resnet(cfg, key):
    k1, k2 = jax.random.split(key)
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    return frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params


def _updated_running_stat_params(cfg, frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params, x):
    logits, bn_stats = SpikingResNet18Model.forward_train_with_bn_stats(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        x,
    )
    params = SpikingResNet18Model.apply_bn_running_stats(
        params,
        bn_stats,
        cfg.resnet_bntt_momentum if cfg.resnet_norm == "bntt" else cfg.resnet_bn_momentum,
    )
    return logits, params


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("resnet_norm", ["group", "batch", "bntt"])
def test_spiking_resnet_forward_shape(dtype, resnet_norm):
    key = jax.random.key(1)
    k1, k2 = jax.random.split(key)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
        timesteps=4,
        pop_size=8,
        dtype=dtype,
        resnet_norm=resnet_norm,
    )
    frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params = _init_resnet(
        cfg, k1
    )
    x = jax.random.bernoulli(k2, 0.3, (4, 4, 3, 32, 32)).astype(jnp.float32)
    if resnet_norm in {"batch", "bntt"}:
        _, params = _updated_running_stat_params(
            cfg, frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params, x
        )
        out = SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            None,
            x,
            norm_training=False,
        )
    else:
        out = SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            None,
            x,
        )
    assert out.shape == (4, 10)
    assert jnp.all(jnp.isfinite(out))


def test_spiking_resnet_forward_debug_reports_activity():
    key = jax.random.key(2)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
        timesteps=4,
        pop_size=8,
    )
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    x = jax.random.bernoulli(k3, 0.3, (4, 4, 3, 32, 32)).astype(jnp.float32)
    common_params = CommonParams(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
    )
    out, stats = SpikingResNet18Model.forward_debug(common_params, x)
    assert out.shape == (4, 10)
    assert len(stats["stem_spike_rates"]) == 4
    assert len(stats["stage_nonzero_rates"]) == 4
    assert all(len(stage_rates) == 4 for stage_rates in stats["stage_nonzero_rates"])
    assert len(stats["classifier_positive_fraction"]) == 4
    assert jnp.isfinite(stats["output_nonzero_fraction"])
    assert jnp.isfinite(stats["output_class_variance_mean"])


def test_batchnorm2d_init_marks_running_stats_excluded():
    key = jax.random.key(31)
    frozen_params, params, scan_map, es_map = BatchNorm2d.rand_init(
        key, channels=8, dtype="float32"
    )
    assert params["running_mean"].shape == (8,)
    assert params["running_var"].shape == (8,)
    assert scan_map["running_mean"] == ()
    assert scan_map["running_var"] == ()
    assert es_map["running_mean"] == 3
    assert es_map["running_var"] == 3
    assert frozen_params["momentum"] == pytest.approx(0.9)
    assert frozen_params["eps"] == pytest.approx(1e-5)


def test_bnttnorm2d_init_marks_running_stats_excluded():
    key = jax.random.key(32)
    frozen_params, params, scan_map, es_map = BNTTNorm2d.rand_init(
        key, channels=8, timesteps=4, dtype="float32"
    )
    assert params["weight"].shape == (4, 8)
    assert params["running_mean"].shape == (4, 8)
    assert params["running_var"].shape == (4, 8)
    assert scan_map["running_mean"] == ()
    assert scan_map["running_var"] == ()
    assert es_map["running_mean"] == EXCLUDED
    assert es_map["running_var"] == EXCLUDED
    assert frozen_params["momentum"] == pytest.approx(0.9)
    assert frozen_params["eps"] == pytest.approx(1e-5)


def test_projection_shortcuts_exist_on_stride_transitions():
    key = jax.random.key(3)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18", n_classes=10)
    _, params, _, _ = SpikingResNet18Model.rand_init(key, cfg)
    assert "shortcut" not in params["stage0_block0"]
    assert "shortcut" in params["stage1_block0"]
    assert "shortcut" in params["stage2_block0"]
    assert "shortcut" in params["stage3_block0"]


@pytest.mark.parametrize("resnet_norm", ["group", "batch", "bntt"])
def test_eval_forward_is_batch_order_invariant(resnet_norm):
    key = jax.random.key(4)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        timesteps=4,
        pop_size=8,
        resnet_norm=resnet_norm,
    )
    frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params = _init_resnet(
        cfg, k1
    )
    x = jax.random.bernoulli(k3, 0.3, (4, 4, 3, 32, 32)).astype(jnp.float32)
    perm = jnp.array([2, 0, 3, 1])
    if resnet_norm in {"batch", "bntt"}:
        _, params = _updated_running_stat_params(
            cfg, frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params, x
        )
        out = SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            None,
            x,
            norm_training=False,
        )
        perm_out = SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            None,
            x[perm],
            norm_training=False,
        )
    else:
        out = SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            None,
            x,
        )
        perm_out = SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            None,
            x[perm],
        )
    assert jnp.allclose(out, perm_out[jnp.argsort(perm)], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("resnet_norm", ["group", "batch", "bntt"])
def test_eval_forward_is_independent_of_batch_companions(resnet_norm):
    key = jax.random.key(12)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        timesteps=4,
        pop_size=8,
        resnet_norm=resnet_norm,
    )
    frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params = _init_resnet(
        cfg, k1
    )
    samples = jax.random.bernoulli(k3, 0.3, (3, 4, 3, 32, 32)).astype(jnp.float32)
    if resnet_norm in {"batch", "bntt"}:
        _, params = _updated_running_stat_params(
            cfg, frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params, samples
        )

    batch_a = jnp.stack([samples[0], samples[1]], axis=0)
    batch_b = jnp.stack([samples[0], samples[2]], axis=0)
    common_kwargs = {
        "norm_training": False,
    } if resnet_norm in {"batch", "bntt"} else {}
    out_a = SpikingResNet18Model.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        batch_a,
        **common_kwargs,
    )
    out_b = SpikingResNet18Model.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        batch_b,
        **common_kwargs,
    )
    assert jnp.allclose(out_a[0], out_b[0], atol=1e-6, rtol=1e-6)


def test_chunked_population_scoring_matches_python_loop():
    key = jax.random.key(21)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    cfg = SNNConfig(
        dataset="mnist",
        model_name="mlp_snn",
        n_inputs=8,
        hidden_size=4,
        n_classes=2,
        timesteps=3,
        pop_size=6,
    )
    frozen_params, params, scan_map, _ = SNNModel.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    x = jax.random.bernoulli(k3, 0.3, (2, 3, 8)).astype(jnp.float32)
    y = jax.random.randint(k4, (2,), 0, cfg.n_classes)
    iterinfo = (jnp.zeros((cfg.pop_size,), dtype=jnp.int32), jnp.arange(cfg.pop_size))

    score_fn = _score_population(
        SNNModel, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, x, y, None
    )

    expected = []
    chunk_size = 4
    for start in range(0, cfg.pop_size, chunk_size):
        c_iter = (iterinfo[0][start : start + chunk_size], iterinfo[1][start : start + chunk_size])
        expected.append(score_fn(c_iter))
    expected = jnp.concatenate(expected)

    pad = (-cfg.pop_size) % chunk_size
    epoch_ids = jnp.pad(iterinfo[0], (0, pad), mode="edge")
    thread_ids = jnp.pad(iterinfo[1], (0, pad), mode="edge")
    starts = jnp.arange(0, cfg.pop_size + pad, chunk_size, dtype=jnp.int32)

    @jax.jit
    def staged_scores():
        def score_chunk(start):
            idx = jnp.arange(chunk_size, dtype=jnp.int32) + start
            c_iter = (epoch_ids[idx], thread_ids[idx])
            return score_fn(c_iter)

        return jax.lax.map(score_chunk, starts).reshape(-1)[: cfg.pop_size]

    actual = staged_scores()
    assert jnp.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_identity_block_has_three_state_tensors_and_spiking_output():
    key = jax.random.key(99)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18")
    frozen_params, params, scan_map, _ = BasicBlock.rand_init(k1, 64, 64, 1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    fnp, np_ = EggRoll.init_noiser(params, 0.01, 0.001, rank=1)
    cp = CommonParams(EggRoll, fnp, np_, frozen_params, params, es_tree_key, None)

    x = jax.random.uniform(k3, (2, 64, 16, 16))
    # SEW-ResNet style: state is (v1, v2, v_out) — 3 membrane potentials per block
    state = (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x))
    out, new_state, stats = BasicBlock._forward(cp, x, state, collect_stats=True)

    assert "shortcut" not in params
    assert len(new_state) == 3
    assert out.shape == x.shape
    assert "shortcut_nonzero_fraction" in stats
    # Block output is binary spikes (SEW-ResNet style)
    assert jnp.all((out == 0.0) | (out == 1.0))


def test_projection_block_applies_projection_without_shortcut_state():
    key = jax.random.key(100)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18")
    frozen_params, params, scan_map, _ = BasicBlock.rand_init(k1, 64, 128, 2, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    fnp, np_ = EggRoll.init_noiser(params, 0.01, 0.001, rank=1)
    cp = CommonParams(EggRoll, fnp, np_, frozen_params, params, es_tree_key, None)

    x = jax.random.uniform(k3, (2, 64, 16, 16))
    state = (jnp.zeros((2, 128, 8, 8)), jnp.zeros((2, 128, 8, 8)))
    out, new_state, _ = BasicBlock._forward(cp, x, state, collect_stats=True)

    assert "shortcut" in params
    assert len(new_state) == 2
    assert out.shape == (2, 128, 8, 8)


@pytest.mark.parametrize("resnet_norm", ["batch", "bntt"])
def test_running_stats_update_only_from_base_forward(resnet_norm):
    key = jax.random.key(77)
    k1, k2 = jax.random.split(key)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        timesteps=4,
        pop_size=4,
        resnet_norm=resnet_norm,
    )
    frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params = _init_resnet(
        cfg, k1
    )
    x = jax.random.bernoulli(k2, 0.3, (2, 4, 3, 32, 32)).astype(jnp.float32)

    _, bn_stats = SpikingResNet18Model.forward_train_with_bn_stats(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        x,
    )
    updated_params = SpikingResNet18Model.apply_bn_running_stats(
        params,
        bn_stats,
        cfg.resnet_bntt_momentum if resnet_norm == "bntt" else cfg.resnet_bn_momentum,
    )
    before_noisy_mean = updated_params["stem_norm"]["running_mean"]
    before_noisy_var = updated_params["stage0_block0"]["norm1"]["running_var"]

    assert not jnp.allclose(
        params["stem_norm"]["running_mean"],
        updated_params["stem_norm"]["running_mean"],
    )
    assert not jnp.allclose(
        params["stage0_block0"]["norm1"]["running_var"],
        updated_params["stage0_block0"]["norm1"]["running_var"],
    )

    iterinfo = (jnp.zeros(cfg.pop_size, dtype=jnp.int32), jnp.arange(cfg.pop_size))
    _ = jax.vmap(
        lambda info: SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            updated_params,
            es_tree_key,
            info,
            x,
            norm_training=True,
        )
    )(iterinfo)

    assert jnp.allclose(before_noisy_mean, updated_params["stem_norm"]["running_mean"])
    assert jnp.allclose(before_noisy_var, updated_params["stage0_block0"]["norm1"]["running_var"])


def test_selective_plan_schedule_resolves_early_mid_and_full_refresh():
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        num_epochs=10,
        selective_stage_perturbation=True,
        stage_perturbation_early_fraction=0.3,
        stage_perturbation_full_epoch_interval=8,
    )

    early = SpikingResNet18Model.resolve_selective_plan(2, cfg.num_epochs, cfg)
    mid = SpikingResNet18Model.resolve_selective_plan(4, cfg.num_epochs, cfg)
    full = SpikingResNet18Model.resolve_selective_plan(8, cfg.num_epochs, cfg)

    assert early["phase"] == "early_selective"
    assert early["active_groups"] == ("stage3", "head")
    assert early["cache_split"] == "after_stage2"
    assert mid["phase"] == "mid_selective"
    assert mid["active_groups"] == ("stage2", "stage3", "head")
    assert mid["cache_split"] == "after_stage1"
    assert full["phase"] == "full_model_refresh"
    assert full["cache_split"] is None
    assert full["active_groups"] == (
        "stem",
        "stage0",
        "stage1",
        "stage2",
        "stage3",
        "head",
    )


def test_selective_es_map_masks_inactive_groups_and_keeps_bn_stats_excluded():
    key = jax.random.key(101)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        resnet_norm="batch",
    )
    frozen_params, _, _, es_map = SpikingResNet18Model.rand_init(key, cfg)
    perturb_group_map = frozen_params["perturb_group_map"]

    early_map = SpikingResNet18Model.build_active_es_map(
        es_map, perturb_group_map, ("stage3", "head")
    )
    mid_map = SpikingResNet18Model.build_active_es_map(
        es_map, perturb_group_map, ("stage2", "stage3", "head")
    )
    full_map = SpikingResNet18Model.build_active_es_map(
        es_map, perturb_group_map, ("stem", "stage0", "stage1", "stage2", "stage3", "head")
    )

    assert early_map["stem_conv"]["weight"] == EXCLUDED
    assert early_map["stage2_block0"]["conv1"]["weight"] == EXCLUDED
    assert early_map["stage3_block0"]["conv1"]["weight"] != EXCLUDED
    assert early_map["linear_out"]["weight"] != EXCLUDED
    assert early_map["stem_norm"]["running_mean"] == EXCLUDED
    assert early_map["stage3_block0"]["norm1"]["running_var"] == EXCLUDED

    assert mid_map["stage2_block0"]["conv1"]["weight"] != EXCLUDED
    assert mid_map["stage1_block0"]["conv1"]["weight"] == EXCLUDED

    assert full_map["stem_conv"]["weight"] == es_map["stem_conv"]["weight"]
    assert full_map["stage1_block0"]["conv1"]["weight"] == es_map["stage1_block0"]["conv1"]["weight"]


@pytest.mark.parametrize("resnet_norm", ["group", "batch", "bntt"])
def test_prefix_suffix_forward_matches_full_forward_after_stage1_and_stage2(resnet_norm):
    key = jax.random.key(102)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        timesteps=3,
        pop_size=4,
        resnet_norm=resnet_norm,
    )
    frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params = _init_resnet(
        cfg, k1
    )
    x = jax.random.bernoulli(k3, 0.3, (2, 3, 3, 32, 32)).astype(jnp.float32)

    common_kwargs = {}
    if resnet_norm in {"batch", "bntt"}:
        _, params = _updated_running_stat_params(
            cfg, frozen_params, params, es_tree_key, frozen_noiser_params, noiser_params, x
        )
        common_kwargs["norm_training"] = False

    full = SpikingResNet18Model.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        x,
        **common_kwargs,
    )
    prefix_stage1 = SpikingResNet18Model.forward_prefix_after_stage1(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        x,
        norm_training=common_kwargs.get("norm_training", False),
    )
    suffix_stage1 = SpikingResNet18Model.forward_suffix_after_stage1(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        prefix_stage1,
        norm_training=common_kwargs.get("norm_training", False),
    )
    prefix_stage2 = SpikingResNet18Model.forward_prefix_after_stage2(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        x,
        norm_training=common_kwargs.get("norm_training", False),
    )
    suffix_stage2 = SpikingResNet18Model.forward_suffix_after_stage2(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        prefix_stage2,
        norm_training=common_kwargs.get("norm_training", False),
    )

    assert jnp.allclose(full, suffix_stage1, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(full, suffix_stage2, atol=1e-5, rtol=1e-5)


def test_selective_update_changes_only_active_stages():
    key = jax.random.key(103)
    k1, k2 = jax.random.split(key)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        timesteps=3,
        pop_size=4,
        rank=1,
    )
    frozen_params, params, scan_map, es_map = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    active_es_map = SpikingResNet18Model.build_active_es_map(
        es_map, frozen_params["perturb_group_map"], ("stage3", "head")
    )
    iterinfo = (jnp.zeros(cfg.pop_size, dtype=jnp.int32), jnp.arange(cfg.pop_size))
    fitnesses = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)

    _, new_params = EggRoll.do_updates(
        frozen_noiser_params,
        noiser_params,
        params,
        es_tree_key,
        fitnesses,
        iterinfo,
        active_es_map,
    )

    assert jnp.allclose(new_params["stem_conv"]["weight"], params["stem_conv"]["weight"])
    assert jnp.allclose(
        new_params["stage2_block0"]["conv1"]["weight"],
        params["stage2_block0"]["conv1"]["weight"],
    )
    assert not jnp.allclose(
        new_params["stage3_block0"]["conv1"]["weight"],
        params["stage3_block0"]["conv1"]["weight"],
    )
    assert not jnp.allclose(
        new_params["linear_out"]["weight"],
        params["linear_out"]["weight"],
    )


@pytest.mark.parametrize("key_factory", [jax.random.key, jax.random.PRNGKey])
def test_selective_update_supports_typed_and_legacy_keys(key_factory):
    key = key_factory(1043)
    k1, k2 = jax.random.split(key)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        timesteps=3,
        pop_size=4,
        rank=1,
        use_batched_update=True,
    )
    frozen_params, params, scan_map, es_map = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        cfg.sigma,
        cfg.lr,
        rank=cfg.rank,
        use_batched_update=cfg.use_batched_update,
    )
    active_es_map = SpikingResNet18Model.build_active_es_map(
        es_map, frozen_params["perturb_group_map"], ("stage3", "head")
    )
    iterinfo = (jnp.zeros(cfg.pop_size, dtype=jnp.int32), jnp.arange(cfg.pop_size))
    fitnesses = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)

    new_noiser_params, new_params = EggRoll.do_updates(
        frozen_noiser_params,
        noiser_params,
        params,
        es_tree_key,
        fitnesses,
        iterinfo,
        active_es_map,
    )

    assert new_params["linear_out"]["weight"].shape == params["linear_out"]["weight"].shape
    assert float(new_noiser_params["sigma"]) == float(noiser_params["sigma"])


def test_conv2d_matrix_lora_matches_kernel_lora_noisy_forward():
    key = jax.random.key(104)
    k1, k2, k3 = jax.random.split(key, 3)
    base_cfg = dict(
        dataset="cifar10",
        model_name="spiking_resnet18",
        dtype="float32",
        pop_size=4,
        rank=2,
        sigma=0.01,
        lr=0.001,
    )
    cfg_kernel = SNNConfig(**base_cfg, conv_es_mode="kernel_lora")
    cfg_matrix = SNNConfig(**base_cfg, conv_es_mode="matrix_lora")
    frozen_k, params, scan_map, _ = Conv2d.rand_init(
        k1,
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        use_bias=False,
        dtype=cfg_kernel.dtype,
        cfg=cfg_kernel,
    )
    frozen_m, _, _, _ = Conv2d.rand_init(
        k1,
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        use_bias=False,
        dtype=cfg_matrix.dtype,
        cfg=cfg_matrix,
    )
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg_kernel.sigma, cfg_kernel.lr, rank=cfg_kernel.rank
    )
    x = jax.random.normal(k3, (2, 3, 9, 9), dtype=jnp.float32)
    iterinfo = (jnp.int32(0), jnp.int32(0))
    common_k = CommonParams(
        EggRoll, frozen_noiser_params, noiser_params, frozen_k, params, es_tree_key, iterinfo
    )
    common_m = CommonParams(
        EggRoll, frozen_noiser_params, noiser_params, frozen_m, params, es_tree_key, iterinfo
    )

    out_kernel = Conv2d._forward(common_k, x)
    out_matrix = Conv2d._forward(common_m, x)

    assert out_kernel.shape == out_matrix.shape
    assert jnp.allclose(out_kernel, out_matrix, atol=1e-5, rtol=1e-5)


def test_augment_batch_shape_and_range():
    """augment_batch preserves shape and keeps values in [0, 1]."""
    key = jax.random.key(42)
    images = jax.random.uniform(key, (8, 32, 32, 3))
    aug = augment_batch(images, key)
    assert aug.shape == images.shape
    assert float(aug.min()) >= 0.0
    assert float(aug.max()) <= 1.0


def test_population_forward_and_update_runs_for_conv_resnet():
    key = jax.random.key(5)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18", timesteps=3, pop_size=4)
    frozen_params, params, scan_map, es_map = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    x = jax.random.bernoulli(k3, 0.25, (2, 3, 3, 32, 32)).astype(jnp.float32)
    labels = jax.random.randint(k4, (2,), 0, cfg.n_classes)
    iterinfo = (jnp.zeros(cfg.pop_size, dtype=jnp.int32), jnp.arange(cfg.pop_size))

    pop_out = jax.vmap(
        lambda info: SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            info,
            x,
        )
    )(iterinfo)
    raw_scores = jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, labels)
    fitnesses = EggRoll.convert_fitnesses(
        frozen_noiser_params, noiser_params, raw_scores
    )
    new_noiser_params, new_params = EggRoll.do_updates(
        frozen_noiser_params,
        noiser_params,
        params,
        es_tree_key,
        fitnesses,
        iterinfo,
        es_map,
    )

    assert pop_out.shape == (cfg.pop_size, 2, cfg.n_classes)
    assert jnp.all(jnp.isfinite(raw_scores))
    assert new_params["stem_conv"]["weight"].shape == params["stem_conv"]["weight"].shape
    assert float(new_noiser_params["sigma"]) == float(noiser_params["sigma"])
