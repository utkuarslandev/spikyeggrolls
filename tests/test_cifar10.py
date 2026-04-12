"""Tests for CIFAR-10 encoding and spiking ResNet wiring."""

import pytest
import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import CommonParams
from hyperscalees.models.common import ConvKernel, simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll
from spikyeggroll.data.cifar10 import augment_batch, encode_batch
from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.spiking_resnet import BasicBlock, SpikingResNet18Model
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


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_spiking_resnet_forward_shape(dtype):
    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
        timesteps=4,
        pop_size=8,
        dtype=dtype,
    )
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    x = jax.random.bernoulli(k3, 0.3, (4, 4, 3, 32, 32)).astype(jnp.float32)
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


def test_projection_shortcuts_exist_on_stride_transitions():
    key = jax.random.key(3)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18", n_classes=10)
    _, params, _, _ = SpikingResNet18Model.rand_init(key, cfg)
    assert "shortcut" not in params["stage0_block0"]
    assert "shortcut" in params["stage1_block0"]
    assert "shortcut" in params["stage2_block0"]
    assert "shortcut" in params["stage3_block0"]


def test_eval_forward_is_batch_order_invariant():
    key = jax.random.key(4)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18", timesteps=4, pop_size=8)
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    x = jax.random.bernoulli(k3, 0.3, (4, 4, 3, 32, 32)).astype(jnp.float32)
    perm = jnp.array([2, 0, 3, 1])

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


def test_eval_forward_is_independent_of_batch_companions():
    key = jax.random.key(12)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18", timesteps=4, pop_size=8)
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    samples = jax.random.bernoulli(k3, 0.3, (3, 4, 3, 32, 32)).astype(jnp.float32)

    batch_a = jnp.stack([samples[0], samples[1]], axis=0)
    batch_b = jnp.stack([samples[0], samples[2]], axis=0)
    out_a = SpikingResNet18Model.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        batch_a,
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


def test_sew_block_output_is_integer_valued():
    """SEW pattern: both branches binary before add, so output ∈ {0, 1, 2}."""
    key = jax.random.key(99)
    k1, k2 = jax.random.split(key)
    cfg = SNNConfig(dataset="cifar10", model_name="spiking_resnet18")
    block_init = BasicBlock.rand_init(k1, 64, 128, 2, cfg)
    frozen_params, params, scan_map, _ = block_init
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    fnp, np_ = EggRoll.init_noiser(params, 0.01, 0.001, rank=1)
    cp = CommonParams(EggRoll, fnp, np_, frozen_params, params, es_tree_key, None)

    x = jax.random.uniform(jax.random.key(1), (2, 64, 16, 16))
    # SEW: 3-tuple state (v1, v2, v_sc)
    state = (jnp.zeros((2, 128, 8, 8)), jnp.zeros((2, 128, 8, 8)), jnp.zeros((2, 128, 8, 8)))
    out, new_state, stats = BasicBlock._forward(cp, x, state, collect_stats=True)

    assert len(new_state) == 3                         # (v1, v2, v_sc)
    assert jnp.all(out >= 0)
    assert jnp.all(out == jnp.floor(out))              # integer-valued
    assert float(out.max()) <= 2.0                     # SEW max: 1+1=2
    assert "shortcut_rate" in stats


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
