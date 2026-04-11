"""Tests for SNNModel initialization and forward pass."""

import pytest
import jax
import jax.numpy as jnp

from spikyeggroll.configs import SNNConfig
from spikyeggroll.eval import evaluate
from spikyeggroll.train import build_config_from_args, build_parser
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll


def test_rand_init_structure():
    """rand_init should return a valid CommonInit with expected keys."""
    key = jax.random.key(42)
    cfg = SNNConfig(hidden_size=32, n_inputs=16, n_classes=5)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(key, cfg)

    assert "linear1" in params
    assert "linear2" in params
    assert "linear_out" in params
    assert frozen_params["beta"] == cfg.beta
    assert frozen_params["threshold"] == cfg.threshold


def test_rand_init_shapes():
    """Weight matrices should have correct shapes."""
    key = jax.random.key(42)
    cfg = SNNConfig(hidden_size=32, n_inputs=16, n_classes=5)
    _, params, _, _ = SNNModel.rand_init(key, cfg)

    # MM stores weights as (out_dim, in_dim)
    assert params["linear1"]["weight"].shape == (32, 16)
    assert params["linear2"]["weight"].shape == (32, 32)
    assert params["linear_out"]["weight"].shape == (5, 32)


def test_spiking_resnet_init_structure():
    """SpikingResNet18 init should expose conv stem/output and residual stage params."""
    key = jax.random.key(7)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
    )
    frozen_params, params, _, _ = SpikingResNet18Model.rand_init(key, cfg)
    assert "stem_conv" in params
    assert "stem_norm" in params
    assert "linear_out" in params
    assert "stage0_block0" in params
    assert "stage3_block1" in params
    assert params["stem_conv"]["weight"].shape == (64, 3, 3, 3)
    assert params["linear_out"]["weight"].shape == (10, 512)
    assert tuple(frozen_params["stage_blocks"]) == (2, 2, 2, 2)


@pytest.mark.parametrize(
    ("cfg", "model_cls", "input_shape"),
    [
        (
            SNNConfig(
                dataset="mnist",
                model_name="mlp_snn",
                n_inputs=784,
                hidden_size=32,
                n_classes=10,
                timesteps=5,
                pop_size=8,
            ),
            SNNModel,
            (4, 5, 784),
        ),
        (
            SNNConfig(
                dataset="cifar10",
                model_name="spiking_resnet18",
                n_inputs=3072,
                n_classes=10,
                timesteps=4,
                pop_size=8,
            ),
            SpikingResNet18Model,
            (4, 4, 3, 32, 32),
        ),
    ],
)
def test_evaluate_supports_all_models(cfg, model_cls, input_shape):
    key = jax.random.key(123)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    frozen_params, params, scan_map, _ = model_cls.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )

    test_data = jax.random.bernoulli(k3, 0.25, input_shape).astype(jnp.float32)
    test_labels = jax.random.randint(k4, (input_shape[0],), 0, cfg.n_classes)

    acc = evaluate(
        cfg,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        test_data,
        test_labels,
    )

    assert 0.0 <= acc <= 1.0


def test_cli_defaults_match_snnconfig():
    parser = build_parser()
    args = parser.parse_args([])

    cfg = build_config_from_args(args)
    expected = SNNConfig()

    assert cfg == expected


def test_cli_explicit_override_only_changes_target_field():
    parser = build_parser()
    args = parser.parse_args(["--pop_size", "256"])

    cfg = build_config_from_args(args)
    expected = SNNConfig(pop_size=256)

    assert cfg == expected


def test_cli_dataset_defaults_switch_with_cifar10():
    parser = build_parser()
    args = parser.parse_args(["--dataset", "cifar10"])

    cfg = build_config_from_args(args)

    assert cfg.dataset == "cifar10"
    assert cfg.n_inputs == 3072
    assert cfg.in_channels == 3
    assert cfg.image_size == 32
    assert cfg.pop_size == SNNConfig().pop_size


def test_cli_rejects_removed_legacy_resnet_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--resnet_width", "128"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--resnet_blocks", "2"])
