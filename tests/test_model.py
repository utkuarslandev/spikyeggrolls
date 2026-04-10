"""Tests for SNNModel initialization and forward pass."""

import pytest
import jax
import jax.numpy as jnp

from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model


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
    """SpikingResNet18 init should expose stem/output and residual block params."""
    key = jax.random.key(7)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
        resnet_width=128,
        resnet_blocks=2,
    )
    frozen_params, params, _, _ = SpikingResNet18Model.rand_init(key, cfg)
    assert "linear1" in params
    assert "linear_out" in params
    assert "block0_a" in params
    assert "block1_b" in params
    assert params["linear1"]["weight"].shape == (128, 3072)
    assert params["linear_out"]["weight"].shape == (10, 128)
    assert frozen_params["resnet_blocks"] == 2
