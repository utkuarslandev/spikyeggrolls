"""Tests for CIFAR-10 encoding and spiking ResNet wiring."""

import jax
import jax.numpy as jnp

from spikyeggroll.data.cifar10 import encode_batch
from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll


def test_cifar_encode_batch_shape_and_binary():
    key = jax.random.key(0)
    images = jax.random.uniform(key, (8, 32, 32, 3))
    spikes = encode_batch(images, timesteps=4, key=key)
    assert spikes.shape == (8, 4, 3072)
    unique = jnp.unique(spikes)
    assert jnp.all((unique == 0.0) | (unique == 1.0))


def test_spiking_resnet_forward_shape():
    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
        timesteps=4,
        pop_size=8,
        resnet_width=128,
        resnet_blocks=2,
    )
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )
    x = jax.random.bernoulli(k3, 0.3, (4, 4, 3072)).astype(jnp.float32)
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
