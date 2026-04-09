"""Tests for MNIST data loading, encoding, and EGGROLL training setup."""

import pytest
import jax
import jax.numpy as jnp

from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.data.mnist import load_mnist, encode_batch, _poisson_encode

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.models.common import simple_es_tree_key


# --- Data loading ---

def test_load_mnist_shapes():
    """Raw MNIST images should be [N, 784] float32 in [0, 1]."""
    train_imgs, train_labels, test_imgs, test_labels = load_mnist("data/mnist")
    assert train_imgs.shape == (60000, 784)
    assert test_imgs.shape == (10000, 784)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)
    assert train_imgs.dtype == jnp.float32
    assert float(train_imgs.min()) >= 0.0
    assert float(train_imgs.max()) <= 1.0


def test_load_mnist_label_range():
    """Labels should be 0-9."""
    _, train_labels, _, test_labels = load_mnist("data/mnist")
    assert int(train_labels.min()) == 0
    assert int(train_labels.max()) == 9


# --- Poisson encoding ---

def test_encode_batch_shape():
    """Encoded batch should be [B, T, 784]."""
    key = jax.random.key(0)
    images = jax.random.uniform(key, (32, 784))
    spikes = encode_batch(images, timesteps=25, key=key)
    assert spikes.shape == (32, 25, 784)


def test_encode_batch_binary():
    """Spikes should be binary {0, 1}."""
    key = jax.random.key(0)
    images = jax.random.uniform(key, (32, 784))
    spikes = encode_batch(images, timesteps=25, key=key)
    unique = jnp.unique(spikes)
    assert jnp.all((unique == 0.0) | (unique == 1.0))


def test_encode_rate_proportional():
    """Brighter pixels should produce more spikes on average."""
    key = jax.random.key(42)
    # 2 pixels: one dark (0.1), one bright (0.9)
    images = jnp.array([[0.1, 0.9]])
    spikes = encode_batch(images, timesteps=10000, key=key)
    rates = spikes.mean(axis=1)  # [1, 2]
    assert float(rates[0, 0]) < float(rates[0, 1])
    # Check rates are approximately correct
    assert abs(float(rates[0, 0]) - 0.1) < 0.02
    assert abs(float(rates[0, 1]) - 0.9) < 0.02


def test_encode_stochastic():
    """Different keys should produce different spike trains."""
    images = jnp.ones((1, 784)) * 0.5
    s1 = encode_batch(images, 25, jax.random.key(0))
    s2 = encode_batch(images, 25, jax.random.key(1))
    assert not jnp.array_equal(s1, s2)


# --- Model setup for MNIST ---

def test_mnist_model_init():
    """SNNModel should initialize with MNIST dimensions."""
    key = jax.random.key(0)
    cfg = SNNConfig(n_inputs=784, hidden_size=128, n_classes=10, timesteps=25)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(key, cfg)

    assert params["linear1"]["weight"].shape == (128, 784)
    assert params["linear2"]["weight"].shape == (128, 128)
    assert params["linear_out"]["weight"].shape == (10, 128)


def test_mnist_forward_pass():
    """Forward pass should produce [B, 10] spike counts."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)

    cfg = SNNConfig(n_inputs=784, hidden_size=128, n_classes=10, timesteps=25)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)

    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )

    # Fake batch of Poisson spikes
    x = jax.random.bernoulli(k3, 0.5, (8, 25, 784)).astype(jnp.float32)

    # No-noise forward (eval mode)
    out = SNNModel.forward(
        EggRoll, frozen_noiser_params, noiser_params,
        frozen_params, params, es_tree_key, None, x
    )
    assert out.shape == (8, 10)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out >= 0)  # spike counts are non-negative


def test_mnist_population_forward():
    """Vmapped population forward should produce [N, B, 10]."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 16  # small population

    cfg = SNNConfig(n_inputs=784, hidden_size=128, n_classes=10, timesteps=25, pop_size=N)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)

    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )

    x = jax.random.bernoulli(k3, 0.5, (4, 25, 784)).astype(jnp.float32)
    iterinfo = (jnp.full(N, 0, dtype=jnp.int32), jnp.arange(N))

    pop_out = jax.vmap(
        lambda n, p, i, x: SNNModel.forward(
            EggRoll, frozen_noiser_params, n,
            frozen_params, p, es_tree_key, i, x
        ),
        in_axes=(None, None, 0, None),
    )(noiser_params, params, iterinfo, x)

    assert pop_out.shape == (N, 4, 10)
    assert jnp.all(jnp.isfinite(pop_out))

    # Population members should have DIFFERENT outputs (noise is working)
    assert not jnp.allclose(pop_out[0], pop_out[1])


def test_mnist_fitness_improves():
    """Fitness should improve over a few EGGROLL steps."""
    import optax
    from spikyeggroll.train import compute_fitness

    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 32

    cfg = SNNConfig(n_inputs=784, hidden_size=64, n_classes=10, timesteps=10, pop_size=N)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)

    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, solver=optax.adamw, rank=cfg.rank
    )

    # Fixed batch
    x = jax.random.bernoulli(k3, 0.3, (16, 10, 784)).astype(jnp.float32)
    y = jax.random.randint(k3, (16,), 0, 10)

    jit_forward = jax.jit(jax.vmap(
        lambda n, p, i, xp: SNNModel.forward(
            EggRoll, frozen_noiser_params, n,
            frozen_params, p, es_tree_key, i, xp
        ),
        in_axes=(None, None, 0, None),
    ))
    jit_forward_eval = jax.jit(
        lambda n, p, xp: SNNModel.forward(
            EggRoll, frozen_noiser_params, n,
            frozen_params, p, es_tree_key, None, xp
        )
    )
    jit_update = jax.jit(
        lambda n, p, f, i: EggRoll.do_updates(
            frozen_noiser_params, n, p, es_tree_key, f, i, es_map
        )
    )

    # Get initial fitness
    init_out = jit_forward_eval(noiser_params, params, x)
    init_fitness = float(compute_fitness(init_out, y))

    # Train for 30 steps
    for epoch in range(30):
        iterinfo = (jnp.full(N, epoch, dtype=jnp.int32), jnp.arange(N))
        pop_out = jit_forward(noiser_params, params, iterinfo, x)
        raw_scores = jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, y)
        fitnesses = EggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

    final_out = jit_forward_eval(noiser_params, params, x)
    final_fitness = float(compute_fitness(final_out, y))

    assert final_fitness > init_fitness, f"Fitness did not improve: {init_fitness:.4f} -> {final_fitness:.4f}"
