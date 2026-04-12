import jax
import jax.numpy as jnp

from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll
from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import SNNModel


def test_centered_rank_fitness_shaping_is_deterministic():
    frozen, noiser = EggRoll.init_noiser(
        {"w": jnp.zeros((4, 4), dtype=jnp.float32)},
        sigma=0.01,
        lr=0.001,
        rank=1,
        fitness_shaping="centered_rank",
    )
    scores = jnp.array([0.2, 0.2, 0.9, -1.0], dtype=jnp.float32)

    shaped_a = EggRoll.convert_fitnesses(frozen, noiser, scores)
    shaped_b = EggRoll.convert_fitnesses(frozen, noiser, scores)

    assert jnp.allclose(shaped_a, shaped_b)
    assert jnp.all(jnp.diff(jnp.sort(shaped_a)) >= 0)
    assert jnp.isclose(jnp.sum(shaped_a), 0.0, atol=1e-6)


def test_batched_update_matches_original_update():
    cfg = SNNConfig(n_inputs=8, hidden_size=4, n_classes=2, pop_size=4, rank=1)
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    iterinfo = (jnp.zeros((cfg.pop_size,), dtype=jnp.int32), jnp.arange(cfg.pop_size))

    frozen_orig, noiser_orig = EggRoll.init_noiser(
        params,
        sigma=cfg.sigma,
        lr=cfg.lr,
        rank=cfg.rank,
        fitness_shaping="zscore",
        use_batched_update=False,
    )
    frozen_batch, noiser_batch = EggRoll.init_noiser(
        params,
        sigma=cfg.sigma,
        lr=cfg.lr,
        rank=cfg.rank,
        fitness_shaping="zscore",
        use_batched_update=True,
    )
    fitnesses = jnp.array([-1.0, -0.25, 0.25, 1.0], dtype=jnp.float32)

    _, updated_orig = EggRoll.do_updates(
        frozen_orig, noiser_orig, params, es_tree_key, fitnesses, iterinfo, es_map
    )
    _, updated_batch = EggRoll.do_updates(
        frozen_batch, noiser_batch, params, es_tree_key, fitnesses, iterinfo, es_map
    )

    leaves_orig = jax.tree.leaves(updated_orig)
    leaves_batch = jax.tree.leaves(updated_batch)
    for lhs, rhs in zip(leaves_orig, leaves_batch, strict=True):
        assert jnp.allclose(lhs, rhs, atol=1e-6, rtol=1e-6)
