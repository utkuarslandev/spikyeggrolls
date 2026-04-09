"""Tests for LIF neuron dynamics."""

import jax.numpy as jnp
from spikyeggroll.models.snn import lif_step


def test_lif_no_spike_below_threshold():
    """Membrane below threshold should not spike."""
    V = jnp.zeros((1, 4))
    I = jnp.ones((1, 4)) * 0.5  # below threshold of 1.0
    V_new, spikes = lif_step(V, I, beta=0.9, threshold=1.0)
    assert jnp.all(spikes == 0)
    assert jnp.allclose(V_new, jnp.ones((1, 4)) * 0.5)


def test_lif_spike_above_threshold():
    """Membrane above threshold should spike and reset."""
    V = jnp.ones((1, 4)) * 0.6
    I = jnp.ones((1, 4)) * 0.6  # beta*0.6 + 0.6 = 1.14 > 1.0
    V_new, spikes = lif_step(V, I, beta=0.9, threshold=1.0)
    assert jnp.all(spikes == 1)
    # After reset: 1.14 - 1.0 = 0.14
    assert jnp.allclose(V_new, jnp.ones((1, 4)) * 0.14, atol=1e-5)


def test_lif_decay():
    """Membrane should decay with beta when no input."""
    V = jnp.ones((1, 4)) * 0.5
    I = jnp.zeros((1, 4))
    V_new, spikes = lif_step(V, I, beta=0.9, threshold=1.0)
    assert jnp.all(spikes == 0)
    assert jnp.allclose(V_new, jnp.ones((1, 4)) * 0.45)
