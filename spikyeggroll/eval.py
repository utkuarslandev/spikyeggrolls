"""Model evaluation helpers."""

import jax.numpy as jnp

from hyperscalees.noiser.eggroll import EggRoll

from spikyeggroll.runtime import get_model_cls


def evaluate(
    cfg,
    frozen_noiser_params,
    noiser_params,
    frozen_params,
    params,
    es_tree_key,
    test_data,
    test_labels,
):
    """Evaluate model accuracy on spike-encoded inputs using base parameters."""
    model_cls = get_model_cls(cfg.model_name)
    outputs = model_cls.forward(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
        test_data,
    )

    predictions = jnp.argmax(outputs, axis=-1)
    accuracy = jnp.mean(predictions == test_labels)
    return float(accuracy)
