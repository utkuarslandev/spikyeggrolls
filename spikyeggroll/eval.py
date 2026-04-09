"""Evaluation on MNIST test set."""

import jax
import jax.numpy as jnp

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.models.common import simple_es_tree_key

from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import SNNModel


def evaluate(cfg, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key,
             test_data, test_labels):
    """Evaluate model accuracy on a dataset using base (unperturbed) parameters.

    Args:
        cfg: SNNConfig
        frozen_noiser_params: EGGROLL frozen noiser params
        noiser_params: EGGROLL noiser params
        frozen_params: model frozen params
        params: model params
        es_tree_key: ES tree key
        test_data: [N, T, C] spike tensor
        test_labels: [N] integer labels

    Returns:
        accuracy (float)
    """
    # Forward with iterinfo=None -> no noise, use base params
    spike_counts = SNNModel.forward(
        EggRoll, frozen_noiser_params, noiser_params,
        frozen_params, params, es_tree_key, None, test_data
    )

    predictions = jnp.argmax(spike_counts, axis=-1)
    accuracy = jnp.mean(predictions == test_labels)
    return float(accuracy)
