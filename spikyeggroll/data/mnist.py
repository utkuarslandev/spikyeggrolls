"""MNIST dataset loading with Poisson rate encoding for SNNs."""

import jax
import jax.numpy as jnp
import numpy as np


def load_mnist(path: str = "data/mnist"):
    """Load raw MNIST images and labels (no spike encoding yet).

    Args:
        path: directory to store/load the dataset

    Returns:
        (train_images [N,784] float32 in [0,1], train_labels [N] int32,
         test_images, test_labels)
    """
    from torchvision import datasets

    train_dataset = datasets.MNIST(root=path, train=True, download=True)
    test_dataset = datasets.MNIST(root=path, train=False, download=True)

    train_images = jnp.array(train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0)
    train_labels = jnp.array(train_dataset.targets.numpy().astype(np.int32))
    test_images = jnp.array(test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0)
    test_labels = jnp.array(test_dataset.targets.numpy().astype(np.int32))

    return train_images, train_labels, test_images, test_labels


def _poisson_encode(images, timesteps, key):
    """Encode pixel intensities as Poisson spike trains.

    Args:
        images: [N, 784] float32 pixel values in [0, 1]
        timesteps: number of time steps T
        key: JAX PRNG key

    Returns:
        [N, T, 784] float32 binary spike tensor
    """
    N, C = images.shape
    # [N, 1, C] probabilities broadcast against [N, T, C] uniform samples
    probs = jnp.array(images)[:, None, :]  # [N, 1, 784]
    uniform = jax.random.uniform(key, (N, timesteps, C))
    spikes = (uniform < probs).astype(jnp.float32)
    return spikes


def encode_batch(images, timesteps, key):
    """Poisson-encode a batch of images (call per mini-batch, not full dataset).

    Args:
        images: [B, 784] float32 pixel values in [0, 1]
        timesteps: number of time steps T
        key: JAX PRNG key

    Returns:
        [B, T, 784] float32 binary spike tensor
    """
    return _poisson_encode(images, timesteps, key)
