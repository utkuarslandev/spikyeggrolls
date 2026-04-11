"""CIFAR-10 dataset loading and Poisson encoding."""

import jax
import jax.numpy as jnp
import numpy as np


def _normalize_cifar(images: np.ndarray) -> np.ndarray:
    """Normalize uint8 CIFAR images to float32 [0, 1]."""
    return images.astype(np.float32) / 255.0


def load_cifar10(path: str = "data/cifar10", augment: bool = False):
    """Load CIFAR-10 images and labels.

    Args:
        path: directory to store/load the dataset
        augment: currently unsupported placeholder

    Returns:
        (train_images [N,32,32,3] float32 in [0,1], train_labels [N] int32,
         test_images, test_labels)
    """
    if augment:
        raise NotImplementedError("CIFAR-10 augmentation is not implemented.")
    from torchvision import datasets

    train_dataset = datasets.CIFAR10(root=path, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=path, train=False, download=True)

    train_images = jnp.array(_normalize_cifar(train_dataset.data))
    train_labels = jnp.array(np.asarray(train_dataset.targets, dtype=np.int32))
    test_images = jnp.array(_normalize_cifar(test_dataset.data))
    test_labels = jnp.array(np.asarray(test_dataset.targets, dtype=np.int32))

    return train_images, train_labels, test_images, test_labels


def _poisson_encode_images(images, timesteps, key):
    """Encode image intensities as Poisson spike trains.

    Args:
        images: [N, H, W, C] float32 pixel values in [0, 1]
        timesteps: number of time steps T
        key: JAX PRNG key

    Returns:
        [N, T, C, H, W] float32 binary spike tensor
    """
    n, h, w, c = images.shape
    probs = jnp.transpose(images, (0, 3, 1, 2))[:, None, :, :, :]
    uniform = jax.random.uniform(key, (n, timesteps, c, h, w))
    spikes = (uniform < probs).astype(jnp.float32)
    return spikes


def encode_batch(images, timesteps, key):
    """Poisson-encode a batch of CIFAR images.

    Args:
        images: [B, 32, 32, 3] float32 pixel values in [0, 1]
        timesteps: number of time steps T
        key: JAX PRNG key

    Returns:
        [B, T, 3, 32, 32] float32 binary spike tensor
    """
    return _poisson_encode_images(images, timesteps, key)
