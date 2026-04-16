"""CIFAR-10 dataset loading and Poisson encoding."""

import jax
import jax.numpy as jnp
import numpy as np


def _normalize_cifar(images: np.ndarray) -> np.ndarray:
    """Normalize uint8 CIFAR images to float32 [0, 1]."""
    return images.astype(np.float32) / 255.0


def _random_horizontal_flip(images, key):
    """Randomly flip [B, H, W, C] images horizontally, independently per image."""
    flip_mask = jax.random.bernoulli(key, 0.5, shape=(images.shape[0],))
    return jnp.where(flip_mask[:, None, None, None], jnp.flip(images, axis=2), images)


def _random_crop(images, key, pad=4):
    """Pad by `pad` pixels each side (reflect), then random-crop back to original size."""
    B, H, W, C = images.shape
    padded = jnp.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    keys = jax.random.split(key, B)

    def crop_one(img, k):
        offsets = jax.random.randint(k, shape=(2,), minval=0, maxval=2 * pad)
        return jax.lax.dynamic_slice(img, (offsets[0], offsets[1], 0), (H, W, C))

    return jax.vmap(crop_one)(padded, keys)


def _cutmix(images, key, alpha=1.0):
    """CutMix augmentation for a batch of CIFAR images (hard labels, no label mixing).

    Cuts a rectangular patch from a randomly permuted donor image and pastes it into
    each original image. Lambda is clamped to >= 0.5 so the original image always
    contributes the majority of pixels, keeping the original label correct.

    Args:
        images: [B, H, W, C] float32 in [0, 1]
        key: JAX PRNG key
        alpha: Beta(alpha, alpha) shape parameter (1.0 = Uniform)

    Returns:
        [B, H, W, C] float32 in [0, 1]
    """
    B, H, W, C = images.shape
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Cut ratio: clamp so the original label always dominates
    lam = jax.random.beta(k1, alpha, alpha)
    lam = jnp.maximum(lam, 1.0 - lam)

    # Box size proportional to sqrt(1 - lam)
    cut_h = jnp.int32(H * jnp.sqrt(1.0 - lam))
    cut_w = jnp.int32(W * jnp.sqrt(1.0 - lam))

    # Random box centre
    cx = jax.random.randint(k2, (), 0, H)
    cy = jax.random.randint(k3, (), 0, W)

    x1 = jnp.clip(cx - cut_h // 2, 0, H)
    x2 = jnp.clip(cx + cut_h // 2, 0, H)
    y1 = jnp.clip(cy - cut_w // 2, 0, W)
    y2 = jnp.clip(cy + cut_w // 2, 0, W)

    # Spatial mask: 1 inside the cut box → replaced by donor
    rows = jnp.arange(H)[:, None]   # [H, 1]
    cols = jnp.arange(W)[None, :]   # [1, W]
    mask = ((rows >= x1) & (rows < x2) & (cols >= y1) & (cols < y2)).astype(images.dtype)
    mask = mask[None, :, :, None]   # [1, H, W, 1] broadcast over batch & channel

    # Random donor permutation (same permutation for all images in the batch)
    perm = jax.random.permutation(k4, B)
    donors = images[perm]

    return images * (1.0 - mask) + donors * mask


def augment_batch(images, key, cutmix: bool = False, cutmix_alpha: float = 1.0):
    """Standard CIFAR-10 augmentation: random crop (pad=4) + random horizontal flip,
    optionally followed by CutMix.

    Args:
        images: [B, H, W, C] float32 in [0, 1]
        key: JAX PRNG key
        cutmix: whether to apply CutMix after flip/crop
        cutmix_alpha: Beta distribution shape parameter for CutMix

    Returns:
        [B, H, W, C] float32 in [0, 1]
    """
    k1, k2, k3 = jax.random.split(key, 3)
    images = _random_crop(images, k1, pad=4)
    images = _random_horizontal_flip(images, k2)
    if cutmix:
        images = _cutmix(images, k3, alpha=cutmix_alpha)
    return images


def load_cifar10(path: str = "data/cifar10", augment: bool = False):
    """Load CIFAR-10 images and labels.

    Args:
        path: directory to store/load the dataset
        augment: unused at load time; augmentation is applied per-batch via augment_batch()

    Returns:
        (train_images [N,32,32,3] float32 in [0,1], train_labels [N] int32,
         test_images, test_labels)
    """
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


def encode_batch_direct(images, timesteps, key):
    """Direct coding: repeat normalized pixel values at every timestep.

    Deterministic alternative to Poisson encoding. Each pixel value is used as a
    constant current injection at every timestep. Equivalent to Poisson in expectation
    but without temporal variance — allows T=4 to match the representational capacity
    of Poisson at T=25 while being ~6x faster.

    Args:
        images: [B, H, W, C] float32 pixel values in [0, 1]
        timesteps: number of time steps T (typically 4–6)
        key: ignored (kept for API compatibility with encode_batch)

    Returns:
        [B, T, C, H, W] float32 — same image current broadcast at every timestep
    """
    n, h, w, c = images.shape
    # Transpose to NCHW then broadcast over T dimension
    nchw = jnp.transpose(images, (0, 3, 1, 2))  # [N, C, H, W]
    return jnp.broadcast_to(nchw[:, None, :, :, :], (n, timesteps, c, h, w))
