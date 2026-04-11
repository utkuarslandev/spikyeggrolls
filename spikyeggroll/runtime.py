"""Shared runtime selectors for datasets and models."""

from dataclasses import dataclass
from typing import Callable

from spikyeggroll.configs import SNNConfig
from spikyeggroll.data.cifar10 import load_cifar10, encode_batch as encode_cifar_batch
from spikyeggroll.data.mnist import load_mnist, encode_batch as encode_mnist_batch
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model


@dataclass(frozen=True)
class DatasetSpec:
    loader: Callable
    encoder: Callable
    n_inputs: int
    in_channels: int
    image_size: int


def get_model_cls(model_name: str):
    if model_name == "mlp_snn":
        return SNNModel
    if model_name == "spiking_resnet18":
        return SpikingResNet18Model
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def get_dataset_spec(cfg: SNNConfig) -> DatasetSpec:
    if cfg.dataset == "mnist":
        return DatasetSpec(
            loader=lambda: load_mnist(cfg.data_path + "/mnist"),
            encoder=encode_mnist_batch,
            n_inputs=784,
            in_channels=1,
            image_size=28,
        )
    if cfg.dataset == "cifar10":
        return DatasetSpec(
            loader=lambda: load_cifar10(cfg.data_path + "/cifar10", augment=cfg.augment),
            encoder=encode_cifar_batch,
            n_inputs=32 * 32 * 3,
            in_channels=3,
            image_size=32,
        )
    raise ValueError(f"Unsupported dataset '{cfg.dataset}'.")
