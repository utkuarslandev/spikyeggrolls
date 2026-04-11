import pytest

from spikyeggroll.configs import SNNConfig
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model
from spikyeggroll.runtime import get_dataset_spec, get_model_cls


@pytest.mark.parametrize(
    ("model_name", "expected_cls"),
    [
        ("mlp_snn", SNNModel),
        ("spiking_resnet18", SpikingResNet18Model),
    ],
)
def test_get_model_cls_supported_models(model_name, expected_cls):
    assert get_model_cls(model_name) is expected_cls


def test_get_model_cls_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model_name"):
        get_model_cls("unknown_model")


@pytest.mark.parametrize(
    ("cfg", "n_inputs", "in_channels", "image_size"),
    [
        (SNNConfig(dataset="mnist", data_path="/tmp/data"), 784, 1, 28),
        (SNNConfig(dataset="cifar10", data_path="/tmp/data", augment=True), 3072, 3, 32),
    ],
)
def test_get_dataset_spec_supported_datasets(cfg, n_inputs, in_channels, image_size):
    spec = get_dataset_spec(cfg)

    assert spec.n_inputs == n_inputs
    assert spec.in_channels == in_channels
    assert spec.image_size == image_size
    assert callable(spec.loader)
    assert callable(spec.encoder)


def test_get_dataset_spec_rejects_unknown_dataset():
    cfg = SNNConfig(dataset="unknown_dataset")

    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_dataset_spec(cfg)
