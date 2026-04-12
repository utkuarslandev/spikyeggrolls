"""Hyperparameter configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SNNConfig:
    # Architecture
    n_inputs: int = 784
    hidden_size: int = 128
    n_classes: int = 10
    timesteps: int = 25
    model_name: str = "mlp_snn"

    # LIF neuron
    beta: float = 0.95
    threshold: float = 1.0
    membrane_readout: bool = False
    escape_noise: bool = False
    escape_beta: float = 50.0
    escape_lambda0: float = 1.0

    # EGGROLL
    pop_size: int = 10000
    rank: int = 3
    sigma: float = 0.007
    sigma_min: float = 0.001
    sigma_max: float = 1.0
    lr: float = 0.005
    fitness_shaping: str = "zscore"
    use_batched_update: bool = False

    # Training
    batch_size: int = 256
    chunk_size: int = 0
    num_epochs: int = 400
    updates_per_epoch: int = 1
    seed: int = 0
    log_interval: int = 10
    test_interval: int = 100
    checkpoint_interval: int = 100
    sigma_warmup_epochs: int = 20

    # Data
    dataset: str = "mnist"
    data_path: str = "data"
    dtype: str = "float32"
    in_channels: int = 1
    image_size: int = 28
    augment: bool = False
    num_test_eval_samples: int = 0

    # Convolutional spiking ResNet options (CIFAR-10 path)
    resnet_channels_base: int = 64
    resnet_block_counts: tuple[int, int, int, int] = field(
        default_factory=lambda: (2, 2, 2, 2)
    )
    resnet_norm: str = "group"
    resnet_norm_groups: int = 8
    resnet_threshold_scale: bool = False  # if True, stage i uses threshold * 2**i

    # Run artifacts
    run_name: str = "default"
    log_dir: str = "logs/spikyeggroll"
    checkpoint_dir: str = "checkpoints/spikyeggroll"
    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.dataset == "cifar10":
            if self.n_inputs == 784:
                self.n_inputs = 3072
            if self.in_channels == 1:
                self.in_channels = 3
            if self.image_size == 28:
                self.image_size = 32
