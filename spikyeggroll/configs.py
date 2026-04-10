"""Hyperparameter configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SNNConfig:
    # Architecture
    n_inputs: int = 784
    hidden_size: int = 128
    n_classes: int = 10
    timesteps: int = 25

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
    lr: float = 0.005

    # Training
    batch_size: int = 256
    chunk_size: int = 0
    num_epochs: int = 400
    seed: int = 0
    log_interval: int = 10
    test_interval: int = 100
    checkpoint_interval: int = 100

    # Data
    dataset: str = "mnist"
    data_path: str = "data"
    dtype: str = "float32"

    # Run artifacts
    run_name: str = "default"
    log_dir: str = "logs/spikyeggroll"
    checkpoint_dir: str = "checkpoints/spikyeggroll"
    resume_from: Optional[str] = None
