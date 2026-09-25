"""Shared runtime setup for image and video experiments."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .artifacts import initialize_run
from .config import ExperimentConfig
from .dataset import SignalData
from .model_factory import build_model


@dataclass(frozen=True)
class ExperimentRuntime:
    """Resources shared by one image or video experiment."""

    device: torch.device
    model: nn.Module
    optimizer: torch.optim.Optimizer
    writer: SummaryWriter
    run_directory: Path
    reconstruction_path: Path


def _select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Using device: {name}")
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@contextmanager
def open_experiment(
    config: ExperimentConfig, signal: SignalData
) -> Iterator[ExperimentRuntime]:
    """Initialize a run and close its TensorBoard writer on every exit path."""
    device = _select_device()
    _seed_everything(config.seed, device)

    model = build_model(config.model_type, **config.model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    resolved_config = config.resolved_dict(
        device=str(device), value_range=signal.value_range
    )
    run_directory = initialize_run(config.run_directory, resolved_config)
    reconstruction_path = config.reconstruction_path
    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir=run_directory / "tensorboard") as writer:
        writer.add_text(
            "resolved_config",
            json.dumps(resolved_config, indent=2, sort_keys=True),
        )
        yield ExperimentRuntime(
            device=device,
            model=model,
            optimizer=optimizer,
            writer=writer,
            run_directory=run_directory,
            reconstruction_path=reconstruction_path,
        )
