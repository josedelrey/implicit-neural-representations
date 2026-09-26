"""Create and save the files belonging to one experiment run."""

import json
import math
from pathlib import Path

import torch


def initialize_run(run_directory: str, resolved_config: dict) -> Path:
    """Create a run directory and write its resolved configuration."""
    directory = Path(run_directory)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run directory already exists; choose a new directory: {directory}"
        ) from exc
    (directory / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return directory


def save_checkpoint(
    run_directory: Path,
    *,
    model_type: str,
    model_kwargs: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
) -> None:
    """Save the trained model and optimizer state."""
    torch.save(
        {
            "completed_steps": completed_steps,
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        run_directory / "checkpoint.pt",
    )


def save_metrics(run_directory: Path, metrics: dict[str, float]) -> None:
    """Save final scalar metrics as standards-compliant JSON."""
    if any(math.isnan(value) for value in metrics.values()):
        raise ValueError("metrics must not contain NaN values")
    serialized_metrics = {}
    for name, value in metrics.items():
        if math.isinf(value):
            serialized_metrics[name] = "Infinity" if value > 0 else "-Infinity"
        else:
            serialized_metrics[name] = value
    (run_directory / "metrics.json").write_text(
        json.dumps(serialized_metrics, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
