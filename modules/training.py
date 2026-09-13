"""Shared fitting and chunked prediction for image and video experiments."""

import datetime
from typing import Literal

import torch

from modules.utils import log_training_metrics


Sampling = Literal['full', 'random']


def _model_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else fallback


def fit(
    model: torch.nn.Module,
    coords: torch.Tensor,
    pixels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    log_interval: int,
    writer,
    sampling: Sampling,
    batch_size: int | None = None,
) -> None:
    """Run exactly ``total_steps`` optimizer updates on full or sampled coordinates."""
    if total_steps < 0:
        raise ValueError('total_steps must be non-negative')
    if log_interval <= 0:
        raise ValueError('log_interval must be positive')
    if coords.ndim != 2 or pixels.ndim != 2 or coords.shape[0] != pixels.shape[0]:
        raise ValueError('coords and pixels must have matching [N, features] shapes')
    if coords.shape[0] == 0:
        raise ValueError('coords and pixels must not be empty')
    if coords.device != pixels.device:
        raise ValueError('coords and pixels must be on the same device')
    if sampling == 'random':
        if batch_size is None or batch_size <= 0:
            raise ValueError('random sampling requires a positive batch_size')
    elif sampling == 'full':
        if batch_size is not None:
            raise ValueError('full sampling does not use batch_size')
    else:
        raise ValueError(f'Unknown sampling mode: {sampling!r}')

    device = _model_device(model, coords.device)
    if sampling == 'full':
        full_coords = coords.to(device)
        full_pixels = pixels.to(device)

    model.train()
    start_time = datetime.datetime.now()
    for step in range(1, total_steps + 1):
        if sampling == 'random':
            indices = torch.randint(0, coords.shape[0], (batch_size,), device=coords.device)
            batch_coords = coords[indices].to(device)
            batch_pixels = pixels[indices].to(device)
        else:
            batch_coords, batch_pixels = full_coords, full_pixels

        optimizer.zero_grad()
        predictions = model(batch_coords)
        if predictions.shape != batch_pixels.shape:
            raise ValueError('model predictions and target pixels must have the same shape')
        loss = ((predictions - batch_pixels) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == total_steps:
            log_training_metrics(step, loss, start_time, writer)


def predict_chunks(model: torch.nn.Module, coords: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Evaluate coordinates in order and return their predictions on the CPU."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError('coords must be a non-empty [N, features] tensor')

    device = _model_device(model, coords.device)
    model.eval()
    with torch.no_grad():
        return torch.cat([
            model(coords[start:start + chunk_size].to(device)).cpu()
            for start in range(0, coords.shape[0], chunk_size)
        ], dim=0)
