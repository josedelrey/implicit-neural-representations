"""Coordinate encodings inspired by NeRF and Fourier features.

NeRF: https://arxiv.org/abs/2003.08934; https://github.com/bmild/nerf
Fourier features: https://arxiv.org/abs/2006.10739; https://github.com/tancik/fourier-feature-networks
"""

import math
from collections.abc import Sequence

import torch
from torch import nn


def _check_coordinates(coords: torch.Tensor, in_features: int) -> None:
    if coords.ndim < 1 or coords.shape[-1] != in_features:
        raise ValueError(f'Expected coordinates with last dimension {in_features}')


class PositionalEncoding(nn.Module):
    """Raw coordinates plus sin/cos of 2**i * coords, without a pi factor."""

    def __init__(self, in_features: int, num_frequencies: int):
        super().__init__()
        if in_features <= 0 or num_frequencies < 0:
            raise ValueError(
                'in_features must be positive and num_frequencies non-negative'
            )
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_dim = in_features * (1 + 2 * num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        _check_coordinates(coords, self.in_features)
        parts = [coords]
        for i in range(self.num_frequencies):
            scaled = (2**i) * coords
            parts.extend((torch.sin(scaled), torch.cos(scaled)))
        return torch.cat(parts, dim=-1)


class FrequencyEncoding(nn.Module):
    """NeRF-style encoding with pi-scaled, per-coordinate sin/cos features."""

    def __init__(
        self,
        in_features: int,
        mapping_input: int | Sequence[int] | None = None,
        use_nyquist: bool = False,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError('in_features must be positive')
        self.in_features = in_features
        if in_features == 3:
            # Preserve the existing three-dimensional NeRF convention.
            self.num_frequencies = 10
        elif use_nyquist and in_features in (1, 2):
            if mapping_input is None:
                raise ValueError(
                    'mapping_input is required when using Nyquist frequencies'
                )
            if isinstance(mapping_input, int):
                samples = mapping_input
            elif (
                isinstance(mapping_input, Sequence)
                and not isinstance(mapping_input, (str, bytes))
                and len(mapping_input) > 0
            ):
                samples = min(mapping_input)
            else:
                raise ValueError(
                    'mapping_input must be a positive sample count or sequence'
                )
            if samples <= 0:
                raise ValueError('mapping_input must contain positive sample counts')
            self.num_frequencies = max(0, math.floor(math.log2(samples / 4)))
        else:
            self.num_frequencies = 4
        self.out_dim = in_features * (1 + 2 * self.num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        _check_coordinates(coords, self.in_features)
        parts = [coords]
        for i in range(self.num_frequencies):
            scaled = (2**i) * math.pi * coords
            for j in range(self.in_features):
                component = scaled[..., j : j + 1]
                parts.extend((torch.sin(component), torch.cos(component)))
        return torch.cat(parts, dim=-1)


class GaussianEncoding(nn.Module):
    """Gaussian random Fourier features, excluding the original coordinates."""

    def __init__(self, in_features: int, mapping_input: int, scale_B: float):
        super().__init__()
        if in_features <= 0 or mapping_input <= 0 or scale_B < 0:
            raise ValueError(
                'in_features and mapping_input must be positive; scale_B non-negative'
            )
        self.in_features = in_features
        self.out_dim = 2 * mapping_input
        self.register_buffer(
            'B_gauss', torch.randn(mapping_input, in_features) * scale_B
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        _check_coordinates(coords, self.in_features)
        projection = (2 * math.pi * coords) @ self.B_gauss.T
        return torch.cat((torch.sin(projection), torch.cos(projection)), dim=-1)


def positional_encoding(coords: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    """Functional form of the unscaled positional encoding."""
    return PositionalEncoding(coords.shape[-1], num_frequencies)(coords)
