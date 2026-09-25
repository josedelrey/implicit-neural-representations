# SPDX-License-Identifier: AGPL-3.0-only
"""MFN layers adapted from Fathony et al. (2021).

Adapted for this project in 2026.
Paper: https://openreview.net/forum?id=OmtmcPkkhT
Code: https://github.com/boschresearch/multiplicative-filter-networks

The wavelet and normalized variants are local extensions.
"""

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn


class MFNBase(nn.Module):
    """Base network combining filter and linear branches multiplicatively."""

    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out


class FourierLayer(nn.Module):
    """Sine filter used by FourierNet."""

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    """MFN with Fourier filters."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(hidden_layers + 1),
                )
                for _ in range(hidden_layers + 1)
            ]
        )


class GaborLayer(nn.Module):
    """Gabor filter used by GaborNet."""

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        D = (
            (x**2).sum(-1)[..., None]
            + (self.mu**2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


class GaborNet(MFNBase):
    """MFN with Gabor filters."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(hidden_layers + 1),
                    alpha / (hidden_layers + 1),
                    beta,
                )
                for _ in range(hidden_layers + 1)
            ]
        )


class WaveletLayer(nn.Module):
    """Wavelet filter with a scalar input frequency."""

    def __init__(
        self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, omega0=5.0
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.omega0 = omega0
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        D = (
            (x**2).sum(-1)[..., None]
            + (self.mu**2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(self.omega0 * x)) * torch.exp(
            -0.5 * D * self.gamma[None, :]
        )


class VectorWaveletLayer(nn.Module):
    """Wavelet filter with per-input-dimension frequencies."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: float,
        omega0: Union[float, Sequence[float], torch.Tensor],
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        if isinstance(omega0, (float, int)):
            omega = torch.full((in_features,), float(omega0))
        else:
            omega = torch.as_tensor(list(omega0), dtype=torch.float32)
            if omega.numel() != in_features:
                raise ValueError(
                    f"omega0 length {omega.numel()} != in_features {in_features}"
                )
        self.register_buffer('omega0', omega)

        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma)[:, None]
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * self.omega0.view(*((1,) * (x.dim() - 1)), -1)
        D = (
            (x**2).sum(-1)[..., None]
            + (self.mu**2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x_scaled)) * torch.exp(
            -0.5 * D * self.gamma[None, :]
        )


class WaveletNet(MFNBase):
    """MFN with wavelet filters."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        omega0=5.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                WaveletLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(hidden_layers + 1),
                    alpha / (hidden_layers + 1),
                    beta,
                    omega0,
                )
                for _ in range(hidden_layers + 1)
            ]
        )


class WaveletNetNormalized(MFNBase):
    """Wavelet MFN with layer normalization before multiplication."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        omega0=5.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                WaveletLayer(
                    in_features,
                    hidden_features,
                    input_scale / np.sqrt(hidden_layers + 1),
                    alpha / (hidden_layers + 1),
                    beta,
                    omega0,
                )
                for _ in range(hidden_layers + 1)
            ]
        )
        self.filter_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_features) for _ in range(hidden_layers + 1)]
        )
        self.linear_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_features) for _ in range(hidden_layers)]
        )

    def forward(self, x):
        out = self.filter_norms[0](self.filters[0](x))
        for i in range(1, len(self.filters)):
            filter_out = self.filters[i](x)
            filter_out = self.filter_norms[i](filter_out)
            linear_out = self.linear[i - 1](out)
            linear_out = self.linear_norms[i - 1](linear_out)
            out = filter_out * linear_out
        out = self.output_linear(out)
        if self.output_act:
            out = torch.sin(out)
        return out


class VectorWaveletNetNormalized(MFNBase):
    """Normalized wavelet MFN with per-dimension frequencies."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hidden_layers: int = 3,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        omega0: Union[float, Sequence[float], torch.Tensor] = 5.0,
        bias: bool = True,
        output_act: bool = False,
    ):
        super().__init__(
            hidden_features, out_features, hidden_layers, weight_scale, bias, output_act
        )
        layer_scale = input_scale / np.sqrt(hidden_layers + 1)
        self.filters = nn.ModuleList(
            [
                VectorWaveletLayer(
                    in_features,
                    hidden_features,
                    layer_scale,
                    omega0,
                    alpha / (hidden_layers + 1),
                    beta,
                )
                for _ in range(hidden_layers + 1)
            ]
        )
        self.filter_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_features) for _ in range(hidden_layers + 1)]
        )
        self.linear_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_features) for _ in range(hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.filter_norms[0](self.filters[0](x))
        for i in range(1, len(self.filters)):
            filter_out = self.filters[i](x)
            filter_out = self.filter_norms[i](filter_out)
            linear_out = self.linear[i - 1](out)
            linear_out = self.linear_norms[i - 1](linear_out)
            out = filter_out * linear_out
        out = self.output_linear(out)
        if self.output_act:
            out = torch.sin(out)
        return out
