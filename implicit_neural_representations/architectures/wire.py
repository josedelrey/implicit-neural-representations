# SPDX-License-Identifier: MIT
"""WIRE layers adapted from Saragadam et al. (2023).

Paper: https://arxiv.org/abs/2301.05187
Code: https://github.com/vishwa91/wire
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from ..encoding import PositionalEncoding


class RealGaborLayer(nn.Module):
    """Real Gabor activation layer."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=10.0,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale**2))


class ComplexGaborLayer(nn.Module):
    """Complex Gabor activation layer."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=40.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class WIRE(nn.Module):
    """Complex Gabor network with optional positional encoding."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
        pos_encode=False,
        L=6,
    ):
        super().__init__()

        self.pos_encode = pos_encode
        self.encoding = (
            PositionalEncoding(in_features, L) if pos_encode else nn.Identity()
        )
        encoded_features = self.encoding.out_dim if pos_encode else in_features

        self.nonlin = ComplexGaborLayer
        self.complex = True
        self.wavelet = 'gabor'

        layers = OrderedDict()
        layers["layer0"] = self.nonlin(
            encoded_features,
            hidden_features,
            omega0=first_omega_0,
            sigma0=scale,
            is_first=True,
            trainable=False,
        )

        for i in range(hidden_layers):
            layers[f"layer{i + 1}"] = self.nonlin(
                hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale
            )

        if outermost_linear:
            layers["final"] = nn.Linear(
                hidden_features, out_features, dtype=torch.cfloat
            )
        else:
            layers["final"] = self.nonlin(
                hidden_features, out_features, omega0=hidden_omega_0, sigma0=scale
            )

        self.net = nn.Sequential(layers)

    def forward(self, coords):
        coords = self.encoding(coords)
        output = self.net(coords)
        if self.wavelet == 'gabor':
            return output.real
        return output
