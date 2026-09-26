"""Sine-based FINER++ implemented from Zhu et al. (2024).

Paper: https://arxiv.org/abs/2407.19434

The activation follows Equation 5 of the paper:
``sin(omega * (abs(x) + 1) * x)``. Bias scales select the range of
variable-periodic components as described by Equation 6.
"""

from math import isfinite, sqrt

import torch
from torch import nn


def finer_activation(inputs: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """Apply the variable-periodic sine activation from FINER++."""
    return torch.sin(omega * (inputs.abs() + 1.0) * inputs)


def _validate_scale(value: float | None, name: str) -> None:
    if value is not None and (
        type(value) not in (int, float) or not isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be a finite, non-negative number or None")


class FinerLayer(nn.Module):
    """Affine layer followed by FINER++ activation unless it is the output."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        omega: float = 30.0,
        is_first: bool = False,
        is_last: bool = False,
        init_method: str = "sine",
        init_gain: float = 1.0,
        bias_scale: float | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not isfinite(omega) or omega <= 0:
            raise ValueError("omega must be a finite, positive number")
        if not isfinite(init_gain) or init_gain <= 0:
            raise ValueError("init_gain must be a finite, positive number")
        if init_method not in {"sine", "pytorch"}:
            raise ValueError("init_method must be 'sine' or 'pytorch'")
        _validate_scale(bias_scale, "bias_scale")

        self.omega = float(omega)
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._initialize(is_first, init_method, init_gain, bias_scale)

    def _initialize(
        self,
        is_first: bool,
        init_method: str,
        init_gain: float,
        bias_scale: float | None,
    ) -> None:
        with torch.no_grad():
            if init_method == "sine":
                if is_first:
                    bound = 1.0 / self.linear.in_features
                else:
                    bound = sqrt(6.0 * init_gain / self.linear.in_features) / self.omega
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None and bias_scale is not None:
                self.linear.bias.uniform_(-bias_scale, bias_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.linear(inputs)
        return values if self.is_last else finer_activation(values, self.omega)


class Finer(nn.Module):
    """MLP using the sine member of the FINER++ activation family."""

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 3,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
        init_method: str = "sine",
        init_gain: float = 1.0,
        fbs: float | None = None,
        hbs: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 0 or hidden_features <= 0:
            raise ValueError(
                "hidden_layers must be non-negative and hidden_features positive"
            )
        _validate_scale(fbs, "fbs")
        _validate_scale(hbs, "hbs")

        layers: list[nn.Module] = [
            FinerLayer(
                in_features,
                hidden_features,
                omega=first_omega,
                is_first=True,
                init_method=init_method,
                init_gain=init_gain,
                bias_scale=fbs,
            )
        ]
        layers.extend(
            FinerLayer(
                hidden_features,
                hidden_features,
                omega=hidden_omega,
                init_method=init_method,
                init_gain=init_gain,
                bias_scale=hbs,
            )
            for _ in range(hidden_layers)
        )
        layers.append(
            FinerLayer(
                hidden_features,
                out_features,
                omega=hidden_omega,
                is_last=True,
                init_method=init_method,
                init_gain=init_gain,
                bias_scale=hbs,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
