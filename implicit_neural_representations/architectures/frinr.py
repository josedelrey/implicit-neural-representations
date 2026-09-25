"""FR-INR implemented from Shi, Zhou, and Gu (2024).

Paper: https://arxiv.org/abs/2401.07402

Hidden-to-hidden weights can be represented as ``W = coefficients @ basis``.
The coefficient matrix is trainable and the cosine basis is fixed, matching
Equations 2--4 of the paper.
"""

import math
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from ..encoding import FrequencyEncoding


def _fourier_basis(
    in_features: int,
    frequency_num: int,
    phase_num: int,
) -> torch.Tensor:
    """Construct fixed cosine bases across the maximum selected period."""
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if frequency_num <= 0:
        raise ValueError("frequency_num must be positive")
    if phase_num <= 0:
        raise ValueError("phase_num must be positive")

    indices = torch.arange(1, frequency_num + 1, dtype=torch.float64)
    frequency = torch.cat((indices / frequency_num, indices))
    phase = 2.0 * math.pi * torch.arange(phase_num, dtype=torch.float64) / phase_num

    maximum_period = 2.0 * math.pi * frequency_num
    positions = torch.linspace(
        -maximum_period / 2.0,
        maximum_period / 2.0,
        in_features,
        dtype=torch.float64,
    )
    basis = torch.cos(
        frequency[:, None, None] * positions[None, None, :] + phase[None, :, None]
    )
    return basis.reshape(-1, in_features).float()


class FourierReparameterizedLinear(nn.Module):
    """Linear map whose effective weight is a learned mixture of fixed bases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        frequency_num: int,
        phase_num: int,
        *,
        omega: float | None = None,
    ) -> None:
        super().__init__()
        basis = _fourier_basis(in_features, frequency_num, phase_num)
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("basis", basis)
        self.coefficients = nn.Parameter(torch.empty(out_features, basis.shape[0]))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._initialize_coefficients(omega)

    def _initialize_coefficients(self, omega: float | None) -> None:
        if omega is not None and (not math.isfinite(omega) or omega <= 0):
            raise ValueError("omega must be a finite, positive number")
        basis_count = self.basis.shape[0]
        norms = self.basis.norm(dim=1)
        if torch.any(norms == 0):
            raise ValueError("Fourier basis contains a zero-norm vector")
        bounds = math.sqrt(6.0 / basis_count) / norms
        if omega is not None:
            bounds = bounds / omega
        with torch.no_grad():
            samples = 2.0 * torch.rand_like(self.coefficients) - 1.0
            self.coefficients.copy_(samples * bounds.unsqueeze(0))
            bias_bound = 1.0 / math.sqrt(self.in_features)
            self.bias.uniform_(-bias_bound, bias_bound)

    @property
    def weight(self) -> torch.Tensor:
        return self.coefficients @ self.basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)

    def merged_linear(self) -> nn.Linear:
        """Return the equivalent ordinary linear layer for inference."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            device=self.bias.device,
            dtype=self.bias.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            linear.bias.copy_(self.bias)
        return linear


class SineLayer(nn.Module):
    """SIREN-style affine layer followed by a sine activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega: float,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not math.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be a finite, positive number")
        self.omega = float(omega)
        self.linear = nn.Linear(in_features, out_features)
        bound = (
            1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / self.omega
        )
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(inputs))


class ReparameterizedSineLayer(nn.Module):
    """Fourier-reparameterized affine map followed by sine."""

    def __init__(
        self,
        features: int,
        frequency_num: int,
        phase_num: int,
        omega: float,
    ) -> None:
        super().__init__()
        self.omega = float(omega)
        self.linear = FourierReparameterizedLinear(
            features,
            features,
            frequency_num,
            phase_num,
            omega=omega,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(inputs))


class FRINR(nn.Module):
    """MLP with optional positional encoding or Fourier reparameterization."""

    MODES: ClassVar[frozenset[str]] = frozenset(
        {"relu", "relu+fr", "relu+pe", "relu+pe+fr", "sin", "sin+fr"}
    )

    @classmethod
    def validate_mode(cls, mode: str) -> None:
        """Reject unsupported architecture combinations."""
        if mode not in cls.MODES:
            raise ValueError(f"Unsupported FR-INR mode: {mode!r}")

    def __init__(
        self,
        mode: str,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool,
        frequency_num: int,
        phase_num: int,
        first_omega_0: float,
        hidden_omega_0: float,
        mapping_input: int = 256,
    ) -> None:
        super().__init__()
        self.validate_mode(mode)
        if hidden_features <= 0 or hidden_layers < 0:
            raise ValueError(
                "hidden_features must be positive and hidden_layers non-negative"
            )

        self.mode = mode
        self.encoding: nn.Module
        if "+pe" in mode:
            self.encoding = FrequencyEncoding(
                in_features, mapping_input=mapping_input, use_nyquist=True
            )
            encoded_features = self.encoding.out_dim
        else:
            self.encoding = nn.Identity()
            encoded_features = in_features

        uses_sine = mode.startswith("sin")
        uses_fr = mode.endswith("+fr")
        layers: list[nn.Module] = []
        if uses_sine:
            layers.append(
                SineLayer(
                    encoded_features,
                    hidden_features,
                    omega=first_omega_0,
                    is_first=True,
                )
            )
        else:
            layers.extend((nn.Linear(encoded_features, hidden_features), nn.ReLU()))

        for _ in range(hidden_layers):
            if uses_fr and uses_sine:
                layers.append(
                    ReparameterizedSineLayer(
                        hidden_features,
                        frequency_num,
                        phase_num,
                        hidden_omega_0,
                    )
                )
            elif uses_fr:
                layers.extend(
                    (
                        FourierReparameterizedLinear(
                            hidden_features,
                            hidden_features,
                            frequency_num,
                            phase_num,
                        ),
                        nn.ReLU(),
                    )
                )
            elif uses_sine:
                layers.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        omega=hidden_omega_0,
                    )
                )
            else:
                layers.extend((nn.Linear(hidden_features, hidden_features), nn.ReLU()))

        if outermost_linear:
            output = nn.Linear(hidden_features, out_features)
            if uses_sine:
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                with torch.no_grad():
                    output.weight.uniform_(-bound, bound)
            layers.append(output)
        elif uses_sine:
            layers.append(
                SineLayer(hidden_features, out_features, omega=hidden_omega_0)
            )
        else:
            layers.extend((nn.Linear(hidden_features, out_features), nn.ReLU()))

        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(inputs))
