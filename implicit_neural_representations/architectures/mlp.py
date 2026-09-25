"""Configurable MLP baseline and activation functions."""

import torch
from torch import nn

from ..encoding import PositionalEncoding


class MLP(nn.Module):
    """MLP with one initial and ``hidden_layers`` additional hidden layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        hidden_layers: int = 4,
        hidden_features: int = 256,
        act: str = 'relu',
        act_trainable: bool = False,
        use_pe: bool = False,
        L: int = 6,
        **kwargs,
    ) -> None:
        super().__init__()
        if hidden_layers < 0 or hidden_features <= 0:
            raise ValueError(
                "hidden_layers must be non-negative and hidden_features positive"
            )
        self.use_pe = use_pe
        self.L = L
        self.encoding = PositionalEncoding(in_features, L) if use_pe else nn.Identity()
        effective_n_in = self.encoding.out_dim if use_pe else in_features

        def make_activation() -> nn.Module:
            if act == 'relu':
                return nn.ReLU(inplace=True)
            if act == 'gaussian':
                return GaussianActivation(
                    a=kwargs.get('a', 1.0), trainable=act_trainable
                )
            if act == 'quadratic':
                return QuadraticActivation(
                    a=kwargs.get('a', 1.0), trainable=act_trainable
                )
            if act == 'multi-quadratic':
                return MultiQuadraticActivation(
                    a=kwargs.get('a', 1.0), trainable=act_trainable
                )
            if act == 'laplacian':
                return LaplacianActivation(
                    a=kwargs.get('a', 1.0), trainable=act_trainable
                )
            if act == 'super-gaussian':
                return SuperGaussianActivation(
                    a=kwargs.get('a', 1.0),
                    b=kwargs.get('b', 1.0),
                    trainable=act_trainable,
                )
            if act == 'expsin':
                return ExpSinActivation(
                    a=kwargs.get('a', 1.0), trainable=act_trainable
                )
            raise ValueError(f"Unknown activation type: {act}")

        layers = [nn.Linear(effective_n_in, hidden_features), make_activation()]
        for _ in range(hidden_layers):
            layers.extend(
                [nn.Linear(hidden_features, hidden_features), make_activation()]
            )
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions for the input coordinates."""
        x = self.encoding(x)
        return self.net(x)


class GaussianActivation(nn.Module):
    """Gaussian activation with an optional trainable scale."""

    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter(
            'a', nn.Parameter(a * torch.ones(1), requires_grad=trainable)
        )

    def forward(self, x):
        return torch.exp(-(x**2) / (2 * self.a**2))


class QuadraticActivation(nn.Module):
    """Quadratic activation with an optional trainable scale."""

    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter(
            'a', nn.Parameter(a * torch.ones(1), requires_grad=trainable)
        )

    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2)


class MultiQuadraticActivation(nn.Module):
    """Multiquadratic activation with an optional trainable scale."""

    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter(
            'a', nn.Parameter(a * torch.ones(1), requires_grad=trainable)
        )

    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2) ** 0.5


class LaplacianActivation(nn.Module):
    """Laplacian activation with an optional trainable scale."""

    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter(
            'a', nn.Parameter(a * torch.ones(1), requires_grad=trainable)
        )

    def forward(self, x):
        return torch.exp(-torch.abs(x) / self.a)


class SuperGaussianActivation(nn.Module):
    """Super-Gaussian activation with optional trainable parameters."""

    def __init__(self, a=1.0, b=1.0, trainable=True):
        super().__init__()
        self.register_parameter(
            'a', nn.Parameter(a * torch.ones(1), requires_grad=trainable)
        )
        self.register_parameter(
            'b', nn.Parameter(b * torch.ones(1), requires_grad=trainable)
        )

    def forward(self, x):
        return torch.exp(-(x**2) / (2 * self.a**2)) ** self.b


class ExpSinActivation(nn.Module):
    """Exponential-sine activation with an optional trainable scale."""

    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter(
            'a', nn.Parameter(a * torch.ones(1), requires_grad=trainable)
        )

    def forward(self, x):
        return torch.exp(-torch.sin(self.a * x))
