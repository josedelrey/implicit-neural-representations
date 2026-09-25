"""Configurable MLP baseline and activation functions."""

import torch
import torch.nn as nn

from ..encoding import PositionalEncoding


class MLP(nn.Module):
    """MLP with optional positional encoding and selectable hidden activations."""

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
        self.use_pe = use_pe
        self.L = L
        self.encoding = PositionalEncoding(in_features, L) if use_pe else nn.Identity()
        effective_n_in = self.encoding.out_dim if use_pe else in_features

        layers = []
        for i in range(hidden_layers):
            if i == 0:
                linear_layer = nn.Linear(effective_n_in, hidden_features)
            elif i < hidden_layers - 1:
                linear_layer = nn.Linear(hidden_features, hidden_features)

            if i < hidden_layers - 1:
                if act == 'relu':
                    activation = nn.ReLU(inplace=True)
                elif act == 'gaussian':
                    activation = GaussianActivation(
                        a=kwargs.get('a', 1.0), trainable=act_trainable
                    )
                elif act == 'quadratic':
                    activation = QuadraticActivation(
                        a=kwargs.get('a', 1.0), trainable=act_trainable
                    )
                elif act == 'multi-quadratic':
                    activation = MultiQuadraticActivation(
                        a=kwargs.get('a', 1.0), trainable=act_trainable
                    )
                elif act == 'laplacian':
                    activation = LaplacianActivation(
                        a=kwargs.get('a', 1.0), trainable=act_trainable
                    )
                elif act == 'super-gaussian':
                    activation = SuperGaussianActivation(
                        a=kwargs.get('a', 1.0),
                        b=kwargs.get('b', 1.0),
                        trainable=act_trainable,
                    )
                elif act == 'expsin':
                    activation = ExpSinActivation(
                        a=kwargs.get('a', 1.0), trainable=act_trainable
                    )
                else:
                    raise ValueError(f"Unknown activation type: {act}")
                layers.extend([linear_layer, activation])
            else:
                output_features = (
                    effective_n_in if hidden_layers == 1 else hidden_features
                )
                layers.extend([nn.Linear(output_features, out_features)])
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
