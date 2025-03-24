import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class SineLayer(nn.Module):
    """
    A fully-connected layer with sine activation.

    This layer multiplies the activations by a frequency factor (omega_0) before applying
    the sine nonlinearity. For the first layer (is_first=True), omega_0 is applied directly.
    For subsequent layers, the weights are scaled by 1/omega_0 to keep the activation
    magnitude constant while boosting gradients.

    Args:
         in_features (int): Number of input features.
         out_features (int): Number of output features.
         bias (bool): If True, includes a bias term in the linear layer.
         is_first (bool): If True, indicates that this is the first layer.
         omega_0 (float): Frequency factor for the sine activation.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 is_first: bool = False, 
                 omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """
    SIREN model composed of multiple SineLayer modules and a final output layer.

    Args:
         in_features (int): Number of input features.
         hidden_features (int): Number of hidden features.
         hidden_layers (int): Number of hidden layers.
         out_features (int): Number of output features.
         outermost_linear (bool): If True, uses a final linear layer without a sine activation.
         first_omega_0 (float): Frequency factor for the first SineLayer.
         hidden_omega_0 (float): Frequency factor for subsequent SineLayers.
    """
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int, 
                 hidden_layers: int, 
                 out_features: int, 
                 outermost_linear: bool = False, 
                 first_omega_0: float = 30.0, 
                 hidden_omega_0: float = 30.0) -> None:
        super().__init__()
        net_layers = []
        net_layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            net_layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
            net_layers.append(final_linear)
        else:
            net_layers.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*net_layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output


class GaborLayer(nn.Module):
    """
    Gabor layer applying a linear transformation followed by a Gabor-like activation.

    Args:
         in_features (int): Number of input features.
         out_features (int): Number of output features.
         weight_scale (float): Scaling factor for the weights.
         alpha (float): Parameter for the gamma distribution.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 weight_scale: float, 
                 alpha: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(1, out_features, in_features) - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, 1.0).sample((out_features,)))
        self.linear.weight.data *= weight_scale * self.gamma[:, None]**0.5
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D = torch.norm(rearrange(x, 'b d -> b 1 d') - self.mu, dim=-1)**2
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None])


class GaborNet(nn.Module):
    """
    Gabor network combining multiple Gabor layers and linear transformations.

    Args:
         in_size (int): Dimensionality of the input.
         hidden_size (int): Number of hidden features.
         out_size (int): Dimensionality of the output.
         n_layers (int): Number of layers (both Gabor and linear).
         input_scale (float): Scaling factor for the input in Gabor layers.
         alpha (float): Parameter for the gamma distribution in Gabor layers.
    """
    def __init__(self, 
                 in_features: int = 2,
                 out_features: int = 3,
                 hidden_features: int = 256,
                 hidden_layers: int = 4, 
                 input_scale: float = 256.0, 
                 alpha: float = 6.0) -> None:
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers)])
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )
        self.filters = nn.ModuleList([
            GaborLayer(
                in_features,
                hidden_features,
                input_scale / np.sqrt(hidden_layers + 1),
                alpha / (hidden_layers + 1)
            ) for _ in range(hidden_layers + 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        return out
