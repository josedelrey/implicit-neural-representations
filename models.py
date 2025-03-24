import torch
import torch.nn as nn
import numpy as np


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


class GaborFilter(nn.Module):
    """
    A Gabor filter module used for feature extraction.

    This layer applies a set of Gabor filters to the input tensor. The filters are 
    parameterized by a set of learned means (mu) and gamma values. The Gabor filter 
    uses a combination of sine and exponential terms to extract features from the 
    input. The learned gamma values control the width of the Gaussian in the filter.

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        alpha (float): A scaling factor for the gamma distribution.
        beta (float, optional): The rate parameter for the Gamma distribution, default is 1.0.
    """
    def __init__(self, in_dim: int, out_dim: int, alpha: float, beta: float = 1.0) -> None:
        super(GaborFilter, self).__init__()
        
        # Initialize the parameters for the Gabor filter
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = nn.Linear(in_dim, out_dim)

        # Initialize the weights and bias
        self.init_weights()
    
    def init_weights(self) -> None:
        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the squared Euclidean distance for the Gabor filter
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        
        # Apply the Gabor filter and return the result
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(x))


class GaborNet(nn.Module):
    """
    A network based on multiple Gabor filters for feature extraction and transformation.

    This model applies a series of Gabor filters and linear transformations to the 
    input tensor. The Gabor filters are used to extract features, while the linear 
    layers are used to transform the extracted features into the final output. 

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of features in the hidden layers.
        out_features (int): Number of output features.
        hidden_layers (int): Number of hidden layers in the network.
    """
    def __init__(self, 
                 in_features: int = 2, 
                 hidden_features: int = 256, 
                 out_features: int = 1, 
                 hidden_layers: int = 4) -> None:
        super(GaborNet, self).__init__()

        # Store the number of hidden layers
        self.hidden_layers = hidden_layers

        # Initialize the Gabor filters for each layer
        self.gabon_filters = nn.ModuleList([
            GaborFilter(in_features, hidden_features, alpha=6.0 / hidden_layers) for _ in range(hidden_layers)
        ])

        # Initialize the linear layers
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers - 1)] + [torch.nn.Linear(hidden_features, out_features)]
        )

        # Initialize weights for the linear layers
        for lin in self.linear[:hidden_layers - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial Gabor filter output
        zi = self.gabon_filters[0](x)  # Eq 3.a
        
        # Recursively apply Gabor filters and linear transformations
        for i in range(self.hidden_layers - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x)  # Eq 3.b

        # Final linear transformation
        return self.linear[self.hidden_layers - 1](zi)  # Eq 3.c
