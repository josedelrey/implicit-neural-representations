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
        beta (float, optional): The rate parameter for the Gamma distribution.
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
        self.gabor_filters = nn.ModuleList([
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
        zi = self.gabor_filters[0](x)  # Eq 3.a
        
        # Recursively apply Gabor filters and linear transformations
        for i in range(self.hidden_layers - 1):
            zi = self.linear[i](zi) * self.gabor_filters[i + 1](x)  # Eq 3.b

        # Final linear transformation
        return self.linear[self.hidden_layers - 1](zi)  # Eq 3.c


class WaveletFilter(nn.Module):
    """
    A Morlet wavelet filter module used for feature extraction.

    This layer applies a set of learnable Morlet wavelet filters to the input tensor.
    The filters are parameterized by learned means (mu) and gamma values, similar to
    the Gabor filter. Instead of using a sine nonlinearity, it uses a Morlet wavelet
    function:
    
        ψ(u) = exp(-u²/2)*cos(ω₀ * u) - exp(-ω₀²/2)
    
    where ω₀ is a learnable frequency parameter.
    
    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        alpha (float): A scaling factor for the gamma distribution.
        beta (float, optional): The rate parameter for the Gamma distribution.
        omega0 (float): Initial frequency parameter for the Morlet wavelet.
    """
    def __init__(self, in_dim: int, out_dim: int, alpha: float, beta: float = 1.0, omega0: float = 30.0) -> None:
        super(WaveletFilter, self).__init__()
        # Learned centers for each filter
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        # Learned gamma values controlling the Gaussian envelope width
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,)))
        # Linear projection to generate the argument for the wavelet nonlinearity
        self.linear = nn.Linear(in_dim, out_dim)
        # Learnable frequency parameter for the Morlet wavelet
        self.omega0 = nn.Parameter(torch.tensor(omega0))
        self.init_weights()
    
    def init_weights(self) -> None:
        # Scale the weights based on gamma (similar to GaborFilter)
        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)
    
    def morlet_wavelet(self, u: torch.Tensor) -> torch.Tensor:
        """
        Applies the Morlet wavelet nonlinearity to the input u.
        
        ψ(u) = exp(-u²/2) * cos(ω₀ * u) - exp(-ω₀²/2)
        """
        return torch.cos(self.omega0 * u) - torch.exp(-0.5 * (self.omega0**2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute squared Euclidean distance between x and each filter's center
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        # Gaussian envelope based on the learned gamma values
        envelope = torch.exp(- self.gamma.unsqueeze(0) / 2. * norm)
        # Linear projection of x
        lin_out = self.linear(x)
        # Apply the Morlet wavelet nonlinearity
        wavelet_response = self.morlet_wavelet(lin_out)
        # Return the modulated response
        return envelope * wavelet_response

class WaveletNet(nn.Module):
    """
    A network based on multiple Morlet wavelet filters for feature extraction and transformation.

    This model mirrors the structure of GaborNet but replaces the Gabor filters with learnable
    Morlet wavelet filters. For an input coordinate (e.g., (x, y)), the network first extracts
    a high-dimensional feature vector using a MorletWaveletFilter. In subsequent layers, it applies
    a linear transformation whose output is element-wise multiplied by a fresh Morlet filter response
    (computed from the same coordinate). Finally, a linear layer decodes the high-dimensional features
    into the pixel value.
    
    Args:
        in_features (int): Number of input features (e.g., 2 for (x,y) coordinates).
        hidden_features (int): Number of features in the hidden layers.
        out_features (int): Number of output features (e.g., 1 for grayscale).
        hidden_layers (int): Number of hidden layers in the network.
    """
    def __init__(self, 
                 in_features: int = 2, 
                 hidden_features: int = 256, 
                 out_features: int = 1, 
                 hidden_layers: int = 4,
                 omega0: float = 5.0) -> None:
        super(WaveletNet, self).__init__()
        self.hidden_layers = hidden_layers

        # Initialize a list of MorletWaveletFilter modules (one per layer)
        self.morlet_filters = nn.ModuleList([
            WaveletFilter(in_features, hidden_features, alpha=6.0 / hidden_layers, omega0=omega0)
            for _ in range(hidden_layers)
        ])

        # Initialize the linear layers.
        # For hidden_layers - 1 layers, we have a linear mapping from hidden_features to hidden_features,
        # and the final layer maps from hidden_features to out_features.
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers - 1)] +
            [nn.Linear(hidden_features, out_features)]
        )

        # Initialize weights for the linear layers for hidden layers
        for lin in self.linear[:hidden_layers - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First wavelet filter: get initial high-dimensional feature vector from the input coordinate
        z = self.morlet_filters[0](x)
        # Recursively apply linear layers and modulate with subsequent Morlet filter responses
        for i in range(self.hidden_layers - 1):
            z = self.linear[i](z) * self.morlet_filters[i + 1](x)
        # Final linear transformation to decode into the pixel value
        return self.linear[self.hidden_layers - 1](z)
