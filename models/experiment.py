import torch
import torch.nn as nn
import numpy as np


class MFNBase(nn.Module):
    """
    MFNBase: Multiplicative filter network base class.
    """
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

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out


class FourierLayer(nn.Module):
    """
    FourierLayer: Sine filter as used in FourierNet.
    """
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    """
    FourierNet: Network using FourierLayer filters.
    """
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
    """
    GaborLayer: Gabor-like filter as used in GaborNet.
    """
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


class WaveletLayer(nn.Module):
    """
    WaveletLayer: A wavelet filter layer analogous to FourierLayer and GaborLayer.
    
    It applies a linear projection followed by a Morlet wavelet nonlinearity,
    modulated by a Gaussian envelope based on the distance between the input and a
    learned center (mu). The envelope uses a learned gamma parameter.
    """
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, omega0=5.0):
        super().__init__()
        # Linear projection of the input
        self.linear = nn.Linear(in_features, out_features)
        # Learned center for each filter
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        # Learned gamma values controlling the envelope width
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,)))
        # Scale the linear weights by weight_scale and sqrt(gamma)
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        # Learnable frequency parameter for the Morlet wavelet
        self.omega0 = omega0
    
    def morlet_wavelet(self, u):
        """
        Applies the Morlet wavelet nonlinearity:
            ψ(u) = cos(ω₀ * u) - exp(-0.5 * ω₀²)
        """
        omega0_tensor = torch.tensor(self.omega0, dtype=u.dtype, device=u.device)
        constant = torch.exp(torch.tensor(-0.5 * (self.omega0 ** 2), dtype=u.dtype, device=u.device))
        return torch.cos(omega0_tensor * u) - constant
    
    def forward(self, x):
        # Compute squared Euclidean distance between x and the learned center μ.
        # x shape: (batch, in_features), mu shape: (out_features, in_features)
        D = (x ** 2).sum(dim=1, keepdim=True) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        # Compute the Gaussian envelope using the learned gamma
        envelope = torch.exp(-0.5 * D * self.gamma.unsqueeze(0))
        # Linear projection followed by the Morlet wavelet nonlinearity
        lin_out = self.linear(x)
        wavelet_response = self.morlet_wavelet(lin_out)
        return wavelet_response * envelope
    

class Experiment(MFNBase):
    """
    MFNWaveletNet: A multiplicative filter network using wavelet filters.
    
    This network follows the same architectural design as FourierNet and GaborNet,
    where a ModuleList of filters is applied multiplicatively with intermediate linear
    transformations. Each filter is a WaveletLayer.
    
    Args:
        in_size (int): Dimensionality of input features.
        hidden_size (int): Dimensionality of the hidden feature space.
        out_size (int): Dimensionality of the output.
        n_layers (int): Number of hidden layers (filters) to use.
        input_scale (float): Scale factor for the filter initialization.
        weight_scale (float): Scale factor for linear weight initialization.
        alpha (float): Parameter for the Gamma distribution (controls envelope width).
        beta (float): Rate parameter for the Gamma distribution.
        omega0 (float): Frequency parameter for the Morlet wavelet.
        bias (bool): Whether to use bias in the linear layers.
        output_act (bool): Whether to apply sine activation to the output.
    """
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
        super().__init__(hidden_features, out_features, hidden_layers, weight_scale, bias, output_act)
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
        # Create normalization layers for each branch before multiplication:
        # One for each filter output (n_layers + 1) and one for each linear branch (n_layers).
        self.filter_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for _ in range(hidden_layers + 1)])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for _ in range(hidden_layers)])
        
    def forward(self, x):
        # Apply the first filter and normalize its output.
        out = self.filter_norms[0](self.filters[0](x))
        # For subsequent filters, normalize both the filter and linear branch outputs before multiplying.
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
