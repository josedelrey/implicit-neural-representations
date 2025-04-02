import torch
import torch.nn as nn
from collections import OrderedDict

from modules.encoding import positional_encoding


class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        This is the implementation from "WIRE: Wavelet Implicit Representation"
        by Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, 
        Ashok Veeraraghavan and Richard G. Baraniuk (2023).
        See: https://github.com/vishwa91/wire/
        
        Args:
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
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
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity

        This is the implementation from "WIRE: Wavelet Implicit Representation"
        by Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, 
        Ashok Veeraraghavan and Richard G. Baraniuk (2023).
        See: https://github.com/vishwa91/wire/
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())


class WIRE(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 outermost_linear=True,
                 first_omega_0=30, 
                 hidden_omega_0=30., 
                 scale=10.0,
                 pos_encode=False, 
                 L=6):
        """
        Implicit Neural Representation with a complex Gabor nonlinearity.

        This is the implementation from "WIRE: Wavelet Implicit Representation"
        by Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, 
        Ashok Veeraraghavan and Richard G. Baraniuk (2023).
        See: https://github.com/vishwa91/wire/

        Args:
            in_features (int): Input feature dimension.
            hidden_features (int): Hidden layer feature dimension.
            hidden_layers (int): Number of hidden layers.
            out_features (int): Output feature dimension.
            outermost_linear (bool): If True, the final layer is a linear projection.
            first_omega_0 (float): Frequency scaling for the first layer.
            hidden_omega_0 (float): Frequency scaling for the hidden layers.
            scale (float): Scaling parameter for the Gaussian envelope.
            pos_encode (bool): If True, apply positional encoding to inputs.
            L (int): Number of frequencies for positional encoding.
            sidelength, fn_samples, use_nyquist: Legacy or optional parameters.
        """
        super().__init__()
        
        # Save positional encoding flag and frequency count if active.
        self.pos_encode = pos_encode
        if self.pos_encode:
            self.L = L

        # Use the ComplexGaborLayer as the default nonlinearity.
        self.nonlin = ComplexGaborLayer
        self.complex = True
        self.wavelet = 'gabor'
        
        # Build the network using an OrderedDict for clarity.
        layers = OrderedDict()
        # First layer with is_first=True and non-trainable frequency parameters.
        layers["layer0"] = self.nonlin(
            in_features, 
            hidden_features, 
            omega0=first_omega_0, 
            sigma0=scale, 
            is_first=True, 
            trainable=False
        )
        
        # Add the hidden layers.
        for i in range(hidden_layers):
            layers[f"layer{i+1}"] = self.nonlin(
                hidden_features, 
                hidden_features, 
                omega0=hidden_omega_0, 
                sigma0=scale
            )
        
        # Final layer: either a linear projection or an extra nonlinearity.
        if outermost_linear:
            layers["final"] = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        else:
            layers["final"] = self.nonlin(
                hidden_features, 
                out_features, 
                omega0=hidden_omega_0, 
                sigma0=scale
            )
            
        self.net = nn.Sequential(layers)
    
    def forward(self, coords):
        # If positional encoding is enabled, transform the input coordinates.
        if self.pos_encode:
            coords = positional_encoding(coords, self.L).to(coords.device)
        output = self.net(coords)
        # For consistency with the Gabor nonlinearity, return only the real part.
        if self.wavelet == 'gabor':
            return output.real
        return output
