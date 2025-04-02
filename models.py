import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from encoding import positional_encoding


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable activation functions and optional positional encoding.

    This class implements an MLP that processes input features and outputs predictions.
    It can optionally apply positional encoding to the input, similar to the ReLUPE class.
    It consists of several linear layers interleaved with activation functions. The final
    layer outputs n_out values passed through a Sigmoid activation to constrain outputs to [0, 1].

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_layers (int, optional): Total number of layers. Default is 4.
        n_hidden_units (int, optional): Number of hidden units per layer. Default is 256.
        act (str, optional): Type of activation function to use. Options include:
            'relu', 'gaussian', 'quadratic', 'multi-quadratic', 'laplacian',
            'super-gaussian', and 'expsin'. Default is 'relu'.
        act_trainable (bool, optional): Whether the activation function's parameter(s)
            should be trainable. Default is False.
        use_pe (bool, optional): If True, apply positional encoding to the input. Default is False.
        L (int, optional): Number of positional encoding frequencies to use (only if use_pe is True). Default is 6.
        **kwargs: Additional keyword arguments for custom activation functions (e.g., a, b).
    """
    def __init__(self, 
                 n_in: int, 
                 n_out: int = 3, 
                 n_layers: int = 4, 
                 n_hidden_units: int = 256, 
                 act: str = 'relu', 
                 act_trainable: bool = False, 
                 use_pe: bool = False, 
                 L: int = 6,
                 **kwargs) -> None:
        super().__init__()
        self.use_pe = use_pe
        self.L = L
        # If positional encoding is active, modify the input dimension accordingly.
        effective_n_in = n_in * (1 + 2 * L) if use_pe else n_in

        layers = []
        for i in range(n_layers):
            # Define linear layer based on position in the network
            if i == 0:
                linear_layer = nn.Linear(effective_n_in, n_hidden_units)
            elif i < n_layers - 1:
                linear_layer = nn.Linear(n_hidden_units, n_hidden_units)
            
            # For all but the final layer, add an activation function
            if i < n_layers - 1:
                if act == 'relu':
                    activation = nn.ReLU(inplace=True)
                elif act == 'gaussian':
                    activation = GaussianActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
                elif act == 'quadratic':
                    activation = QuadraticActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
                elif act == 'multi-quadratic':
                    activation = MultiQuadraticActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
                elif act == 'laplacian':
                    activation = LaplacianActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
                elif act == 'super-gaussian':
                    activation = SuperGaussianActivation(a=kwargs.get('a', 1.0), 
                                                         b=kwargs.get('b', 1.0), 
                                                         trainable=act_trainable)
                elif act == 'expsin':
                    activation = ExpSinActivation(a=kwargs.get('a', 1.0), trainable=act_trainable)
                else:
                    raise ValueError(f"Unknown activation type: {act}")
                layers.extend([linear_layer, activation])
            else:
                # Final layer: output n_out values and apply a Sigmoid activation
                layers.extend([nn.Linear(n_hidden_units, n_out)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, n_in).

        Returns:
            torch.Tensor: Output tensor of shape (B, n_out).
        """
        if self.use_pe:
            # Apply positional encoding to the input
            x = positional_encoding(x, self.L).to(x.device)
        return self.net(x)


class GaussianActivation(nn.Module):
    """
    Gaussian Activation Function.

    Applies a Gaussian function to the input:
        f(x) = exp(-x^2 / (2 * a^2))
    where 'a' is a learnable parameter if act_trainable is True.
    """
    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), requires_grad=trainable))

    def forward(self, x):
        return torch.exp(-x**2 / (2 * self.a**2))


class QuadraticActivation(nn.Module):
    """
    Quadratic Activation Function.

    Applies a quadratic function to the input:
        f(x) = 1 / (1 + (a * x)^2)
    where 'a' is a learnable parameter if act_trainable is True.
    """
    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), requires_grad=trainable))

    def forward(self, x):
        return 1 / (1 + (self.a * x)**2)


class MultiQuadraticActivation(nn.Module):
    """
    Multi-Quadratic Activation Function.

    Applies a multi-quadratic function to the input:
        f(x) = 1 / sqrt(1 + (a * x)^2)
    where 'a' is a learnable parameter if act_trainable is True.
    """
    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), requires_grad=trainable))

    def forward(self, x):
        return 1 / (1 + (self.a * x)**2)**0.5


class LaplacianActivation(nn.Module):
    """
    Laplacian Activation Function.

    Applies a Laplacian function to the input:
        f(x) = exp(-|x| / a)
    where 'a' is a learnable parameter if act_trainable is True.
    """
    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), requires_grad=trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x) / self.a)


class SuperGaussianActivation(nn.Module):
    """
    Super Gaussian Activation Function.

    Applies a super Gaussian function to the input:
        f(x) = (exp(-x^2 / (2 * a^2)))^b
    where 'a' and 'b' are learnable parameters if act_trainable is True.
    """
    def __init__(self, a=1.0, b=1.0, trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), requires_grad=trainable))
        self.register_parameter('b', nn.Parameter(b * torch.ones(1), requires_grad=trainable))

    def forward(self, x):
        return torch.exp(-x**2 / (2 * self.a**2))**self.b


class ExpSinActivation(nn.Module):
    """
    Exponential Sine Activation Function.

    Applies an exponential sine function to the input:
        f(x) = exp(-sin(a * x))
    where 'a' is a learnable parameter if act_trainable is True.
    """
    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), requires_grad=trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a * x))


class SineLayer(nn.Module):
    """
    SineLayer: a fully-connected layer with sine activation.
    
    This is the original implementation from "Implicit Neural Representations with Periodic Activation Functions"
    by Sitzmann et al. (2020). See: https://github.com/vsitzmann/siren

    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    nonlinearity. Different signals may require different omega_0 in the first layer - this is a hyperparameter.
    
    If is_first=False, then the weights will be divided by omega_0 to keep the activation magnitude constant
    while boosting gradients to the weight matrix (see supplement Sec. 1.5).
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class SIREN(nn.Module):
    """
    SIREN model composed of multiple SineLayer modules and a final output layer.
    
    This is the original implementation from "Implicit Neural Representations with Periodic Activation Functions"
    by Sitzmann et al. (2020). See: https://github.com/vsitzmann/siren

    Args:
         in_features (int): Number of input features.
         hidden_features (int): Number of hidden features.
         hidden_layers (int): Number of hidden layers.
         out_features (int): Number of output features.
         outermost_linear (bool): If True, uses a final linear layer without a sine activation.
         first_omega_0 (float): Frequency factor for the first SineLayer.
         hidden_omega_0 (float): Frequency factor for subsequent SineLayers.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        """
        Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later.
        """
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join((str(layer.__class__), f"{activation_count}"))] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join((str(layer.__class__), f"{activation_count}"))] = x
            activation_count += 1

        return activations
    

class MFNBase(nn.Module):
    """
    MFNBase: Multiplicative filter network base class.
    
    This is the original implementation from "Multiplicative Filter Networks"
    by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter (2021). 
    See: https://github.com/boschresearch/multiplicative-filter-networks/

    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
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

        return

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
    
    This is the original implementation from "Multiplicative Filter Networks"
    by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter (2021). 
    See: https://github.com/boschresearch/multiplicative-filter-networks/
    """
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    """
    FourierNet: Network using FourierLayer filters.
    
    This is the original implementation from "Multiplicative Filter Networks"
    by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter (2021). 
    See: https://github.com/boschresearch/multiplicative-filter-networks/
    """
    def __init__(
        self,
        in_size,
        hidden_size,
        out_size,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )


class GaborLayer(nn.Module):
    """
    GaborLayer: Gabor-like filter as used in GaborNet.
    
    This is the original implementation from "Multiplicative Filter Networks"
    by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter (2021). 
    See: https://github.com/boschresearch/multiplicative-filter-networks/
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
        return

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


class GaborNet(MFNBase):
    """
    GaborNet: Network using GaborLayer filters.
    
    This is the original implementation from "Multiplicative Filter Networks"
    by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter (2021).
    See: https://github.com/boschresearch/multiplicative-filter-networks/
    """
    def __init__(
        self,
        in_size,
        hidden_size,
        out_size,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_size,
                    hidden_size,
                    input_scale / np.sqrt(n_layers + 1),
                    alpha / (n_layers + 1),
                    beta,
                )
                for _ in range(n_layers + 1)
            ]
        )


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
    def __init__(self, in_dim: int, out_dim: int, alpha: float, beta: float = 1.0, omega0: float = 5.0) -> None:
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
        
        ψ(u) = e̶x̶p̶(̶-̶u̶²̶/̶2̶) * cos(ω₀ * u) - exp(-ω₀²/2)
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
    

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
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