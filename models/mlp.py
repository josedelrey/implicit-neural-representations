import torch
import torch.nn as nn

from modules.encoding import positional_encoding


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
                 in_features: int, 
                 out_features: int = 3, 
                 hidden_layers: int = 4, 
                 hidden_features: int = 256, 
                 act: str = 'relu', 
                 act_trainable: bool = False, 
                 use_pe: bool = False, 
                 L: int = 6,
                 **kwargs) -> None:
        super().__init__()
        self.use_pe = use_pe
        self.L = L
        # If positional encoding is active, modify the input dimension accordingly.
        effective_n_in = in_features * (1 + 2 * L) if use_pe else in_features

        layers = []
        for i in range(hidden_layers):
            # Define linear layer based on position in the network
            if i == 0:
                linear_layer = nn.Linear(effective_n_in, hidden_features)
            elif i < hidden_layers - 1:
                linear_layer = nn.Linear(hidden_features, hidden_features)
            
            # For all but the final layer, add an activation function
            if i < hidden_layers - 1:
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
                layers.extend([nn.Linear(hidden_features, out_features)])
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
