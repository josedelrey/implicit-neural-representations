import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


def init_weights(m, omega=1, c=1, is_first=False):
    """
    Initialize weights for a given module using the FINER++ scheme.

    This function implements a weight initialization strategy based on the 
    FINER++ paper:
    
        FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation
        Hao Zhu*, Zhen Liu*, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao
        Repository: https://github.com/liuzhen0212/FINERplusplus

    Args:
        m (nn.Module): The module whose weights will be initialized.
        omega (float): Frequency parameter used in the initialization.
        c (float): Constant scaling factor.
        is_first (bool): If True, use the first-layer initialization scheme.
    """
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:
            bound = 1 / fan_in  # SIREN-style initialization for first layer.
        else:
            bound = np.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)


def init_bias(m, k):
    """
    Initialize bias for a given module using a uniform distribution.

    Args:
        m (nn.Module): The module whose bias will be initialized.
        k (float): The range bound for uniform initialization.
    """
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)


def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    """
    Conditionally initialize weights for a linear layer according to the chosen method.

    This function applies the FINER++ weight initialization strategy if 'sine' is selected.
    See the FINER++ paper:
    
        FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation
        Hao Zhu*, Zhen Liu*, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao
        Repository: https://github.com/liuzhen0212/FINERplusplus

    Args:
        init_method (str): The method of initialization (e.g., 'sine').
        linear (nn.Linear): The linear layer to initialize.
        omega (float): Frequency parameter.
        c (float): Constant scaling factor.
        is_first (bool): Indicates if this is the first layer.
    """
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)    # SIREN-style initialization adapted for FINER++.
    # Default: PyTorch initialization is used if no specific method is provided.


def init_bias_cond(linear, fbs=None, is_first=True):
    """
    Conditionally initialize bias for a linear layer.

    If the layer is the first layer and a bias scaling factor is provided,
    this function applies uniform initialization to the bias.

    Args:
        linear (nn.Linear): The linear layer whose bias is to be initialized.
        fbs (float or None): The bias scaling factor. If None, no special initialization is performed.
        is_first (bool): True if this is the first layer.
    """
    if is_first and fbs is not None:
        init_bias(linear, fbs)


def generate_alpha(x, alphaType=None, alphaReqGrad=False):
    """
    Generate an adaptive scaling factor alpha from input x.

    This function computes a non-negative scaling value for the activation,
    following the FINER++ framework.
    
    FINER++ Paper Reference:
        FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation
        Hao Zhu*, Zhen Liu*, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao
        Repository: https://github.com/liuzhen0212/FINERplusplus

    Args:
        x (torch.Tensor): Input tensor.
        alphaType: (Optional) Type of alpha scaling function to use.
        alphaReqGrad (bool): Whether the generated alpha requires gradient computation.

    Returns:
        torch.Tensor: The computed scaling factor, guaranteed to be non-negative.
    """
    with torch.no_grad():
        return torch.abs(x) + 1


def finer_activation(x, omega=1, alphaType=None, alphaReqGrad=False):
    """
    Apply the FINER++ activation function.

    The activation is defined as:
        f(x) = sin(omega * alpha(x) * x)
    where alpha(x) is an adaptive scaling factor generated from x.
    
    FINER++ Paper Reference:
        FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation
        Hao Zhu*, Zhen Liu*, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao
        Repository: https://github.com/liuzhen0212/FINERplusplus

    Args:
        x (torch.Tensor): The input tensor.
        omega (float): Frequency scaling parameter.
        alphaType: (Optional) The type of alpha scaling to apply.
        alphaReqGrad (bool): Whether the computed alpha should require gradients.

    Returns:
        torch.Tensor: The result of applying the FINER++ activation.
    """
    return torch.sin(omega * generate_alpha(x, alphaType, alphaReqGrad) * x)


class FinerLayer(nn.Module):
    """
    A single FINER++ layer with a linear transformation followed by a variable-periodic activation.

    This layer implements the FINER++ activation:
        f(x) = sin(omega * alpha(x) * (Wx + b))
    and provides an option to extract intermediate values for debugging or visualization.
    
    FINER++ Paper Reference:
        FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation
        Hao Zhu*, Zhen Liu*, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao
        Repository: https://github.com/liuzhen0212/FINERplusplus

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        bias (bool): If True, include a bias term in the linear transformation.
        omega (float): Frequency scaling parameter for the activation.
        is_first (bool): If True, applies first-layer specific initialization.
        is_last (bool): If True, the layer omits the non-linear activation.
        init_method (str): The initialization method for weights (e.g., 'sine').
        init_gain (float): Gain parameter for weight initialization.
        fbs: Optional bias scaling factor for the first layer.
        hbs: Optional bias scaling factor for hidden layers.
        alphaType: (Optional) Type specifier for alpha scaling.
        alphaReqGrad (bool): Whether alpha should require gradient computation.
    """
    def __init__(self, in_features, out_features, bias=True, omega=30, 
                 is_first=False, is_last=False, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.omega = omega
        self.is_last = is_last  # If True, no activation is applied.
        self.alphaType = alphaType
        self.alphaReqGrad = alphaReqGrad
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize weights and bias according to the FINER++ scheme.
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return finer_activation(wx_b, self.omega)
        return wx_b  # For the last layer, no activation is applied.

    def forward_with_interm(self, input):
        """
        Forward pass with extraction of intermediate variables for analysis.

        Returns the linear output, the scaled linear output, and the activated output.
        
        Returns:
            tuple: (wx_b, wx_b_finer, sin_activated)
        """
        wx_b = self.linear(input) 
        if not self.is_last:
            alpha = generate_alpha(wx_b, self.alphaType, self.alphaReqGrad)
            return self.omega * wx_b, self.omega * alpha * wx_b, torch.sin(self.omega * alpha * wx_b)
        return wx_b  # For the last layer, return the linear output only.


class Finer(nn.Module):
    """
    FINER++ network: A family of variable-periodic functions for implicit neural representations.

    This class builds a multi-layer perceptron using FINER++ layers.
    It sequentially applies a FINER++ layer for the input, a number of hidden FINER++ layers,
    and a final linear layer without activation.
    
    FINER++ Paper Reference:
        FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation
        Hao Zhu*, Zhen Liu*, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao
        Repository: https://github.com/liuzhen0212/FINERplusplus

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        hidden_layers (int): Number of hidden FINER++ layers.
        hidden_features (int): Number of features in each hidden layer.
        first_omega (float): Frequency scaling for the first layer.
        hidden_omega (float): Frequency scaling for hidden layers.
        init_method (str): Initialization method for weights (default is 'sine').
        init_gain (float): Gain factor for weight initialization.
        fbs: Optional bias scaling factor for the first layer.
        hbs: Optional bias scaling factor for hidden layers.
        alphaType: (Optional) Type specifier for alpha scaling.
        alphaReqGrad (bool): Whether alpha should require gradient computation.
    """
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, 
                                   omega=hidden_omega, 
                                   init_method=init_method, init_gain=init_gain, hbs=hbs))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """
        Forward pass of the FINER++ network.

        Args:
            coords (torch.Tensor): Input tensor of shape (B, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features).
        """
        return self.net(coords)
    
    def forward_with_interm(self, input):
        """
        Forward pass that returns intermediate outputs for each layer.

        This is useful for debugging or visualizing the activations within the FINER++ network.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary containing intermediate outputs for each layer.
        """
        interm = {}
        N = len(self.net)
        for idx, layer in enumerate(self.net):
            if idx != N - 1:
                wxb, wxb_finer, sin_activated = layer.forward_with_interm(input)
                interm[f'layer_{idx}_wxb'] = wxb
                interm[f'layer_{idx}_wxb_finer'] = wxb_finer
                interm[f'layer_{idx}_sin_acted'] = sin_activated
                interm[f'layer_{idx}_out'] = sin_activated
                out = sin_activated
            else:
                out = layer(input)
                interm[f'layer_{idx}_out'] = out
            input = out
        return interm