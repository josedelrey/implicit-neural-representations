import math
import unittest

import torch
from torch import nn

from implicit_neural_representations.architectures.mfn import (
    GaborLayer,
    MFNBase,
    VectorWaveletLayer,
    WaveletLayer,
)
from implicit_neural_representations.architectures.siren import SineLayer, Siren
from implicit_neural_representations.architectures.wire import WIRE, ComplexGaborLayer
from implicit_neural_representations.encoding import (
    GaussianEncoding,
    PositionalEncoding,
)


class ArchitectureEquationTests(unittest.TestCase):
    def test_sine_layer_matches_activation_and_initialization_equations(self):
        layer = SineLayer(2, 2, omega_0=2.5).double()
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[0.4, -0.2], [0.1, 0.3]]))
            layer.linear.bias.copy_(torch.tensor([0.15, -0.25]))
        inputs = torch.tensor([[0.5, -0.75], [-0.2, 0.4]], dtype=torch.float64)

        affine = inputs @ layer.linear.weight.T + layer.linear.bias
        expected = torch.sin(2.5 * affine)
        outputs, intermediate = layer.forward_with_intermediate(inputs)

        torch.testing.assert_close(outputs, expected)
        torch.testing.assert_close(intermediate, 2.5 * affine)

        for is_first, omega_0 in ((True, 30.0), (False, 12.0)):
            with self.subTest(is_first=is_first):
                torch.manual_seed(4)
                initialized = SineLayer(8, 32, is_first=is_first, omega_0=omega_0)
                bound = (
                    1 / initialized.in_features
                    if is_first
                    else math.sqrt(6 / initialized.in_features) / omega_0
                )
                self.assertLessEqual(initialized.linear.weight.abs().max(), bound)

        network = Siren(
            in_features=2,
            hidden_features=8,
            hidden_layers=1,
            out_features=2,
            outermost_linear=True,
            hidden_omega_0=12.0,
        )
        output_bound = math.sqrt(6 / 8) / 12.0
        self.assertLessEqual(network.net[-1].weight.abs().max(), output_bound)

    def test_mfn_matches_multiplicative_recurrence(self):
        network = MFNBase(
            hidden_size=2,
            out_size=1,
            n_layers=2,
            weight_scale=1.0,
            bias=True,
        ).double()
        network.filters = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        with torch.no_grad():
            network.linear[0].weight.copy_(torch.tensor([[0.5, -0.25], [0.75, 0.2]]))
            network.linear[0].bias.copy_(torch.tensor([0.1, -0.3]))
            network.linear[1].weight.copy_(torch.tensor([[-0.4, 0.6], [0.3, 0.8]]))
            network.linear[1].bias.copy_(torch.tensor([0.2, 0.05]))
            network.output_linear.weight.copy_(torch.tensor([[0.7, -0.5]]))
            network.output_linear.bias.copy_(torch.tensor([0.12]))

        inputs = torch.tensor([[0.2, -0.4], [0.75, 0.1]], dtype=torch.float64)
        first = inputs
        second = inputs * (first @ network.linear[0].weight.T + network.linear[0].bias)
        third = inputs * (second @ network.linear[1].weight.T + network.linear[1].bias)
        expected = third @ network.output_linear.weight.T + network.output_linear.bias

        torch.testing.assert_close(network(inputs), expected)

    def test_gabor_and_wavelet_layers_match_carrier_envelope_equations(self):
        inputs = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
        weights = torch.tensor([[0.3, -0.2], [-0.5, 0.4]], dtype=torch.float64)
        bias = torch.tensor([0.1, -0.25], dtype=torch.float64)
        centers = torch.tensor([[0.0, -0.1], [0.5, 0.2]], dtype=torch.float64)
        precision = torch.tensor([1.5, 0.75], dtype=torch.float64)

        layers_and_scaled_inputs = (
            (GaborLayer(2, 2, 1.0), inputs),
            (WaveletLayer(2, 2, 1.0, omega0=2.5), 2.5 * inputs),
            (
                VectorWaveletLayer(2, 2, 1.0, omega0=[2.0, 3.0]),
                inputs * torch.tensor([2.0, 3.0]),
            ),
        )
        squared_distance = (inputs[:, None, :] - centers[None, :, :]).square().sum(-1)
        envelope = torch.exp(-0.5 * squared_distance * precision)

        for layer, scaled_inputs in layers_and_scaled_inputs:
            with self.subTest(layer=type(layer).__name__):
                layer.double()
                with torch.no_grad():
                    layer.linear.weight.copy_(weights)
                    layer.linear.bias.copy_(bias)
                    layer.mu.copy_(centers)
                    layer.gamma.copy_(precision)
                carrier = torch.sin(scaled_inputs @ weights.T + bias)
                torch.testing.assert_close(layer(inputs), carrier * envelope)

    def test_wire_matches_complex_gabor_equation_and_returns_real_output(self):
        layer = ComplexGaborLayer(2, 2, is_first=False, omega0=1.75, sigma0=0.6)
        inputs = torch.tensor([[0.2 + 0.1j, -0.4 + 0.3j], [0.5 - 0.2j, 0.1 + 0.4j]])
        weights = torch.tensor([[0.3 + 0.2j, -0.1 + 0.4j], [0.5 - 0.3j, 0.2 + 0.1j]])
        bias = torch.tensor([0.1 - 0.2j, -0.25 + 0.15j])
        with torch.no_grad():
            layer.linear.weight.copy_(weights)
            layer.linear.bias.copy_(bias)

        affine = inputs @ weights.T + bias
        expected = torch.exp(1j * 1.75 * affine - (0.6 * affine).abs().square())
        torch.testing.assert_close(layer(inputs), expected)

        network = WIRE(
            in_features=2,
            hidden_features=4,
            hidden_layers=1,
            out_features=2,
        )
        coordinates = torch.tensor([[0.2, -0.4], [0.5, 0.1]])
        complex_output = network.net(coordinates)
        self.assertTrue(complex_output.is_complex())
        self.assertFalse(network(coordinates).is_complex())
        torch.testing.assert_close(network(coordinates), complex_output.real)

    def test_coordinate_encodings_match_their_feature_equations(self):
        coordinates = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float64)
        positional = PositionalEncoding(in_features=2, num_frequencies=2)
        expected_positional = torch.cat(
            (
                coordinates,
                torch.sin(coordinates),
                torch.cos(coordinates),
                torch.sin(2 * coordinates),
                torch.cos(2 * coordinates),
            ),
            dim=-1,
        )
        torch.testing.assert_close(positional(coordinates), expected_positional)

        gaussian = GaussianEncoding(in_features=2, mapping_input=3, scale_B=1.0)
        gaussian.double()
        projection_matrix = torch.tensor(
            [[0.5, -0.25], [1.0, 0.75], [-0.4, 0.2]], dtype=torch.float64
        )
        with torch.no_grad():
            gaussian.B_gauss.copy_(projection_matrix)
        projection = 2 * math.pi * coordinates @ projection_matrix.T
        expected_gaussian = torch.cat(
            (torch.sin(projection), torch.cos(projection)), dim=-1
        )
        torch.testing.assert_close(gaussian(coordinates), expected_gaussian)


if __name__ == "__main__":
    unittest.main()
