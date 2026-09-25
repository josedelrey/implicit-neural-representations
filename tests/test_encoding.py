import unittest

import torch

from implicit_neural_representations.architectures.frinr import FRINR
from implicit_neural_representations.architectures.mlp import MLP
from implicit_neural_representations.architectures.siren import Siren
from implicit_neural_representations.architectures.wire import WIRE
from implicit_neural_representations.encoding import (
    FrequencyEncoding,
    GaussianEncoding,
    PositionalEncoding,
)


class EncodingTests(unittest.TestCase):
    def test_positional_encoding_keeps_leading_dimensions_and_convention(self):
        encoding = PositionalEncoding(in_features=2, num_frequencies=2)
        coords = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]])
        result = encoding(coords)

        self.assertEqual(encoding.out_dim, 10)
        self.assertEqual(result.shape, (1, 2, 10))
        torch.testing.assert_close(result[0, 0, 2:4], torch.sin(coords[0, 0]))
        torch.testing.assert_close(result[0, 0, 6:8], torch.sin(2 * coords[0, 0]))

    def test_nerf_encoding_keeps_shape_and_pi_scaled_frequency(self):
        encoding = FrequencyEncoding(in_features=2, mapping_input=256, use_nyquist=True)
        coords = torch.tensor([[0.5, 0.0]])
        result = encoding(coords)

        self.assertEqual(encoding.num_frequencies, 6)
        self.assertEqual(encoding.out_dim, 26)
        self.assertEqual(result.shape, (1, 26))
        torch.testing.assert_close(result[0, 2], torch.tensor(1.0))
        torch.testing.assert_close(result[0, 5], torch.tensor(1.0))

        video_encoding = FrequencyEncoding(
            in_features=3, mapping_input=256, use_nyquist=True
        )
        self.assertEqual(video_encoding.num_frequencies, 10)
        self.assertEqual(video_encoding(torch.zeros(2, 4, 3)).shape, (2, 4, 63))
        self.assertEqual(
            FrequencyEncoding(2, mapping_input=[128, 256], use_nyquist=True).out_dim,
            22,
        )

    def test_gaussian_projection_is_a_buffer_and_survives_checkpoint_roundtrip(self):
        encoding = GaussianEncoding(in_features=2, mapping_input=4, scale_B=3.0)
        coords = torch.randn(2, 3, 2)
        result = encoding(coords)
        self.assertEqual(encoding.out_dim, 8)
        self.assertEqual(result.shape, (2, 3, 8))
        self.assertIn("B_gauss", encoding.state_dict())

        restored = GaussianEncoding(in_features=2, mapping_input=4, scale_B=3.0)
        restored.load_state_dict(encoding.state_dict())
        torch.testing.assert_close(restored(coords), result)

        encoding.to(dtype=torch.float64)
        self.assertEqual(encoding.B_gauss.dtype, torch.float64)
        self.assertEqual(encoding(coords.double()).dtype, torch.float64)
        if torch.cuda.is_available():
            encoding.cuda()
            self.assertEqual(encoding.B_gauss.device.type, "cuda")
            self.assertEqual(encoding(coords.double().cuda()).device.type, "cuda")

    def test_models_use_encoded_input_dimensions(self):
        mlp = MLP(
            2, out_features=1, hidden_layers=2, hidden_features=8, use_pe=True, L=2
        )
        wire = WIRE(
            2, hidden_features=8, hidden_layers=1, out_features=1, pos_encode=True, L=2
        )
        frinr = FRINR(
            mode="relu+pe",
            in_features=2,
            hidden_features=8,
            hidden_layers=1,
            out_features=1,
            outermost_linear=True,
            frequency_num=1,
            phase_num=1,
            first_omega_0=30,
            hidden_omega_0=30,
        )

        self.assertEqual(mlp.net[0].in_features, 10)
        self.assertEqual(wire.net.layer0.linear.in_features, 10)
        self.assertEqual(frinr.net[0].in_features, 26)
        coords = torch.randn(3, 2, requires_grad=True)
        for model in (mlp, wire, frinr):
            with self.subTest(model=type(model).__name__):
                predictions = model(coords)
                self.assertEqual(predictions.shape, (3, 1))
                predictions.sum().backward()
                self.assertIsNotNone(coords.grad)
                coords.grad = None

    def test_siren_leaves_coordinate_gradient_control_to_caller(self):
        model = Siren(2, hidden_features=8, hidden_layers=1, out_features=1)
        coords = torch.randn(3, 2, requires_grad=True)
        model(coords).sum().backward()
        self.assertIsNotNone(coords.grad)
        self.assertEqual(coords.grad.shape, coords.shape)
        self.assertIs(model.forward_with_activations(coords)["input"], coords)
        with torch.no_grad():
            self.assertFalse(model(coords).requires_grad)


if __name__ == "__main__":
    unittest.main()
