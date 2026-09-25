import unittest

import torch

from implicit_neural_representations.architectures.finer import (
    Finer,
    FinerLayer,
    finer_activation,
)
from implicit_neural_representations.architectures.frinr import (
    FourierReparameterizedLinear,
    _fourier_basis,
)
from implicit_neural_representations.model_factory import (
    MODEL_CLASSES,
    build_model,
    validate_model_kwargs,
)
from implicit_neural_representations.presets import BASE_PRESETS, resolve_model_preset


class ModelPresetTests(unittest.TestCase):
    def test_every_model_resolves_for_image_and_video(self):
        self.assertEqual(set(BASE_PRESETS), set(MODEL_CLASSES))
        for task, dimensions in (("image", 2), ("video", 3)):
            for model_type in BASE_PRESETS:
                for channels in (1, 3):
                    with self.subTest(task=task, model=model_type, channels=channels):
                        preset = resolve_model_preset(model_type, task, channels)
                        self.assertEqual(preset.kwargs["in_features"], dimensions)
                        self.assertEqual(preset.kwargs["out_features"], channels)
                        self.assertGreater(preset.learning_rate, 0)

    def test_vector_wavelet_frequencies_match_the_task(self):
        image = resolve_model_preset("vectorwaveletnetnormalized", "image", 3)
        video = resolve_model_preset("vectorwaveletnetnormalized", "video", 3)
        self.assertEqual(image.kwargs["omega0"], [5.0, 5.0])
        self.assertEqual(video.kwargs["omega0"], [0.7, 5.0, 5.0])

        image.kwargs["omega0"].append(99)
        self.assertEqual(
            resolve_model_preset("vectorwaveletnetnormalized", "image", 3).kwargs[
                "omega0"
            ],
            [5.0, 5.0],
        )

    def test_invalid_frequency_vector_is_rejected_before_construction(self):
        preset = resolve_model_preset("vectorwaveletnetnormalized", "image", 3)
        preset.kwargs["omega0"] = [0.7, 5.0, 5.0]
        with self.assertRaisesRegex(ValueError, "omega0 length 3 != in_features 2"):
            validate_model_kwargs("vectorwaveletnetnormalized", preset.kwargs)

    def test_factory_uses_explicit_dimensions(self):
        model = build_model(
            "siren",
            in_features=2,
            out_features=1,
            hidden_features=8,
            hidden_layers=1,
        )
        self.assertEqual(model.net[0].linear.in_features, 2)
        self.assertEqual(model.net[-1].linear.out_features, 1)

    def test_single_layer_mlp_uses_input_dimension(self):
        for use_pe in (False, True):
            with self.subTest(use_pe=use_pe):
                model = build_model(
                    "mlp",
                    in_features=2,
                    out_features=3,
                    hidden_layers=1,
                    hidden_features=8,
                    use_pe=use_pe,
                    L=2,
                )
                self.assertEqual(model.net[0].in_features, 10 if use_pe else 2)
                self.assertEqual(model(torch.zeros(2, 2)).shape, (2, 3))

    def test_frinr_frequency_and_phase_counts_must_be_positive(self):
        for parameter in ("frequency_num", "phase_num"):
            with (
                self.subTest(parameter=parameter),
                self.assertRaisesRegex(ValueError, parameter),
            ):
                resolve_model_preset("frinr", "image", 3, overrides={parameter: 0})

    def test_frinr_rejects_unsupported_mode_combinations(self):
        with self.assertRaisesRegex(ValueError, "Unsupported FR-INR mode"):
            resolve_model_preset(
                "frinr", "image", 3, overrides={"mode": "sin+pe"}
            )
        with self.assertRaisesRegex(ValueError, "Unknown model override"):
            resolve_model_preset("frinr", "image", 3, overrides={"pe": True})

    def test_finer_initialization_overrides_preserve_defaults(self):
        torch.manual_seed(7)
        default = FinerLayer(4, 8, omega=30)
        torch.manual_seed(7)
        scaled = FinerLayer(4, 8, omega=30, init_gain=4)
        self.assertTrue(torch.allclose(scaled.linear.weight, 2 * default.linear.weight))

        model = Finer(
            in_features=2, out_features=3, hidden_layers=1, hidden_features=8, hbs=0
        )
        self.assertTrue(torch.count_nonzero(model.net[1].linear.bias) == 0)
        self.assertTrue(torch.count_nonzero(model.net[2].linear.bias) == 0)

    def test_finer_activation_matches_paper_equation_and_gradient(self):
        inputs = torch.tensor([-0.75, 0.4], dtype=torch.float64, requires_grad=True)
        omega = 2.5

        outputs = finer_activation(inputs, omega)
        outputs.sum().backward()

        phase = omega * (inputs.detach().abs() + 1.0) * inputs.detach()
        expected = torch.sin(phase)
        expected_gradient = (
            omega * torch.cos(phase) * (2.0 * inputs.detach().abs() + 1.0)
        )
        torch.testing.assert_close(outputs.detach(), expected)
        torch.testing.assert_close(inputs.grad, expected_gradient)

    def test_finer_rejects_unknown_initialization_method(self):
        with self.assertRaisesRegex(ValueError, "initialization method"):
            resolve_model_preset(
                "finer", "image", 3, overrides={"init_method": "unknown"}
            )

    def test_frinr_modes_preserve_output_shape(self):
        for mode in (
            "relu",
            "relu+fr",
            "relu+pe",
            "relu+pe+fr",
            "sin",
            "sin+fr",
        ):
            for outermost_linear in (False, True):
                with self.subTest(mode=mode, outermost_linear=outermost_linear):
                    preset = resolve_model_preset(
                        "frinr",
                        "image",
                        3,
                        overrides={
                            "mode": mode,
                            "hidden_layers": 1,
                            "hidden_features": 8,
                            "frequency_num": 2,
                            "phase_num": 2,
                            "mapping_input": 8,
                            "outermost_linear": outermost_linear,
                        },
                    )
                    model = build_model("frinr", **preset.kwargs)
                    self.assertEqual(model(torch.randn(2, 2)).shape, (2, 3))

    def test_fourier_reparameterization_is_fixed_and_mergeable(self):
        layer = FourierReparameterizedLinear(
            in_features=5,
            out_features=7,
            frequency_num=2,
            phase_num=3,
        )
        inputs = torch.randn(4, 5)

        self.assertIn("basis", dict(layer.named_buffers()))
        self.assertNotIn("basis", dict(layer.named_parameters()))
        torch.testing.assert_close(layer(inputs), layer.merged_linear()(inputs))

    def test_fourier_basis_and_initialization_match_paper_equations(self):
        basis = _fourier_basis(in_features=6, frequency_num=2, phase_num=3)
        positions = torch.linspace(-2 * torch.pi, 2 * torch.pi, 6)
        frequencies = torch.tensor([0.5, 1.0, 1.0, 2.0])
        phases = 2 * torch.pi * torch.arange(3) / 3
        expected = torch.cos(
            frequencies[:, None, None] * positions[None, None, :]
            + phases[None, :, None]
        ).reshape(-1, 6)
        torch.testing.assert_close(basis, expected)

        layer = FourierReparameterizedLinear(6, 4, 2, 3)
        bounds = (6.0 / layer.basis.shape[0]) ** 0.5 / layer.basis.norm(dim=1)
        self.assertTrue(torch.all(layer.coefficients.abs() <= bounds))

    def test_vector_wavelet_constructs_for_both_tasks(self):
        for task, dimensions in (("image", 2), ("video", 3)):
            with self.subTest(task=task):
                preset = resolve_model_preset("vectorwaveletnetnormalized", task, 3)
                preset.kwargs.update(hidden_features=8, hidden_layers=1)
                model = build_model("vectorwaveletnetnormalized", **preset.kwargs)
                self.assertEqual(model(torch.zeros(2, dimensions)).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
