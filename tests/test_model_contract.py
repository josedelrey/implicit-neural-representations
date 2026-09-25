import io
import unittest

import torch
from torch import nn

from implicit_neural_representations.model_factory import MODEL_CLASSES, build_model
from implicit_neural_representations.presets import BASE_PRESETS, resolve_model_preset

MFN_MODELS = {
    "fouriernet",
    "gabornet",
    "waveletnet",
    "waveletnetnormalized",
    "vectorwaveletnetnormalized",
}


def _small_model(model_type: str, task: str, channels: int, **overrides):
    preset = resolve_model_preset(
        model_type,
        task,
        channels,
        overrides={"hidden_features": 8, **overrides},
    )
    return build_model(model_type, **preset.kwargs), preset.kwargs


def _hidden_stage_count(model_type: str, model: nn.Module) -> int:
    if model_type in MFN_MODELS:
        return len(model.filters)
    if model_type == "mlp":
        return sum(isinstance(layer, nn.Linear) for layer in model.net) - 1
    return len(model.net) - 1


class ModelContractTests(unittest.TestCase):
    def test_every_preset_has_finite_gradients_and_checkpoint_roundtrip(self):
        self.assertEqual(set(MODEL_CLASSES), set(BASE_PRESETS))
        for task, in_features in (("image", 2), ("video", 3)):
            for channels in (1, 3):
                for model_type in sorted(MODEL_CLASSES):
                    with self.subTest(
                        task=task, channels=channels, model_type=model_type
                    ):
                        torch.manual_seed(7)
                        model, kwargs = _small_model(model_type, task, channels)
                        coordinates = torch.linspace(
                            -0.5, 0.5, steps=4 * in_features
                        ).reshape(4, in_features)
                        coordinates.requires_grad_()

                        predictions = model(coordinates)
                        self.assertEqual(predictions.shape, (4, channels))
                        self.assertFalse(predictions.is_complex())
                        self.assertTrue(torch.isfinite(predictions).all())

                        predictions.square().mean().backward()
                        self.assertIsNotNone(coordinates.grad)
                        self.assertTrue(torch.isfinite(coordinates.grad).all())
                        gradients = [
                            parameter.grad
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ]
                        self.assertTrue(gradients)
                        self.assertTrue(all(gradient is not None for gradient in gradients))
                        self.assertTrue(
                            all(torch.isfinite(gradient).all() for gradient in gradients)
                        )

                        checkpoint = io.BytesIO()
                        torch.save(model.state_dict(), checkpoint)
                        checkpoint.seek(0)
                        restored = build_model(model_type, **kwargs)
                        restored.load_state_dict(
                            torch.load(checkpoint, map_location="cpu", weights_only=True)
                        )
                        with torch.no_grad():
                            torch.testing.assert_close(
                                restored(coordinates.detach()), predictions.detach()
                            )

    def test_hidden_layers_count_additional_stages_after_first(self):
        for model_type in sorted(MODEL_CLASSES):
            default_depth = BASE_PRESETS[model_type].kwargs["hidden_layers"]
            for hidden_layers in (0, default_depth):
                with self.subTest(
                    model_type=model_type, hidden_layers=hidden_layers
                ):
                    model, _ = _small_model(
                        model_type,
                        "image",
                        1,
                        hidden_layers=hidden_layers,
                    )
                    self.assertEqual(
                        _hidden_stage_count(model_type, model), hidden_layers + 1
                    )
                    self.assertEqual(model(torch.zeros(2, 2)).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
