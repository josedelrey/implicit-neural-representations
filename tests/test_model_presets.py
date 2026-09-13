import unittest

import torch

from models.model_factory import MODEL_CLASSES, build_model, validate_model_kwargs
from models.presets import BASE_PRESETS, resolve_model_preset


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
            resolve_model_preset("vectorwaveletnetnormalized", "image", 3).kwargs["omega0"],
            [5.0, 5.0],
        )

    def test_invalid_frequency_vector_is_rejected_before_construction(self):
        preset = resolve_model_preset("vectorwaveletnetnormalized", "image", 3)
        preset.kwargs["omega0"] = [0.7, 5.0, 5.0]
        with self.assertRaisesRegex(ValueError, "omega0 length 3 != in_features 2"):
            validate_model_kwargs("vectorwaveletnetnormalized", preset.kwargs)

    def test_factory_uses_explicit_dimensions(self):
        model = build_model(
            "siren", in_features=2, out_features=1,
            hidden_features=8, hidden_layers=1,
        )
        self.assertEqual(model.net[0].linear.in_features, 2)
        self.assertEqual(model.net[-1].linear.out_features, 1)

    def test_vector_wavelet_constructs_for_both_tasks(self):
        for task, dimensions in (("image", 2), ("video", 3)):
            with self.subTest(task=task):
                preset = resolve_model_preset("vectorwaveletnetnormalized", task, 3)
                preset.kwargs.update(hidden_features=8, hidden_layers=1)
                model = build_model("vectorwaveletnetnormalized", **preset.kwargs)
                self.assertEqual(model(torch.zeros(2, dimensions)).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
