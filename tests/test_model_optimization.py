import unittest

import torch

from implicit_neural_representations.model_factory import MODEL_CLASSES, build_model
from implicit_neural_representations.presets import resolve_model_preset


class ModelOptimizationTests(unittest.TestCase):
    def test_every_model_fits_a_tiny_deterministic_signal(self):
        coordinates = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        target = 0.25 + 0.20 * coordinates[:, :1] - 0.15 * coordinates[:, 1:]

        for model_type in sorted(MODEL_CLASSES):
            with self.subTest(model_type=model_type):
                torch.manual_seed(123)
                preset = resolve_model_preset(
                    model_type,
                    "image",
                    channels=1,
                    overrides={"hidden_features": 16, "hidden_layers": 1},
                )
                model = build_model(model_type, **preset.kwargs)
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

                with torch.no_grad():
                    initial_mse = ((model(coordinates) - target) ** 2).mean().item()

                for _ in range(50):
                    optimizer.zero_grad()
                    loss = ((model(coordinates) - target) ** 2).mean()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    final_mse = ((model(coordinates) - target) ** 2).mean().item()

                self.assertLess(final_mse, initial_mse * 0.1)
                self.assertLess(final_mse, 0.01)


if __name__ == "__main__":
    unittest.main()
