import tempfile
import unittest
from pathlib import Path

import yaml

from implicit_neural_representations.config import ConfigError, load_experiment_config


class ExperimentConfigTests(unittest.TestCase):
    def _load_document(self, document: dict, task: str = "image"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            return load_experiment_config(str(path), task)

    @staticmethod
    def _image_document():
        return {
            "data": {"path": "source.png", "sidelength": 4, "is_rgb": True},
            "model": {"name": "waveletnetnormalized", "overrides": {}},
            "training": {"total_steps": 1, "log_interval": 1},
            "output": {
                "directory": "run",
                "reconstruction": "result.png",
                "chunk_size": 8,
            },
        }

    def test_model_and_learning_rate_can_be_overridden_per_run(self):
        document = self._image_document()
        document["model"]["overrides"] = {"hidden_features": 384, "omega0": 4.0}
        document["training"]["learning_rate"] = 0.0005
        config = self._load_document(document)

        self.assertEqual(config.model_kwargs["hidden_features"], 384)
        self.assertEqual(config.model_kwargs["omega0"], 4.0)
        self.assertEqual(config.learning_rate, 0.0005)
        resolved = config.resolved_dict(device="cpu", value_range=(-1.0, 1.0))
        self.assertEqual(resolved["model"]["kwargs"]["hidden_features"], 384)
        self.assertEqual(resolved["training"]["learning_rate"], 0.0005)
        self.assertEqual(resolved["data"]["value_range"], (-1.0, 1.0))

    def test_scientific_notation_learning_rate_is_a_number(self):
        document = self._image_document()
        document["training"]["learning_rate"] = 0.0002
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            contents = yaml.safe_dump(document)
            path.write_text(
                contents.replace("learning_rate: 0.0002", "learning_rate: 2e-4"),
                encoding="utf-8",
            )
            config = load_experiment_config(str(path), "image")
        self.assertEqual(config.learning_rate, 0.0002)

    def test_finer_bias_scales_accept_zero_null_and_positive_values(self):
        for task in ("image", "video"):
            for key in ("fbs", "hbs"):
                for value in (0, 0.0, None, 0.5):
                    with self.subTest(task=task, key=key, value=value):
                        document = self._image_document()
                        document["model"] = {"name": "finer", "overrides": {key: value}}
                        if task == "video":
                            document["training"]["batch_size"] = 8
                        config = self._load_document(document, task)
                        self.assertEqual(config.model_kwargs[key], value)

    def test_finer_bias_scales_reject_invalid_values(self):
        for key in ("fbs", "hbs"):
            for value in (
                -1,
                float("nan"),
                float("inf"),
                -float("inf"),
                True,
                "0",
                [],
                {},
            ):
                with self.subTest(key=key, value=value):
                    document = self._image_document()
                    document["model"] = {"name": "finer", "overrides": {key: value}}
                    with self.assertRaisesRegex(ConfigError, f"model.overrides.{key}"):
                        self._load_document(document)

    def test_unknown_and_misspelled_fields_are_rejected(self):
        document = self._image_document()
        document["training"]["totl_steps"] = document["training"].pop("total_steps")
        with self.assertRaisesRegex(ConfigError, "missing required keys: total_steps"):
            self._load_document(document)

        document = self._image_document()
        document["model"]["overrides"] = {"hidden_featuers": 384}
        with self.assertRaisesRegex(ConfigError, "Unknown model override"):
            self._load_document(document)

    def test_wrong_types_and_task_specific_fields_are_rejected(self):
        document = self._image_document()
        document["data"]["is_rgb"] = "false"
        with self.assertRaisesRegex(ConfigError, "data.is_rgb must be a boolean"):
            self._load_document(document)

        document = self._image_document()
        document["training"]["batch_size"] = 8
        with self.assertRaisesRegex(ConfigError, "unknown keys: batch_size"):
            self._load_document(document)

        document = self._image_document()
        document["training"]["learning_rate"] = -0.1
        with self.assertRaisesRegex(ConfigError, "positive number"):
            self._load_document(document)

        document = self._image_document()
        document["model"]["overrides"] = {"hidden_features": "384"}
        with self.assertRaisesRegex(ConfigError, "model.overrides.hidden_features"):
            self._load_document(document)

        document = self._image_document()
        document["model"]["overrides"] = {"in_features": 3}
        with self.assertRaisesRegex(ConfigError, "Unknown model override"):
            self._load_document(document)

        document = self._image_document()
        document["output"]["reconstruction"] = "../outside.png"
        with self.assertRaisesRegex(ConfigError, "contained in output.directory"):
            self._load_document(document)

    def test_invalid_model_dimensions_are_rejected_before_model_construction(self):
        document = self._image_document()
        document["model"]["name"] = "vectorwaveletnetnormalized"
        document["model"]["overrides"] = {"omega0": [0.7, 5.0, 5.0]}
        with self.assertRaisesRegex(ConfigError, "omega0 length 3 != in_features 2"):
            self._load_document(document)

    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("data: {}\ndata: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "Duplicate YAML key"):
                load_experiment_config(str(path), "image")


if __name__ == "__main__":
    unittest.main()
