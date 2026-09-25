import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from implicit_neural_representations.experiment import open_experiment


class ExperimentRuntimeTests(unittest.TestCase):
    @staticmethod
    def _config(directory: str):
        run_directory = Path(directory) / "run"
        config = SimpleNamespace(
            seed=7,
            model_type="test-model",
            model_kwargs={"in_features": 2, "out_features": 1},
            learning_rate=1e-3,
            run_directory=str(run_directory),
            reconstruction_path=run_directory / "nested" / "reconstruction.png",
        )
        config.resolved_dict = MagicMock(return_value={"task": "image"})
        return config

    def test_shared_runtime_is_initialized_and_writer_is_closed(self):
        model = torch.nn.Linear(2, 1)
        writer = MagicMock()
        writer.__enter__.return_value = writer
        signal = SimpleNamespace(value_range=(-1.0, 1.0))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "implicit_neural_representations.experiment.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "implicit_neural_representations.experiment.build_model",
                return_value=model,
            ),
            patch(
                "implicit_neural_representations.experiment.SummaryWriter",
                return_value=writer,
            ),
            patch(
                "implicit_neural_representations.experiment.np.random.seed"
            ) as numpy_seed,
            patch(
                "implicit_neural_representations.experiment.torch.manual_seed"
            ) as torch_seed,
        ):
            config = self._config(directory)
            with open_experiment(config, signal) as runtime:
                self.assertIs(runtime.model, model)
                self.assertIs(runtime.writer, writer)
                self.assertEqual(runtime.device.type, "cpu")
                self.assertIsInstance(runtime.optimizer, torch.optim.Adam)
                self.assertTrue(runtime.reconstruction_path.parent.is_dir())

            numpy_seed.assert_called_once_with(7)
            torch_seed.assert_called_once_with(7)
            writer.add_text.assert_called_once()
            writer.__exit__.assert_called_once()

    def test_writer_is_closed_if_resolved_config_logging_fails(self):
        writer = MagicMock()
        writer.__enter__.return_value = writer
        writer.add_text.side_effect = RuntimeError("logging failed")
        signal = SimpleNamespace(value_range=(-1.0, 1.0))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "implicit_neural_representations.experiment.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "implicit_neural_representations.experiment.build_model",
                return_value=torch.nn.Linear(2, 1),
            ),
            patch(
                "implicit_neural_representations.experiment.SummaryWriter",
                return_value=writer,
            ),
            self.assertRaisesRegex(RuntimeError, "logging failed"),
        ):
            with open_experiment(self._config(directory), signal):
                self.fail("the context should not yield after logging fails")

        writer.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
