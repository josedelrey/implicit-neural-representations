import json
import math
import tempfile
import unittest
from pathlib import Path

from implicit_neural_representations.artifacts import initialize_run, save_metrics


class ArtifactTests(unittest.TestCase):
    def test_run_directory_is_created_once_with_resolved_config(self):
        resolved = {'task': 'image'}
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / 'run'
            initialize_run(str(run_directory), resolved)
            saved = json.loads(
                (run_directory / 'resolved_config.json').read_text(encoding='utf-8')
            )
            with self.assertRaisesRegex(FileExistsError, 'choose a new directory'):
                initialize_run(str(run_directory), resolved)

        self.assertEqual(saved, resolved)

    def test_metrics_are_standards_compliant_for_perfect_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            save_metrics(run_directory, {'mse': 0.0, 'psnr': math.inf})
            text = (run_directory / 'metrics.json').read_text(encoding='utf-8')

        self.assertNotIn(': Infinity', text)
        self.assertEqual(json.loads(text), {'mse': 0.0, 'psnr': 'Infinity'})

    def test_nan_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'must not contain NaN'):
                save_metrics(Path(directory), {'mse': math.nan})


if __name__ == '__main__':
    unittest.main()
