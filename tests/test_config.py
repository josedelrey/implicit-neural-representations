import tempfile
import unittest
from pathlib import Path

import yaml

from implicit_neural_representations.config import ConfigError, load_experiment_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'


class ExperimentConfigTests(unittest.TestCase):
    def _load_document(self, document: dict, task: str = 'image'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            path.write_text(yaml.safe_dump(document), encoding='utf-8')
            return load_experiment_config(str(path), task)

    def _image_document(self):
        return yaml.safe_load((CONFIG_DIR / 'image.yaml').read_text(encoding='utf-8'))

    def test_example_configs_resolve_all_settings(self):
        image = load_experiment_config(str(CONFIG_DIR / 'image.yaml'), 'image')
        video = load_experiment_config(str(CONFIG_DIR / 'video.yaml'), 'video')

        self.assertEqual(image.model_type, 'waveletnetnormalized')
        self.assertEqual(image.model_kwargs['in_features'], 2)
        self.assertEqual(image.model_kwargs['out_features'], 3)
        self.assertEqual(image.learning_rate, 1e-3)
        self.assertEqual(image.export_path, 'outputs/test_reconstructed.png')
        self.assertIsNone(image.batch_size)
        self.assertEqual(video.model_kwargs['in_features'], 3)
        self.assertEqual(video.model_kwargs['omega0'], [0.7, 5.0, 5.0])
        self.assertEqual(video.batch_size, 32768)
        self.assertEqual(video.export_path, 'outputs/akiyo_reconstructed.mp4')

    def test_model_and_learning_rate_can_be_overridden_per_run(self):
        document = self._image_document()
        document['model']['overrides'] = {'hidden_features': 384, 'omega0': 4.0}
        document['training']['learning_rate'] = 0.0005
        config = self._load_document(document)

        self.assertEqual(config.model_kwargs['hidden_features'], 384)
        self.assertEqual(config.model_kwargs['omega0'], 4.0)
        self.assertEqual(config.learning_rate, 0.0005)
        resolved = config.resolved_dict(device='cpu', value_range=(-1.0, 1.0))
        self.assertEqual(resolved['model']['kwargs']['hidden_features'], 384)
        self.assertEqual(resolved['training']['learning_rate'], 0.0005)
        self.assertEqual(resolved['data']['value_range'], (-1.0, 1.0))
        self.assertEqual(resolved['working_directory'], str(Path.cwd()))

    def test_scientific_notation_learning_rate_is_a_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            contents = (CONFIG_DIR / 'image.yaml').read_text(encoding='utf-8')
            path.write_text(
                contents.replace('  # learning_rate: 0.001', '  learning_rate: 1e-3'),
                encoding='utf-8',
            )
            config = load_experiment_config(str(path), 'image')
        self.assertEqual(config.learning_rate, 0.001)

    def test_unknown_and_misspelled_fields_are_rejected(self):
        document = self._image_document()
        document['training']['totl_steps'] = document['training'].pop('total_steps')
        with self.assertRaisesRegex(ConfigError, 'missing required keys: total_steps'):
            self._load_document(document)

        document = self._image_document()
        document['model']['overrides'] = {'hidden_featuers': 384}
        with self.assertRaisesRegex(ConfigError, 'Unknown model override'):
            self._load_document(document)

    def test_wrong_types_and_task_specific_fields_are_rejected(self):
        document = self._image_document()
        document['data']['is_rgb'] = 'false'
        with self.assertRaisesRegex(ConfigError, 'data.is_rgb must be a boolean'):
            self._load_document(document)

        document = self._image_document()
        document['training']['batch_size'] = 8
        with self.assertRaisesRegex(ConfigError, 'unknown keys: batch_size'):
            self._load_document(document)

        document = self._image_document()
        document['training']['learning_rate'] = -0.1
        with self.assertRaisesRegex(ConfigError, 'positive number'):
            self._load_document(document)

        document = self._image_document()
        document['model']['overrides'] = {'hidden_features': '384'}
        with self.assertRaisesRegex(ConfigError, 'model.overrides.hidden_features'):
            self._load_document(document)

        document = self._image_document()
        document['model']['overrides'] = {'in_features': 3}
        with self.assertRaisesRegex(ConfigError, 'Unknown model override'):
            self._load_document(document)

    def test_invalid_model_dimensions_are_rejected_before_model_construction(self):
        document = self._image_document()
        document['model']['name'] = 'vectorwaveletnetnormalized'
        document['model']['overrides'] = {'omega0': [0.7, 5.0, 5.0]}
        with self.assertRaisesRegex(ConfigError, 'omega0 length 3 != in_features 2'):
            self._load_document(document)

    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.yaml'
            path.write_text('data: {}\ndata: {}\n', encoding='utf-8')
            with self.assertRaisesRegex(ConfigError, 'Duplicate YAML key'):
                load_experiment_config(str(path), 'image')


if __name__ == '__main__':
    unittest.main()
