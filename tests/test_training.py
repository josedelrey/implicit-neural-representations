import unittest
from unittest.mock import patch

import torch

from modules.training import fit, predict_chunks


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.batch_sizes = []
        self.grad_enabled = []
        self.input_devices = []

    def forward(self, coords):
        self.batch_sizes.append(coords.shape[0])
        self.grad_enabled.append(torch.is_grad_enabled())
        self.input_devices.append(coords.device.type)
        return coords[:, :1] * self.scale


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.model = RecordingModel()
        self.coords = torch.arange(5, dtype=torch.float32).unsqueeze(1)
        self.pixels = 2 * self.coords
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def test_full_sampling_runs_exact_number_of_updates(self):
        with patch('modules.training.log_training_metrics') as log, patch.object(
            self.optimizer, 'step', wraps=self.optimizer.step
        ) as step:
            fit(
                self.model, self.coords, self.pixels, self.optimizer,
                total_steps=3, log_interval=2, writer=None, sampling='full',
            )

        self.assertEqual(step.call_count, 3)
        self.assertEqual(self.model.batch_sizes, [5, 5, 5])
        self.assertEqual([call.args[0] for call in log.call_args_list], [2, 3])

    def test_random_sampling_uses_requested_batch_size(self):
        with patch('modules.training.log_training_metrics') as log, patch.object(
            self.optimizer, 'step', wraps=self.optimizer.step
        ) as step:
            fit(
                self.model, self.coords, self.pixels, self.optimizer,
                total_steps=4, log_interval=3, writer=None,
                sampling='random', batch_size=2,
            )

        self.assertEqual(step.call_count, 4)
        self.assertEqual(self.model.batch_sizes, [2, 2, 2, 2])
        self.assertEqual([call.args[0] for call in log.call_args_list], [3, 4])

    def test_prediction_preserves_order_and_evaluates_in_chunks(self):
        predictions = predict_chunks(self.model, self.coords, chunk_size=2)

        torch.testing.assert_close(predictions, self.coords)
        self.assertEqual(self.model.batch_sizes, [2, 2, 1])
        self.assertEqual(self.model.grad_enabled, [False, False, False])
        self.assertFalse(self.model.training)
        self.assertEqual(predictions.device.type, 'cpu')

    def test_invalid_sampling_configuration_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, 'positive batch_size'):
            fit(
                self.model, self.coords, self.pixels, self.optimizer,
                total_steps=1, log_interval=1, writer=None, sampling='random',
            )
        self.assertEqual(self.model.batch_sizes, [])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_random_sampling_and_prediction_transfer_only_batches(self):
        self.model.to('cuda')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        with patch('modules.training.log_training_metrics'):
            fit(
                self.model, self.coords, self.pixels, optimizer,
                total_steps=2, log_interval=1, writer=None,
                sampling='random', batch_size=2,
            )
        predictions = predict_chunks(self.model, self.coords, chunk_size=2)

        self.assertEqual(self.coords.device.type, 'cpu')
        self.assertEqual(self.pixels.device.type, 'cpu')
        self.assertEqual(self.model.batch_sizes, [2, 2, 2, 2, 1])
        self.assertEqual(self.model.input_devices, ['cuda'] * 5)
        self.assertEqual(predictions.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
