import datetime
import math
import unittest
from unittest.mock import Mock, patch

import torch

from implicit_neural_representations.loss import mse_to_psnr
from implicit_neural_representations.utils import log_training_metrics


class SignalRangeMetricTests(unittest.TestCase):
    def test_psnr_matches_equivalent_unit_range_signal(self):
        normalized_mse = 0.04
        unit_mse = normalized_mse / 4

        normalized_psnr = mse_to_psnr(normalized_mse, (-1.0, 1.0))
        unit_psnr = mse_to_psnr(unit_mse, (0.0, 1.0))
        self.assertAlmostEqual(normalized_psnr, unit_psnr)
        self.assertAlmostEqual(normalized_psnr, 20.0)

    def test_zero_error_has_infinite_psnr(self):
        self.assertEqual(mse_to_psnr(0.0, (-1.0, 1.0)), math.inf)

    def test_invalid_range_and_error_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "value_range"):
            mse_to_psnr(0.1, (1.0, -1.0))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            mse_to_psnr(-0.1, (-1.0, 1.0))

    def test_training_log_uses_signal_range(self):
        writer = Mock()
        with patch("implicit_neural_representations.utils.tqdm.write"):
            log_training_metrics(
                1,
                torch.tensor(0.04),
                datetime.datetime.now(),
                writer,
                (-1.0, 1.0),
            )

        psnr_call = writer.add_scalar.call_args_list[1]
        self.assertEqual(psnr_call.args[0], "psnr")
        self.assertAlmostEqual(psnr_call.args[1], 20.0, places=6)
        self.assertEqual(psnr_call.args[2], 1)


if __name__ == "__main__":
    unittest.main()
