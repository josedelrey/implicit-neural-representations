import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from implicit_neural_representations.dataset import (
    SignalData,
    load_image_signal,
    load_video_signal,
)


class FakeReader:
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        self.closed = True


class SignalDataTests(unittest.TestCase):
    def test_image_and_video_share_preprocessing_and_coordinate_order(self):
        frame = np.array(
            [
                [[0, 10, 20], [30, 40, 50], [60, 70, 80]],
                [[90, 100, 110], [120, 130, 140], [210, 220, 230]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'frame.png'
            Image.fromarray(frame).save(image_path)
            image = load_image_signal(str(image_path), sidelength=6, channels=3)

        reader = FakeReader([frame, 255 - frame])
        with patch(
            'implicit_neural_representations.dataset.imageio.get_reader',
            return_value=reader,
        ):
            video = load_video_signal('example.mp4', sidelength=6, channels=3)

        self.assertEqual(image.spatial_shape, (4, 6))
        self.assertEqual(image.signal_shape, (4, 6))
        self.assertEqual(image.coordinate_order, ('y', 'x'))
        self.assertEqual(video.signal_shape, (2, 4, 6))
        self.assertEqual(video.coordinate_order, ('t', 'y', 'x'))
        self.assertEqual(image.value_range, (-1.0, 1.0))
        self.assertEqual(video.value_range, image.value_range)
        np.testing.assert_allclose(
            image.to_unit_range(np.array([-1.0, 0.0, 1.0])),
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(image.coords.shape, (24, 2))
        self.assertEqual(video.coords.shape, (48, 3))
        self.assertEqual(image.pixels.shape, (24, 3))
        self.assertEqual(video.pixels.shape, (48, 3))
        self.assertEqual(image.coords.device.type, 'cpu')
        self.assertEqual(video.pixels.device.type, 'cpu')
        torch.testing.assert_close(video.coords[:24, 1:], image.coords)
        torch.testing.assert_close(video.pixels[:24], image.pixels)
        torch.testing.assert_close(video.coords[0], torch.tensor([-1.0, -1.0, -1.0]))
        torch.testing.assert_close(video.coords[-1], torch.tensor([1.0, 1.0, 1.0]))
        self.assertTrue(reader.closed)

    def test_empty_video_closes_reader_and_fails_clearly(self):
        reader = FakeReader([])
        with patch(
            'implicit_neural_representations.dataset.imageio.get_reader',
            return_value=reader,
        ):
            with self.assertRaisesRegex(ValueError, 'contains no frames'):
                load_video_signal('empty.mp4', sidelength=8, channels=1)
        self.assertTrue(reader.closed)

    def test_signal_shape_must_match_flattened_tensors(self):
        with self.assertRaisesRegex(ValueError, 'declared signal shape'):
            SignalData(
                torch.zeros(3, 2),
                torch.zeros(3, 1),
                (2, 2),
                ('y', 'x'),
                (-1.0, 1.0),
            )


if __name__ == '__main__':
    unittest.main()
