import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio
import numpy as np

from implicit_neural_representations.video import save_video


class VideoExportTests(unittest.TestCase):
    def test_mp4_padding_preserves_pixels_and_even_dimensions(self):
        for height, width in ((5, 8), (8, 5), (5, 5), (6, 8)):
            with (
                self.subTest(height=height, width=width),
                tempfile.TemporaryDirectory() as directory,
            ):
                frames = np.full((2, height, width, 3), 128, dtype=np.uint8)
                path = Path(directory) / "reconstruction.mp4"
                save_video(path, frames, 2)
                with imageio.get_reader(path) as reader:
                    decoded = list(reader.iter_data())
                self.assertEqual(len(decoded), len(frames))
                expected_shape = (height + height % 2, width + width % 2, 3)
                for frame in decoded:
                    self.assertEqual(frame.shape, expected_shape)
                    np.testing.assert_allclose(frame, 128, atol=2)

    def test_empty_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction.mp4"
            frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
            with (
                patch(
                    "implicit_neural_representations.video.imageio.mimwrite",
                    side_effect=lambda *args, **kwargs: path.touch(),
                ),
                self.assertRaisesRegex(RuntimeError, "produced no data"),
            ):
                save_video(path, frames, 2)

    def test_nonempty_undecodable_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction.mp4"
            frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
            with (
                patch(
                    "implicit_neural_representations.video.imageio.mimwrite",
                    side_effect=lambda *args, **kwargs: path.write_bytes(b"invalid"),
                ),
                self.assertRaisesRegex(RuntimeError, "cannot be decoded"),
            ):
                save_video(path, frames, 2)

    def test_export_missing_final_frames_is_rejected(self):
        encode = imageio.mimwrite
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction.mp4"
            frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
            with (
                patch(
                    "implicit_neural_representations.video.imageio.mimwrite",
                    side_effect=lambda path, frames, **options: encode(
                        path, frames[:1], **options
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "cannot be decoded"),
            ):
                save_video(path, frames, 2)


if __name__ == "__main__":
    unittest.main()
