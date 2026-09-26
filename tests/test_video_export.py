import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio
import numpy as np

from implicit_neural_representations.dataset import load_video_signal
from implicit_neural_representations.video import save_video


class VideoExportTests(unittest.TestCase):
    def test_webm_and_m4v_exports_decode_and_can_be_loaded_as_signals(self):
        for extension, codec in (("webm", "vp9"), ("m4v", "h264")):
            for channels in (1, 3):
                with (
                    self.subTest(extension=extension, channels=channels),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    frames = np.stack(
                        [
                            np.full((5, 7, channels), value, dtype=np.uint8)
                            for value in (32, 128, 224)
                        ]
                    )
                    if channels == 1:
                        frames = frames[..., 0]
                    path = Path(directory) / f"reconstruction.{extension}"

                    save_video(path, frames, 2)

                    format_hint = ".mp4" if extension == "m4v" else None
                    with imageio.get_reader(path, format=format_hint) as reader:
                        metadata = reader.get_meta_data()
                        decoded = list(reader.iter_data())
                    self.assertGreater(path.stat().st_size, 0)
                    self.assertEqual(metadata["codec"], codec)
                    self.assertAlmostEqual(metadata["fps"], 2)
                    self.assertEqual(len(decoded), len(frames))
                    self.assertTrue(all(frame.shape == (6, 8, 3) for frame in decoded))
                    for frame, value in zip(decoded, (32, 128, 224)):
                        np.testing.assert_allclose(frame, value, atol=3)

                    signal = load_video_signal(
                        str(path), sidelength=8, channels=channels
                    )
                    self.assertEqual(signal.frame_count, len(frames))
                    self.assertEqual(signal.spatial_shape, (6, 8))
                    self.assertEqual(signal.channels, channels)
                    self.assertAlmostEqual(signal.frame_rate, 2)

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
