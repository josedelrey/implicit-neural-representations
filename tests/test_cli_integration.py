import json
import math
import os
import subprocess
import sysconfig
import tempfile
import unittest
from pathlib import Path

import imageio
import numpy as np
import yaml
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class CliIntegrationTests(unittest.TestCase):
    def _run_cli(self, command: str, run_directory: Path, config: dict):
        config_path = run_directory / "experiment.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        executable = Path(sysconfig.get_path("scripts")) / command
        if os.name == "nt":
            executable = executable.with_suffix(".exe")
        self.assertTrue(executable.is_file(), f"Missing entry point: {executable}")

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["MPLBACKEND"] = "Agg"
        completed = subprocess.run(
            [str(executable), "--config", str(config_path)],
            cwd=run_directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return completed

    def _assert_tensorboard_output(self, run_directory: Path, task: str):
        event_files = list((run_directory / "runs").rglob("events.out.tfevents.*"))
        self.assertEqual(len(event_files), 1)
        events = EventAccumulator(str(event_files[0].parent)).Reload()

        scalar_tags = events.Tags()["scalars"]
        self.assertIn("loss", scalar_tags)
        self.assertIn("psnr", scalar_tags)
        for tag in ("loss", "psnr"):
            values = events.Scalars(tag)
            self.assertTrue(values)
            self.assertTrue(all(math.isfinite(event.value) for event in values))

        text_tag = "resolved_config/text_summary"
        self.assertIn(text_tag, events.Tags()["tensors"])
        tensor = events.Tensors(text_tag)[0].tensor_proto
        resolved = json.loads(tensor.string_val[0].decode("utf-8"))
        self.assertEqual(resolved["task"], task)
        self.assertEqual(resolved["device"], "cpu")
        self.assertEqual(resolved["working_directory"], str(run_directory))
        return resolved

    @staticmethod
    def _config(data_path: Path, output_path: Path, *, task: str):
        training = {"total_steps": 1, "log_interval": 1, "seed": 3}
        if task == "video":
            training["batch_size"] = 8
        return {
            "data": {"path": str(data_path), "sidelength": 4, "is_rgb": True},
            "model": {
                "name": "mlp",
                "overrides": {
                    "hidden_features": 4,
                    "hidden_layers": 0,
                    "act": "relu",
                    "act_trainable": False,
                },
            },
            "training": training,
            "output": {"path": str(output_path), "chunk_size": 8},
        }

    def test_image_entry_point_writes_reconstruction_and_run_data(self):
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) * 5
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            source_path = run_directory / "source.png"
            output_path = run_directory / "nested" / "image" / "reconstruction.png"
            Image.fromarray(pixels).save(source_path)
            self.assertFalse(output_path.parent.exists())

            completed = self._run_cli(
                "inr-image",
                run_directory,
                self._config(source_path, output_path, task="image"),
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.parent.is_dir())
            with Image.open(output_path) as reconstruction:
                self.assertEqual(reconstruction.size, (4, 4))
            self.assertIn("Reconstructed image saved to:", completed.stdout)
            resolved = self._assert_tensorboard_output(run_directory, "image")
            self.assertEqual(resolved["output"]["path"], str(output_path))

    def test_video_entry_point_writes_reconstruction_and_run_data(self):
        first = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) * 5
        frames = np.stack((first, 255 - first))
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            source_path = run_directory / "source.mp4"
            output_path = run_directory / "nested" / "video" / "reconstruction.gif"
            imageio.mimwrite(source_path, frames, fps=2, macro_block_size=1, quality=10)
            self.assertFalse(output_path.parent.exists())

            completed = self._run_cli(
                "inr-video",
                run_directory,
                self._config(source_path, output_path, task="video"),
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.parent.is_dir())
            with Image.open(output_path) as reconstruction:
                self.assertEqual(reconstruction.n_frames, 2)
                self.assertEqual(reconstruction.size, (4, 4))
            self.assertIn("Reconstructed video saved to:", completed.stdout)
            prefix = "Average PSNR over all frames: "
            reported_psnr = next(
                float(line.removeprefix(prefix))
                for line in completed.stdout.splitlines()
                if line.startswith(prefix)
            )
            self.assertTrue(math.isfinite(reported_psnr))
            resolved = self._assert_tensorboard_output(run_directory, "video")
            self.assertEqual(resolved["output"]["path"], str(output_path))


if __name__ == "__main__":
    unittest.main()
