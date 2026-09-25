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
import torch
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

    def _assert_tensorboard_output(self, artifact_directory: Path, task: str):
        event_files = list(
            (artifact_directory / "tensorboard").rglob("events.out.tfevents.*")
        )
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
        self.assertEqual(
            resolved["working_directory"], str(artifact_directory.parents[1])
        )
        return resolved

    @staticmethod
    def _config(
        data_path: Path,
        artifact_directory: Path,
        reconstruction_file: str,
        *,
        task: str,
    ):
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
            "output": {
                "directory": str(artifact_directory),
                "reconstruction": reconstruction_file,
                "chunk_size": 8,
            },
        }

    def _assert_run_artifacts(self, artifact_directory: Path, task: str):
        for name in ("resolved_config.json", "checkpoint.pt", "metrics.json"):
            self.assertTrue((artifact_directory / name).is_file(), name)

        resolved = json.loads(
            (artifact_directory / "resolved_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(resolved["task"], task)
        serialized = json.dumps(resolved)
        for excluded in (
            "checksum",
            "sha256",
            "signal_shape",
            "git_commit",
            "python_version",
            "torch_version",
            "cuda_version",
            "device_name",
        ):
            self.assertNotIn(excluded, serialized)

        metrics = json.loads(
            (artifact_directory / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertTrue(math.isfinite(metrics["mse"]))
        self.assertTrue(math.isfinite(metrics["psnr"]))
        checkpoint = torch.load(
            artifact_directory / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(checkpoint["completed_steps"], 1)
        self.assertEqual(checkpoint["model_type"], "mlp")
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        return resolved, metrics

    def test_image_entry_point_writes_reconstruction_and_run_data(self):
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) * 5
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            source_path = run_directory / "source.png"
            artifact_directory = run_directory / "nested" / "image"
            output_path = artifact_directory / "reconstruction.png"
            Image.fromarray(pixels).save(source_path)
            self.assertFalse(artifact_directory.exists())

            completed = self._run_cli(
                "inr-image",
                run_directory,
                self._config(
                    source_path,
                    artifact_directory,
                    "reconstruction.png",
                    task="image",
                ),
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.parent.is_dir())
            with Image.open(output_path) as reconstruction:
                self.assertEqual(reconstruction.size, (4, 4))
            self.assertIn("Reconstructed image saved to:", completed.stdout)
            self.assertIn("Run artifacts saved to:", completed.stdout)
            resolved, _ = self._assert_run_artifacts(artifact_directory, "image")
            tensorboard_resolved = self._assert_tensorboard_output(
                artifact_directory, "image"
            )
            self.assertEqual(tensorboard_resolved, resolved)
            self.assertEqual(
                resolved["output"]["directory"], str(artifact_directory)
            )
            self.assertEqual(
                resolved["output"]["reconstruction"], "reconstruction.png"
            )

    def test_video_entry_point_writes_reconstruction_and_run_data(self):
        first = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) * 5
        frames = np.stack((first, 255 - first))
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            source_path = run_directory / "source.mp4"
            artifact_directory = run_directory / "nested" / "video"
            output_path = artifact_directory / "reconstruction.gif"
            imageio.mimwrite(source_path, frames, fps=2, macro_block_size=1, quality=10)
            self.assertFalse(output_path.parent.exists())

            completed = self._run_cli(
                "inr-video",
                run_directory,
                self._config(
                    source_path,
                    artifact_directory,
                    "reconstruction.gif",
                    task="video",
                ),
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
            self.assertIn("Run artifacts saved to:", completed.stdout)
            resolved, metrics = self._assert_run_artifacts(
                artifact_directory, "video"
            )
            self.assertTrue(math.isfinite(metrics["mean_frame_psnr"]))
            tensorboard_resolved = self._assert_tensorboard_output(
                artifact_directory, "video"
            )
            self.assertEqual(tensorboard_resolved, resolved)
            self.assertEqual(
                resolved["output"]["directory"], str(artifact_directory)
            )
            self.assertEqual(
                resolved["output"]["reconstruction"], "reconstruction.gif"
            )


if __name__ == "__main__":
    unittest.main()
