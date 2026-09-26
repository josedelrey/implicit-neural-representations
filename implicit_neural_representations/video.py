import argparse
from pathlib import Path

import imageio
import numpy as np

from .artifacts import save_checkpoint, save_metrics
from .config import load_experiment_config
from .dataset import load_video_signal
from .experiment import open_experiment
from .loss import mse_to_psnr
from .training import fit, predict_chunks


def save_video(path: Path, frames: np.ndarray, frame_rate: float) -> None:
    """Encode a reconstruction and check that the resulting video is readable."""
    writer_options = {"fps": frame_rate}
    ffmpeg_extensions = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
    uses_ffmpeg = path.suffix.lower() in ffmpeg_extensions
    if uses_ffmpeg:
        height, width = frames.shape[1:3]
        padding = [(0, 0), (0, height % 2), (0, width % 2)]
        if frames.ndim == 4:
            padding.append((0, 0))
        if height % 2 or width % 2:
            frames = np.pad(frames, padding, mode="edge")
        writer_options["macro_block_size"] = 1
    imageio.mimwrite(path, frames, **writer_options)

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video export produced no data: {path}")
    try:
        with imageio.get_reader(path) as reader:
            first_frame = reader.get_data(0)
            if uses_ffmpeg:
                reader.get_data(len(frames) - 1)
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        raise RuntimeError(f"Exported video cannot be decoded: {path}") from exc
    if first_frame.shape[:2] != frames.shape[1:3]:
        raise RuntimeError(f"Exported video dimensions do not match: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the video regression experiment. Parameters are loaded from a config file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the experiment YAML file"
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config, task="video")

    signal = load_video_signal(config.data_path, config.sidelength, config.channels)
    height, width = signal.spatial_shape
    num_frames = signal.frame_count

    with open_experiment(config, signal) as run:
        fit(
            run.model,
            signal.coords,
            signal.pixels,
            run.optimizer,
            total_steps=config.total_steps,
            log_interval=config.log_interval,
            writer=run.writer,
            value_range=signal.value_range,
            sampling="random",
            batch_size=config.batch_size,
        )

        preds_all = predict_chunks(run.model, signal.coords, config.chunk_size).numpy()

        video_pred = preds_all.reshape(num_frames, height, width, config.channels)
        video_truth = signal.pixels.numpy().reshape(
            num_frames, height, width, config.channels
        )
        mse = float(np.mean((video_pred - video_truth) ** 2))
        psnr_vals = [
            mse_to_psnr(
                float(np.mean((video_pred[t] - video_truth[t]) ** 2)),
                signal.value_range,
            )
            for t in range(num_frames)
        ]
        avg_psnr = float(np.mean(psnr_vals))
        metrics = {
            "mse": mse,
            "psnr": mse_to_psnr(mse, signal.value_range),
            "mean_frame_psnr": avg_psnr,
        }

        reconstructed = np.clip(signal.to_unit_range(video_pred), 0, 1)
        frames = np.rint(255 * reconstructed).astype(np.uint8)
        if config.channels == 1:
            frames = frames[..., 0]

        save_video(run.reconstruction_path, frames, signal.frame_rate)

        save_metrics(run.run_directory, metrics)
        save_checkpoint(
            run.run_directory,
            model_type=config.model_type,
            model_kwargs=config.model_kwargs,
            model=run.model,
            optimizer=run.optimizer,
            completed_steps=config.total_steps,
        )

    print(f"Average PSNR over all frames: {avg_psnr:.6f}")
    print(f"Reconstructed video saved to: {run.reconstruction_path}")
    print(f"Run artifacts saved to: {run.run_directory}")


if __name__ == "__main__":
    main()
