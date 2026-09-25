import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import load_experiment_config
from .dataset import load_video_signal
from .loss import mse_to_psnr
from .model_factory import build_model
from .training import fit, predict_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run the video regression experiment. Parameters are loaded from a config file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the experiment YAML file"
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config, task='video')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )

    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    signal = load_video_signal(config.data_path, config.sidelength, config.channels)
    height, width = signal.spatial_shape
    num_frames = signal.frame_count

    model = build_model(config.model_type, **config.model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter()
    writer.add_text(
        'resolved_config',
        json.dumps(
            config.resolved_dict(device=str(device), value_range=signal.value_range),
            indent=2,
            sort_keys=True,
        ),
    )

    fit(
        model,
        signal.coords,
        signal.pixels,
        optimizer,
        total_steps=config.total_steps,
        log_interval=config.log_interval,
        writer=writer,
        value_range=signal.value_range,
        sampling='random',
        batch_size=config.batch_size,
    )

    preds_all = predict_chunks(model, signal.coords, config.chunk_size).numpy()

    video_pred = preds_all.reshape(num_frames, height, width, config.channels)
    video_truth = signal.pixels.numpy().reshape(
        num_frames, height, width, config.channels
    )
    psnr_vals = [
        mse_to_psnr(((video_pred[t] - video_truth[t]) ** 2).mean(), signal.value_range)
        for t in range(num_frames)
    ]
    avg_psnr = np.mean(psnr_vals)
    print(f"Average PSNR over all frames: {avg_psnr:.6f}")

    reconstructed = np.clip(signal.to_unit_range(video_pred), 0, 1)
    frames = np.rint(255 * reconstructed).astype(np.uint8)
    if config.channels == 1:
        frames = frames[..., 0]

    export_path = Path(config.export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    writer_options = {'fps': signal.frame_rate}
    ffmpeg_extensions = {'.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm'}
    if export_path.suffix.lower() in ffmpeg_extensions:
        writer_options['macro_block_size'] = 1
    imageio.mimwrite(export_path, frames, **writer_options)
    print(f"Reconstructed video saved to: {config.export_path}")

    writer.close()


if __name__ == '__main__':
    main()
