import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.dataset import load_video_signal
from modules.loss import mse_to_psnr
from modules.training import fit, predict_chunks
from models.model_factory import build_model
from models.presets import resolve_model_preset
from modules.utils import parse_config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the video regression experiment. Parameters are loaded from a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (each line: key = value)"
    )
    args = parser.parse_args()
    config = parse_config(args.config)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Parameters (loaded from config, with fallbacks)
    task = config.get('task', 'video')
    video_path = config.get('video_path', 'videos/akiyo_cif.y4m')
    is_rgb = config.get('is_rgb', 'True').lower() == 'true'
    sidelength = int(config.get('sidelength', '256'))
    channels = 3 if is_rgb else 1
    total_steps = int(config.get('total_steps', '10000'))
    log_interval = int(config.get('log_interval', '10'))
    batch_size = int(config.get('batch_size', '32768'))
    chunk_size = int(config.get('chunk_size', '1024'))
    model_type = config.get('model_type', 'vectorwaveletnetnormalized')
    if task != 'video':
        raise ValueError(f"video.py requires task = video, got {task!r}")
    preset = resolve_model_preset(model_type, task, channels)

    # Data
    signal = load_video_signal(video_path, sidelength, channels)
    height, width = signal.spatial_shape
    num_frames = signal.frame_count

    # Model and optimizer
    model = build_model(model_type, **preset.kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=preset.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_text('config', str(config))
    writer.add_text('resolved_model', json.dumps({
        'model_type': model_type,
        'task': task,
        'learning_rate': preset.learning_rate,
        'model_kwargs': preset.kwargs,
        'device': str(device),
        'seed': seed,
    }, indent=2, sort_keys=True))

    fit(
        model, signal.coords, signal.pixels, optimizer,
        total_steps=total_steps,
        log_interval=log_interval,
        writer=writer,
        sampling='random',
        batch_size=batch_size,
    )

    # Evaluation
    preds_all = predict_chunks(model, signal.coords, chunk_size).numpy()

    # Reshape to video format and compute average PSNR
    video_pred = preds_all.reshape(num_frames, height, width, channels)
    video_truth = signal.pixels.numpy().reshape(num_frames, height, width, channels)
    psnr_vals = [
        mse_to_psnr(((video_pred[t] - video_truth[t]) ** 2).mean())
        for t in range(num_frames)
    ]
    avg_psnr = np.mean(psnr_vals)
    print(f"Average PSNR over all frames: {avg_psnr:.6f}")

    # Visualization of first frame
    value_min, value_max = signal.value_range
    first = (video_pred[0] - value_min) / (value_max - value_min)
    if channels == 3:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(first, 0, 1))
        plt.title("Reconstructed First Frame")
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(first[..., 0], 0, 1), cmap='gray')
        plt.title("Reconstructed First Frame")
    plt.axis('off')
    plt.show()

    writer.close()


if __name__ == '__main__':
    main()
