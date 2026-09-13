import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import load_experiment_config
from .dataset import load_video_signal
from .loss import mse_to_psnr
from .training import fit, predict_chunks
from .model_factory import build_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the video regression experiment. Parameters are loaded from a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML file"
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config, task='video')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Reproducibility
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Data
    signal = load_video_signal(config.data_path, config.sidelength, config.channels)
    height, width = signal.spatial_shape
    num_frames = signal.frame_count

    # Model and optimizer
    model = build_model(config.model_type, **config.model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_text(
        'resolved_config',
        json.dumps(config.resolved_dict(device=str(device), value_range=signal.value_range), indent=2, sort_keys=True),
    )

    fit(
        model, signal.coords, signal.pixels, optimizer,
        total_steps=config.total_steps,
        log_interval=config.log_interval,
        writer=writer,
        value_range=signal.value_range,
        sampling='random',
        batch_size=config.batch_size,
    )

    # Evaluation
    preds_all = predict_chunks(model, signal.coords, config.chunk_size).numpy()

    # Reshape to video format and compute average PSNR
    video_pred = preds_all.reshape(num_frames, height, width, config.channels)
    video_truth = signal.pixels.numpy().reshape(num_frames, height, width, config.channels)
    psnr_vals = [
        mse_to_psnr(((video_pred[t] - video_truth[t]) ** 2).mean(), signal.value_range)
        for t in range(num_frames)
    ]
    avg_psnr = np.mean(psnr_vals)
    print(f"Average PSNR over all frames: {avg_psnr:.6f}")

    # Visualization of first frame
    first = signal.to_unit_range(video_pred[0])
    if config.channels == 3:
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
