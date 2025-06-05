import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.dataset import VideoDataset
from modules.loss import mse_to_psnr
from models.model_factory import build_model
from modules.utils import parse_config, log_training_metrics


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

    # Data
    dataset = VideoDataset(sidelength, path=video_path, channels=channels)
    height, width, num_frames = dataset.height, dataset.width, dataset.num_frames
    coords = dataset.coords.to(device)
    pixels = dataset.pixels.to(device)
    num_coords = coords.shape[0]

    # Model and optimizer
    model, learning_rate = build_model(model_type, task, channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_text('config', str(config))

    # Training loop
    start_time = datetime.datetime.now()
    for step in range(total_steps + 1):
        optimizer.zero_grad()
        indices = torch.randint(0, num_coords, (batch_size,), device=coords.device)
        batch_coords = coords[indices]
        batch_pixels = pixels[indices]
        preds = model(batch_coords)
        loss = ((preds - batch_pixels) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            log_training_metrics(step, loss, start_time, writer)

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_list = []
        for i in range(0, num_coords, chunk_size):
            preds_list.append(model(coords[i:i+chunk_size]))
        preds_all = torch.cat(preds_list, dim=0).cpu().numpy()

    # Reshape to video format and compute average PSNR
    video_pred = preds_all.reshape(num_frames, height, width, channels)
    video_truth = dataset.pixels.cpu().numpy().reshape(num_frames, height, width, channels)
    psnr_vals = [
        mse_to_psnr(((video_pred[t] - video_truth[t]) ** 2).mean())
        for t in range(num_frames)
    ]
    avg_psnr = np.mean(psnr_vals)
    print(f"Average PSNR over all frames: {avg_psnr:.6f}")

    # Visualization of first frame
    first = (video_pred[0] + 1) / 2
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
