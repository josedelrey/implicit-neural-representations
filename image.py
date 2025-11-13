import os
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.dataset import ImageDataset
from models.model_factory import build_model
from modules.utils import parse_config, log_training_metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the image regression experiment. "
                    "Parameters are loaded from a config file."
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

    # Parameters
    task = config.get('task', 'image')
    image_path = config.get('image_path', 'images/cameraman.png')
    export_path = config.get('export_path', 'reconstructed.png')  # NEW

    is_rgb = config.get('is_rgb', 'False').lower() == 'true'
    sidelength = int(config.get('sidelength', '256'))
    channels = 3 if is_rgb else 1

    total_steps = int(config.get('total_steps', '1000'))
    log_interval = int(config.get('log_interval', '10'))
    chunk_size = int(config.get('chunk_size', '4096'))
    model_type = config.get('model_type', 'waveletnetnormalized')

    # Data
    dataset = ImageDataset(sidelength, path=image_path, channels=channels)
    loader = DataLoader(dataset, batch_size=1, pin_memory=(device.type == 'cuda'), num_workers=0)
    height, width = dataset.height, dataset.width

    # Model and optimizer
    model, learning_rate = build_model(model_type, task, channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_text('config', str(config))

    # Training loop
    coords, pixels = next(iter(loader))
    coords, pixels = coords.to(device), pixels.to(device)

    start_time = datetime.datetime.now()
    for step in range(total_steps + 1):
        coords_squeezed = coords.squeeze(0)
        preds = model(coords_squeezed)
        loss = ((preds - pixels) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            log_training_metrics(step, loss, start_time, writer)

    # Evaluation
    model.eval()
    with torch.no_grad():
        full_uv = dataset.coords.to(device)
        predictions = []
        for i in range(0, full_uv.shape[0], chunk_size):
            predictions.append(model(full_uv[i:i+chunk_size]))
        preds_all = torch.cat(predictions, dim=0).cpu().numpy()

    # Reconstruct image buffer
    image = (
        preds_all.reshape(height, width, channels)
        if channels == 3
        else preds_all.reshape(height, width)
    )
    image = (image + 1) / 2  # normalize to [0,1]

    # Ensure output directory exists
    export_dir = os.path.dirname(export_path)
    if export_dir != "":
        os.makedirs(export_dir, exist_ok=True)

    # Always export the reconstructed PNG (no visualization)
    plt.imsave(export_path, image, cmap=None if channels == 3 else 'gray')
    print(f"Reconstructed image saved to: {export_path}")

    writer.close()


if __name__ == '__main__':
    main()
