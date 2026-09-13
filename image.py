import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.dataset import ImageDataset
from modules.training import fit, predict_chunks
from models.model_factory import build_model
from models.presets import resolve_model_preset
from modules.utils import parse_config


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
    if task != 'image':
        raise ValueError(f"image.py requires task = image, got {task!r}")
    preset = resolve_model_preset(model_type, task, channels)

    # Data
    dataset = ImageDataset(sidelength, path=image_path, channels=channels)
    height, width = dataset.height, dataset.width

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

    coords, pixels = dataset.coords.to(device), dataset.pixels.to(device)
    fit(
        model, coords, pixels, optimizer,
        total_steps=total_steps,
        log_interval=log_interval,
        writer=writer,
        sampling='full',
    )

    # Evaluation
    preds_all = predict_chunks(model, coords, chunk_size).numpy()

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
