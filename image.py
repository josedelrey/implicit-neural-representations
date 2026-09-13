import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.config import load_experiment_config
from modules.dataset import load_image_signal
from modules.training import fit, predict_chunks
from models.model_factory import build_model


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
        help="Path to the experiment YAML file"
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config, task='image')

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
    signal = load_image_signal(config.data_path, config.sidelength, config.channels)
    height, width = signal.spatial_shape

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
        sampling='full',
    )

    # Evaluation
    preds_all = predict_chunks(model, signal.coords, config.chunk_size).numpy()

    # Reconstruct image buffer
    image = (
        preds_all.reshape(height, width, config.channels)
        if config.channels == 3
        else preds_all.reshape(height, width)
    )
    image = np.clip(signal.to_unit_range(image), 0, 1)

    # Ensure output directory exists
    export_dir = os.path.dirname(config.export_path)
    if export_dir != "":
        os.makedirs(export_dir, exist_ok=True)

    # Always export the reconstructed PNG (no visualization)
    plt.imsave(config.export_path, image, cmap=None if config.channels == 3 else 'gray')
    print(f"Reconstructed image saved to: {config.export_path}")

    writer.close()


if __name__ == '__main__':
    main()
