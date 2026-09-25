import argparse

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import save_checkpoint, save_metrics
from .config import load_experiment_config
from .dataset import load_image_signal
from .experiment import open_experiment
from .loss import mse_to_psnr
from .training import fit, predict_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run the image regression experiment. "
        "Parameters are loaded from a config file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the experiment YAML file"
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config, task='image')

    signal = load_image_signal(config.data_path, config.sidelength, config.channels)
    height, width = signal.spatial_shape

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
            sampling='full',
        )

        preds_all = predict_chunks(
            run.model, signal.coords, config.chunk_size
        ).numpy()
        targets = signal.pixels.numpy()
        mse = float(np.mean((preds_all - targets) ** 2))
        metrics = {'mse': mse, 'psnr': mse_to_psnr(mse, signal.value_range)}

        image = (
            preds_all.reshape(height, width, config.channels)
            if config.channels == 3
            else preds_all.reshape(height, width)
        )
        image = np.clip(signal.to_unit_range(image), 0, 1)
        plt.imsave(
            run.reconstruction_path,
            image,
            cmap=None if config.channels == 3 else 'gray',
        )

        save_metrics(run.run_directory, metrics)
        save_checkpoint(
            run.run_directory,
            model_type=config.model_type,
            model_kwargs=config.model_kwargs,
            model=run.model,
            optimizer=run.optimizer,
            completed_steps=config.total_steps,
        )

    print(f"Reconstructed image saved to: {run.reconstruction_path}")
    print(f"Run artifacts saved to: {run.run_directory}")


if __name__ == '__main__':
    main()
