import datetime
from tqdm import tqdm

from modules.loss import mse_to_psnr


def parse_config(config_path: str) -> dict:
    """
    Parse a configuration file where each non-empty, non-comment line is of the format:
        key = value  # optional inline comment
    Returns a dictionary mapping keys to values.
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            line = line.split('#', 1)[0].strip()

            if not line:
                continue

            if '=' in line:
                key, value = line.split('=', maxsplit=1)
                config[key.strip()] = value.strip()
            else:
                print(f"Warning: Invalid line in config file: {line}")

    return config


def format_elapsed_time(start_time: datetime.datetime) -> str:
    """
    Compute the elapsed time since start_time and format it as HH:MM:SS.
    """
    elapsed_time = datetime.datetime.now() - start_time
    total_seconds = int(elapsed_time.total_seconds())
    return '{:02d}:{:02d}:{:02d}'.format(
        total_seconds // 3600,
        (total_seconds % 3600) // 60,
        total_seconds % 60
    )


def log_training_metrics(step, loss, start_time, writer):
    """
    Log training metrics.
    """
    elapsed_str = format_elapsed_time(start_time)
    log_message = (f"[{elapsed_str}] [Iter {step:07d}]"
                   f"MSE: {loss.item():.4f} PSNR: {mse_to_psnr(loss.item()):.2f}")
    tqdm.write(log_message)
    writer.add_scalar('loss', loss.item(), step)
    writer.add_scalar('psnr', mse_to_psnr(loss.item()), step)
