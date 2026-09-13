import datetime
from tqdm import tqdm

from .loss import mse_to_psnr


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


def log_training_metrics(step, loss, start_time, writer, value_range):
    """
    Log training metrics.
    """
    elapsed_str = format_elapsed_time(start_time)
    mse = loss.item()
    psnr = mse_to_psnr(mse, value_range)
    log_message = (f"[{elapsed_str}] [Iter {step:07d}]"
                   f"MSE: {mse:.4f} PSNR: {psnr:.2f}")
    tqdm.write(log_message)
    writer.add_scalar('loss', mse, step)
    writer.add_scalar('psnr', psnr, step)
