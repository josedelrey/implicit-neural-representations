import math


def mse_to_psnr(mse: float, value_range: tuple[float, float]) -> float:
    """Compute PSNR for MSE measured in the supplied signal value range."""
    lower, upper = value_range
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("value_range must contain finite, increasing bounds")
    if not math.isfinite(mse) or mse < 0:
        raise ValueError("mse must be finite and non-negative")
    if mse == 0:
        return math.inf
    return 20 * math.log10(upper - lower) - 10 * math.log10(mse)
