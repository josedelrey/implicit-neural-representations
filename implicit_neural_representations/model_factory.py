"""Construct INR models from explicit architecture parameters."""

from inspect import signature

from .architectures.finer import Finer
from .architectures.frinr import FRINR
from .architectures.mfn import (
    FourierNet,
    GaborNet,
    VectorWaveletNetNormalized,
    WaveletNet,
    WaveletNetNormalized,
)
from .architectures.mlp import MLP
from .architectures.siren import Siren
from .architectures.wire import WIRE

MODEL_CLASSES = {
    "mlp": MLP,
    "siren": Siren,
    "fouriernet": FourierNet,
    "gabornet": GaborNet,
    "waveletnet": WaveletNet,
    "waveletnetnormalized": WaveletNetNormalized,
    "vectorwaveletnetnormalized": VectorWaveletNetNormalized,
    "wire": WIRE,
    "finer": Finer,
    "frinr": FRINR,
}


def validate_model_kwargs(model_type: str, kwargs: dict) -> None:
    """Check configuration errors without constructing a model."""
    try:
        model_class = MODEL_CLASSES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_type!r}") from exc

    signature(model_class).bind(**kwargs)
    in_features = kwargs["in_features"]
    out_features = kwargs["out_features"]
    if in_features <= 0 or out_features <= 0:
        raise ValueError("in_features and out_features must be positive")

    if model_type == "vectorwaveletnetnormalized":
        omega0 = kwargs.get("omega0", 5.0)
        if not isinstance(omega0, (float, int)) and len(omega0) != in_features:
            raise ValueError(
                f"omega0 length {len(omega0)} != in_features {in_features}"
            )
    elif model_type == "frinr":
        FRINR.validate_mode(kwargs["mode"])


def build_model(model_type: str, *, in_features: int, out_features: int, **kwargs):
    """Return a model without choosing its task, optimizer, or device."""
    model_kwargs = {"in_features": in_features, "out_features": out_features, **kwargs}
    validate_model_kwargs(model_type, model_kwargs)
    return MODEL_CLASSES[model_type](**model_kwargs)
