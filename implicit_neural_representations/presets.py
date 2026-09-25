"""Task-specific model defaults and learning rates for experiments."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

from .model_factory import validate_model_kwargs


@dataclass(frozen=True)
class ModelPreset:
    lr: float
    kwargs: dict


@dataclass(frozen=True)
class ResolvedModelPreset:
    learning_rate: float
    kwargs: dict


BASE_PRESETS = {
    "mlp": ModelPreset(
        lr=1e-3,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "act": "gaussian",
            "act_trainable": True,
            "use_pe": False,
            "L": 10,
            "a": 0.1,
        },
    ),
    "siren": ModelPreset(
        lr=1e-4,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "outermost_linear": True,
            "first_omega_0": 30,
            "hidden_omega_0": 30,
        },
    ),
    "fouriernet": ModelPreset(
        lr=1e-2,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "input_scale": 256.0,
            "weight_scale": 1.0,
            "bias": True,
            "output_act": False,
        },
    ),
    "gabornet": ModelPreset(
        lr=1e-2,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "input_scale": 256.0,
            "weight_scale": 1.0,
            "alpha": 6.0,
            "beta": 1.0,
            "bias": True,
            "output_act": False,
        },
    ),
    "waveletnet": ModelPreset(
        lr=1e-3,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "input_scale": 128.0,
            "weight_scale": 1.0,
            "alpha": 6.0,
            "beta": 1.0,
            "omega0": 5.0,
            "bias": True,
            "output_act": False,
        },
    ),
    "waveletnetnormalized": ModelPreset(
        lr=1e-3,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "input_scale": 128.0,
            "weight_scale": 1.0,
            "alpha": 6.0,
            "beta": 1.0,
            "omega0": 5.0,
            "bias": True,
            "output_act": False,
        },
    ),
    "vectorwaveletnetnormalized": ModelPreset(
        lr=1e-3,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "input_scale": 128.0,
            "weight_scale": 1.0,
            "alpha": 6.0,
            "beta": 1.0,
            "omega0": [5.0, 5.0],
            "bias": True,
            "output_act": False,
        },
    ),
    "wire": ModelPreset(
        lr=2e-2,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "outermost_linear": True,
            "first_omega_0": 10.0,
            "hidden_omega_0": 10.0,
            "scale": 6.0,
            "pos_encode": False,
            "L": 6,
        },
    ),
    "finer": ModelPreset(
        lr=1e-4,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "first_omega": 30,
            "hidden_omega": 30,
            "init_method": "sine",
            "init_gain": 1,
            "fbs": 2**-0.5,
            "hbs": 2**-0.5,
        },
    ),
    "frinr": ModelPreset(
        lr=1e-4,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "mode": "sin",
            "outermost_linear": True,
            "frequency_num": 128,
            "phase_num": 32,
            "first_omega_0": 30.0,
            "hidden_omega_0": 30.0,
            "mapping_input": 256,
        },
    ),
}

# Overrides for fitting a video instead of an image.
OVERRIDE_PRESETS = {
    "image": {},
    "video": {
        "mlp": {"kwargs": {"hidden_features": 512}},
        "siren": {"kwargs": {"hidden_features": 512}},
        "fouriernet": {"kwargs": {"hidden_features": 512}},
        "gabornet": {"kwargs": {"hidden_features": 512}},
        "waveletnet": {"kwargs": {"hidden_features": 512, "omega0": 0.8}},
        "waveletnetnormalized": {"kwargs": {"hidden_features": 512, "omega0": 0.8}},
        "vectorwaveletnetnormalized": {
            "kwargs": {
                "hidden_features": 512,
                "omega0": [0.7, 5.0, 5.0],
            }
        },
        "wire": {"kwargs": {"hidden_features": 512}},
        "finer": {"kwargs": {"hidden_features": 512}},
        "frinr": {"kwargs": {"hidden_features": 512}},
    },
}


def _validate_override(model_type: str, key: str, value, default) -> None:
    label = f"model.overrides.{key}"
    if key == "omega0" and model_type == "vectorwaveletnetnormalized":
        values = value if isinstance(value, list) else [value]
        if not values or any(
            type(item) not in (int, float) or not isfinite(item) or item <= 0
            for item in values
        ):
            raise ValueError(
                f"{label} must be a positive number or list of positive numbers"
            )
        return
    if default is None:
        if value is None:
            return
        if (
            key in ("fbs", "hbs")
            and type(value) in (int, float)
            and isfinite(value)
            and value >= 0
        ):
            return
        raise ValueError(f"{label} has an invalid type or value")
    if type(default) is bool:
        valid = type(value) is bool
    elif key in (
        "hidden_layers",
        "hidden_features",
        "L",
        "frequency_num",
        "phase_num",
        "mapping_input",
    ):
        valid = type(value) is int and value >= (
            1
            if key
            in (
                "hidden_features",
                "frequency_num",
                "phase_num",
                "mapping_input",
            )
            else 0
        )
    elif type(default) in (int, float):
        valid = type(value) in (int, float) and isfinite(value) and value > 0
    elif type(default) is str:
        valid = isinstance(value, str) and bool(value)
    else:
        valid = type(value) is type(default)
    if not valid:
        raise ValueError(f"{label} has an invalid type or value")
    if key == "act" and value not in {
        "relu",
        "gaussian",
        "quadratic",
        "multi-quadratic",
        "laplacian",
        "super-gaussian",
        "expsin",
    }:
        raise ValueError(f"{label} is not a supported activation")
    if key == "init_method" and value not in {"sine", "pytorch"}:
        raise ValueError(f"{label} is not a supported initialization method")


def resolve_model_preset(
    model_type: str,
    task: str,
    channels: int,
    *,
    overrides: Mapping[str, object] | None = None,
    learning_rate: float | None = None,
) -> ResolvedModelPreset:
    """Resolve and validate the complete model settings before construction."""
    if model_type not in BASE_PRESETS:
        raise ValueError(f"Unknown model: {model_type!r}")
    if task not in OVERRIDE_PRESETS:
        raise ValueError(f"Unknown task: {task!r}")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")

    base = BASE_PRESETS[model_type]
    task_overrides = OVERRIDE_PRESETS[task].get(model_type, {})
    defaults = {**base.kwargs, **task_overrides.get("kwargs", {})}
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("model.overrides must be a mapping")
    allowed = set(defaults) | ({"b"} if model_type == "mlp" else set())
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"Unknown model override: {key!r}")
        _validate_override(model_type, key, value, defaults.get(key, 1.0))
    if learning_rate is not None and (
        type(learning_rate) not in (int, float)
        or not isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("training.learning_rate must be a positive number")
    kwargs = {
        **deepcopy(defaults),
        **deepcopy(overrides),
        "in_features": 2 if task == "image" else 3,
        "out_features": channels,
    }
    validate_model_kwargs(model_type, kwargs)
    lr = (
        learning_rate
        if learning_rate is not None
        else task_overrides.get("lr", base.lr)
    )
    return ResolvedModelPreset(lr, kwargs)
