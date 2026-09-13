"""Task-specific model defaults and learning rates for experiments."""

from copy import deepcopy
from dataclasses import dataclass

from models.model_factory import validate_model_kwargs


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
            "init_method": 'sine',
            "init_gain": 1,
            "fbs": None,
            "hbs": None,
            "alphaType": None,
            "alphaReqGrad": False,
        },
    ),
    "frinr": ModelPreset(
        lr=1e-4,
        kwargs={
            "hidden_layers": 4,
            "hidden_features": 256,
            "mode": 'sin',
            "outermost_linear": True,
            "high_freq_num": 128,
            "low_freq_num": 128,
            "phi_num": 32,
            "alpha": 0.01,
            "first_omega_0": 30.0,
            "hidden_omega_0": 30.0,
            "pe": False,
        },
    ),
}

# Overrides for fitting a video instead of an image.
OVERRIDE_PRESETS = {
    "image": {},
    "video": {
        "mlp": dict(kwargs={"hidden_features": 512}),
        "siren": dict(kwargs={"hidden_features": 512}),
        "fouriernet": dict(kwargs={"hidden_features": 512}),
        "gabornet": dict(kwargs={"hidden_features": 512}),
        "waveletnet": dict(kwargs={"hidden_features": 512, "omega0": 0.8}),
        "waveletnetnormalized": dict(kwargs={"hidden_features": 512, "omega0": 0.8}),
        "vectorwaveletnetnormalized": dict(kwargs={
            "hidden_features": 512,
            "omega0": [0.7, 5.0, 5.0],
        }),
        "wire": dict(kwargs={"hidden_features": 512}),
        "finer": dict(kwargs={"hidden_features": 512}),
        "frinr": dict(kwargs={"hidden_features": 512}),
    },
}


def resolve_model_preset(model_type: str, task: str, channels: int) -> ResolvedModelPreset:
    """Resolve and validate the complete model settings before construction."""
    if model_type not in BASE_PRESETS:
        raise ValueError(f"Unknown model: {model_type!r}")
    if task not in OVERRIDE_PRESETS:
        raise ValueError(f"Unknown task: {task!r}")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")

    base = BASE_PRESETS[model_type]
    overrides = OVERRIDE_PRESETS[task].get(model_type, {})
    kwargs = {
        **deepcopy(base.kwargs),
        **deepcopy(overrides.get("kwargs", {})),
        "in_features": 2 if task == "image" else 3,
        "out_features": channels,
    }
    validate_model_kwargs(model_type, kwargs)
    return ResolvedModelPreset(overrides.get("lr", base.lr), kwargs)
