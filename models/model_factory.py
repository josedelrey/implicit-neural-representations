from collections import namedtuple

from models.mlp import MLP
from models.siren import Siren
from models.mfn import FourierNet, GaborNet, MFNWaveletNet
from models.wavelet import WaveletNet
from models.wire import WIRE
from models.finer import Finer
from models.frinr import FRINR

# ----------------------------------------------------------------------------
# Base defaults for each model architecture
# ----------------------------------------------------------------------------
ModelSpec = namedtuple("ModelSpec", ["cls", "lr", "kwargs"])

BASE_SPECS = {
    "mlp": ModelSpec(
        cls=MLP,
        lr=1e-3,
        kwargs={
            "n_in": 2,
            "n_out": None,
            "n_layers": 4,
            "n_hidden_units": 256,
            "act": "gaussian",
            "act_trainable": True,
            "use_pe": False,
            "L": 10,
            "a": 0.1,
        },
    ),
    "siren": ModelSpec(
        cls=Siren,
        lr=1e-4,
        kwargs={
            "in_features": 2,
            "out_features": None,
            "hidden_layers": 4,
            "hidden_features": 256,
            "outermost_linear": True,
            "first_omega_0": 30,
            "hidden_omega_0": 30,
        },
    ),
    "gabornet": ModelSpec(
        cls=GaborNet,
        lr=1e-2,
        kwargs={
            "in_size": 2,
            "out_size": None,
            "n_layers": 4,
            "hidden_size": 256,
            "input_scale": 256.0,
            "weight_scale": 1.0,
            "alpha": 6.0,
            "beta": 1.0,
            "bias": True,
            "output_act": False,
        },
    ),
    "fouriernet": ModelSpec(
        cls=FourierNet,
        lr=1e-2,
        kwargs={
            "in_size": 2,
            "out_size": None,
            "n_layers": 4,
            "hidden_size": 256,
            "input_scale": 256.0,
            "weight_scale": 1.0,
            "bias": True,
            "output_act": False,
        },
    ),
    "mfnwaveletnet": ModelSpec(
        cls=MFNWaveletNet,
        lr=1e-3,
        kwargs={
            "in_size": 2,
            "out_size": None,
            "n_layers": 4,
            "hidden_size": 256,
            "input_scale": 128.0,
            "weight_scale": 1.0,
            "alpha": 6.0,
            "beta": 1.0,
            "omega0": 5.0,
        },
    ),
    "waveletnet": ModelSpec(
        cls=WaveletNet,
        lr=1e-3,
        kwargs={
            "in_features": 2,
            "out_features": None,
            "hidden_layers": 4,
            "hidden_features": 256,
            "omega0": 5.0,
        },
    ),
    "wire": ModelSpec(
        cls=WIRE,
        lr=1e-3,
        kwargs={
            "in_features": 2,
            "out_features": None,
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
    "finer": ModelSpec(
        cls=Finer,
        lr=1e-4,
        kwargs={
            "in_features": 2,
            "out_features": None,
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
    "frinr": ModelSpec(
        cls=FRINR,
        lr=1e-4,
        kwargs={
            "in_features": 2,
            "out_features": None,
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

# ----------------------------------------------------------------------------
# Optional overrides per task (e.g. image vs video)
# ----------------------------------------------------------------------------
OVERRIDE_SPECS = {
    "image": {
        "mlp": dict(lr=1e-3, kwargs={}),
        "siren": dict(lr=5e-5, kwargs={}),
        "gabornet": dict(lr=1e-2, kwargs={}),
        "fouriernet": dict(lr=1e-2, kwargs={}),
        "mfnwaveletnet": dict(lr=1e-3, kwargs={}),
        "waveletnet": dict(lr=1e-3, kwargs={}),
        "wire": dict(lr=1e-3, kwargs={}),
        "finer": dict(lr=1e-4, kwargs={}),
        "frinr": dict(lr=1e-4, kwargs={}),
    },
    "video": {
        "mlp": dict(lr=5e-4, kwargs={"n_hidden_units": 128}),
        "siren": dict(lr=1e-4, kwargs={"hidden_features": 128}),
    },
}

# ----------------------------------------------------------------------------
# Factory to build a model + learning rate
# ----------------------------------------------------------------------------
def build_model(model_type: str,
                task: str,
                channels: int,
                device="cuda"):
    if model_type not in BASE_SPECS:
        raise ValueError(f"Unknown model: {model_type!r}")
    base = BASE_SPECS[model_type]
    overrides = OVERRIDE_SPECS.get(task, {}).get(model_type, {})

    lr = overrides.get("lr", base.lr)
    kwargs = {**base.kwargs, **overrides.get("kwargs", {})}

    # Fill dynamic output size fields
    if "n_out" in kwargs and kwargs["n_out"] is None:
        kwargs["n_out"] = channels
    if "out_features" in kwargs and kwargs["out_features"] is None:
        kwargs["out_features"] = channels
    if "out_size" in kwargs and kwargs["out_size"] is None:
        kwargs["out_size"] = channels

    model = base.cls(**kwargs).to(device)
    return model, lr
