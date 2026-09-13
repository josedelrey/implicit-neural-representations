# implicit-neural-representations

Python experiments for fitting implicit neural representations to images and videos.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
```

uv selects a Python version compatible with `pyproject.toml`, installs the package and dependencies from `uv.lock`, and creates a local `.venv`.

## Run

Put an input image or video at the path in the corresponding config file, then run:

```bash
uv run --locked inr-image --config config/image.yaml
uv run --locked inr-video --config config/video.yaml
```

Edit the YAML config files to choose the input path, model, and training settings. Each script determines the task, so there is no `task` setting in the file. Paths are relative to the working directory. Image reconstructions are written to `output.path`; training metrics and the fully resolved config are written to TensorBoard's `runs/` directory. The video script displays its reconstructed first frame.

Architecture implementations live in `implicit_neural_representations/architectures/`, while model defaults live in `implicit_neural_representations/presets.py`. Use `model.overrides` to change a preset parameter for one run, or set `training.learning_rate` to override its default learning rate. The task and color mode determine the input and output dimensions. For example, within a complete config:

```yaml
model:
  name: waveletnetnormalized
  overrides:
    hidden_features: 384
training:
  total_steps: 1000
  log_interval: 10
  learning_rate: 0.0005
```

The loader rejects unknown keys, invalid types, and unsupported model overrides before loading the input data.

Pixel values are normalized to `[-1, 1]` during fitting. PSNR uses that two-unit value range, and reconstructions are mapped to `[0, 1]` for saving or display.

To change dependencies, edit `pyproject.toml` (or use `uv add`) and regenerate `uv.lock` with `uv lock`.
