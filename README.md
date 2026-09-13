# implicit-neural-representations

Python experiments for fitting implicit neural representations to images and videos.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
```

uv selects a Python version compatible with `pyproject.toml`, installs the dependencies from `uv.lock`, and creates a local `.venv`.

## Run

Put an input image or video at the path in the corresponding config file, then run:

```bash
uv run --locked image.py --config config/image_config.txt
uv run --locked video.py --config config/video_config.txt
```

Edit the config files to choose the input path and model. Image reconstructions are written to the configured `export_path`; training metrics are written to TensorBoard's `runs/` directory. The video script displays its reconstructed first frame.

Pixel values are normalized to `[-1, 1]` during fitting. PSNR uses that two-unit value range, and reconstructions are mapped to `[0, 1]` for saving or display.

To change dependencies, edit `pyproject.toml` (or use `uv add`) and regenerate `uv.lock` with `uv lock`.
