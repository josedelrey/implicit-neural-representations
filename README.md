# implicit-neural-representations

Python experiments for fitting implicit neural representations to images and videos.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
```

uv selects a Python version compatible with `pyproject.toml`, installs the
package and dependencies from `uv.lock`, and creates a local `.venv`.

To change dependencies, edit `pyproject.toml` (or use `uv add`) and regenerate
`uv.lock` with `uv lock`.

## Run

Put an input image in `examples/` or a video at the path in its config file, then run:

```bash
uv run --locked inr-image --config configs/image.yaml
uv run --locked inr-video --config configs/video.yaml
```

Edit the YAML config files to choose the input path, model, and training
settings. Each script determines the task, so there is no `task` setting in the
file. Paths are relative to the working directory. Image reconstructions are
written to `output.path`; training metrics and the fully resolved config are
written to TensorBoard's `runs/` directory. The video script displays its
reconstructed first frame.

Architecture implementations live in
`implicit_neural_representations/architectures/`, while model defaults live in
`implicit_neural_representations/presets.py`. Use `model.overrides` to change a
preset parameter for one run, or set `training.learning_rate` to override its
default learning rate. The task and color mode determine the input and output
dimensions. For example, within a complete config:

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

The `assets/` directory is reserved for images used in this README.

Pixel values are normalized to `[-1, 1]` during fitting. PSNR uses that two-unit
value range, and reconstructions are mapped to `[0, 1]` for saving or display.

## Architecture sources

The architecture files are either adapted from cited implementations or
implemented directly from the cited paper equations. The wavelet MFN models,
including the normalized and vector-frequency versions, are local extensions.
The configurable MLP is a local baseline.

| Models | Paper | Implementation basis |
| --- | --- | --- |
| `siren` | [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) (Sitzmann et al., 2020) | [vsitzmann/siren](https://github.com/vsitzmann/siren) |
| `fouriernet`, `gabornet`, `waveletnet`, `waveletnetnormalized`, `vectorwaveletnetnormalized` | [Multiplicative Filter Networks](https://openreview.net/forum?id=OmtmcPkkhT) (Fathony et al., 2021) | [boschresearch/multiplicative-filter-networks](https://github.com/boschresearch/multiplicative-filter-networks) |
| `wire` | [WIRE: Wavelet Implicit Neural Representations](https://arxiv.org/abs/2301.05187) (Saragadam et al., 2023) | [vishwa91/wire](https://github.com/vishwa91/wire) |
| `finer` | [FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation](https://arxiv.org/abs/2407.19434) (Zhu et al., 2024) | Paper-based local implementation |
| `frinr` | [Improved Implicit Neural Representation with Fourier Reparameterized Training](https://arxiv.org/abs/2401.07402) (Shi et al., 2024) | Paper-based local implementation |

The coordinate encodings draw on [NeRF](https://arxiv.org/abs/2003.08934)
([code](https://github.com/bmild/nerf)) and
[Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
([code](https://github.com/tancik/fourier-feature-networks)). They are implemented
locally and do not reproduce either full method.

## License

Project-authored code and the MFN adaptation are covered by
[AGPL-3.0-only](LICENSE). Copyright (c) 2026 José del Rey for project-authored
code. The SIREN and WIRE implementations retain their upstream MIT notices in
the third-party notices section of [LICENSE](LICENSE).
