"""Strict YAML configuration for image and video experiments."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re

import yaml

from .presets import resolve_model_preset


class ConfigError(ValueError):
    """The experiment configuration is malformed or unsupported."""


class StrictSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError(f'YAML keys must be strings (line {key_node.start_mark.line + 1})')
            if key in mapping:
                raise ConfigError(f'Duplicate YAML key {key!r} (line {key_node.start_mark.line + 1})')
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


# PyYAML otherwise treats common notation such as 1e-3 as a string.
StrictSafeLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+$'),
    list('-+0123456789.'),
)


def _mapping(value, label: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f'{label} must be a mapping')
    allowed = required | (optional or set())
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        raise ConfigError(f'{label} is missing required keys: {", ".join(sorted(missing))}')
    if unknown:
        raise ConfigError(f'{label} has unknown keys: {", ".join(sorted(unknown))}')
    return value


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{label} must be a non-empty string')
    return value.strip()


def _integer(value, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f'{label} must be an integer >= {minimum}')
    return value


def _positive_number(value, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ConfigError(f'{label} must be a positive number')
    return float(value)


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    source_path: str
    data_path: str
    sidelength: int
    is_rgb: bool
    model_type: str
    model_kwargs: dict
    learning_rate: float
    total_steps: int
    log_interval: int
    batch_size: int | None
    chunk_size: int
    export_path: str | None
    seed: int

    @property
    def channels(self) -> int:
        return 3 if self.is_rgb else 1

    def resolved_dict(self, *, device: str, value_range: tuple[float, float]) -> dict:
        training = {
            'total_steps': self.total_steps,
            'log_interval': self.log_interval,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
        }
        if self.batch_size is not None:
            training['batch_size'] = self.batch_size
        output = {'chunk_size': self.chunk_size}
        if self.export_path is not None:
            output['path'] = self.export_path
        return {
            'source_path': self.source_path,
            'working_directory': str(Path.cwd()),
            'task': self.task,
            'device': device,
            'data': {
                'path': self.data_path,
                'sidelength': self.sidelength,
                'is_rgb': self.is_rgb,
                'channels': self.channels,
                'value_range': value_range,
            },
            'model': {'name': self.model_type, 'kwargs': self.model_kwargs},
            'training': training,
            'output': output,
        }


def load_experiment_config(path: str, task: str) -> ExperimentConfig:
    """Read a YAML run config and resolve its model preset before data loading."""
    if task not in ('image', 'video'):
        raise ConfigError(f'Unknown task: {task!r}')
    try:
        with Path(path).open('r', encoding='utf-8') as stream:
            document = yaml.load(stream, Loader=StrictSafeLoader)
    except OSError as exc:
        raise ConfigError(f'Cannot read config {path!r}: {exc}') from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f'Invalid YAML in {path!r}: {exc}') from exc

    root = _mapping(document, 'config', {'data', 'model', 'training', 'output'})
    data = _mapping(root['data'], 'data', {'path', 'sidelength', 'is_rgb'})
    model = _mapping(root['model'], 'model', {'name'}, {'overrides'})
    training_required = {'total_steps', 'log_interval'}
    training_optional = {'learning_rate', 'seed'}
    if task == 'video':
        training_required.add('batch_size')
    training = _mapping(root['training'], 'training', training_required, training_optional)
    output_required = {'chunk_size'} | ({'path'} if task == 'image' else set())
    output = _mapping(root['output'], 'output', output_required)

    data_path = _text(data['path'], 'data.path')
    sidelength = _integer(data['sidelength'], 'data.sidelength', 1)
    if type(data['is_rgb']) is not bool:
        raise ConfigError('data.is_rgb must be a boolean')
    is_rgb = data['is_rgb']
    model_type = _text(model['name'], 'model.name')
    overrides = model.get('overrides', {})
    if not isinstance(overrides, dict):
        raise ConfigError('model.overrides must be a mapping')
    total_steps = _integer(training['total_steps'], 'training.total_steps', 0)
    log_interval = _integer(training['log_interval'], 'training.log_interval', 1)
    seed = _integer(training.get('seed', 42), 'training.seed', 0)
    if seed >= 2 ** 32:
        raise ConfigError('training.seed must be less than 2**32')
    batch_size = (
        _integer(training['batch_size'], 'training.batch_size', 1)
        if task == 'video' else None
    )
    chunk_size = _integer(output['chunk_size'], 'output.chunk_size', 1)
    export_path = _text(output['path'], 'output.path') if task == 'image' else None
    learning_rate_override = (
        _positive_number(training['learning_rate'], 'training.learning_rate')
        if 'learning_rate' in training else None
    )

    try:
        preset = resolve_model_preset(
            model_type, task, 3 if is_rgb else 1,
            overrides=overrides, learning_rate=learning_rate_override,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    return ExperimentConfig(
        task=task,
        source_path=str(Path(path).resolve()),
        data_path=data_path,
        sidelength=sidelength,
        is_rgb=is_rgb,
        model_type=model_type,
        model_kwargs=preset.kwargs,
        learning_rate=preset.learning_rate,
        total_steps=total_steps,
        log_interval=log_interval,
        batch_size=batch_size,
        chunk_size=chunk_size,
        export_path=export_path,
        seed=seed,
    )
