"""Load image and video signals as flattened coordinate/value pairs."""

from dataclasses import dataclass
from math import isfinite, prod

import imageio
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

VALUE_RANGE = (-1.0, 1.0)


@dataclass(frozen=True)
class SignalData:
    """A signal with coordinates [N, D] and normalized pixels [N, C]."""

    coords: torch.Tensor
    pixels: torch.Tensor
    spatial_shape: tuple[int, int]  # height, width
    coordinate_order: tuple[str, ...]
    value_range: tuple[float, float]
    frame_count: int = 1
    frame_rate: float | None = None

    def __post_init__(self):
        if len(self.spatial_shape) != 2 or min(self.spatial_shape) <= 0:
            raise ValueError("spatial_shape must contain positive height and width")
        if (
            self.frame_count <= 0
            or len(self.value_range) != 2
            or not all(isfinite(bound) for bound in self.value_range)
            or self.value_range[0] >= self.value_range[1]
        ):
            raise ValueError("frame_count and value_range must be valid")
        if self.coordinate_order not in (("y", "x"), ("t", "y", "x")):
            raise ValueError("coordinate_order must be (y, x) or (t, y, x)")
        if self.coordinate_order == ("y", "x") and self.frame_count != 1:
            raise ValueError("image signals must have one frame")
        if self.coordinate_order == ("y", "x") and self.frame_rate is not None:
            raise ValueError("image signals must not have a frame rate")
        if self.coordinate_order[0] == "t" and (
            self.frame_rate is None
            or not isfinite(self.frame_rate)
            or self.frame_rate <= 0
        ):
            raise ValueError("video signals must have a positive frame rate")
        expected_count = prod(self.signal_shape)
        if (
            self.coords.ndim != 2
            or self.pixels.ndim != 2
            or self.coords.shape != (expected_count, len(self.coordinate_order))
            or self.pixels.shape[0] != expected_count
            or self.pixels.shape[1] not in (1, 3)
        ):
            raise ValueError("coords and pixels must match the declared signal shape")

    @property
    def signal_shape(self) -> tuple[int, ...]:
        if self.coordinate_order[0] == "t":
            return (self.frame_count, *self.spatial_shape)
        return self.spatial_shape

    @property
    def channels(self) -> int:
        return self.pixels.shape[1]

    def to_unit_range(self, values):
        """Linearly rescale values using the declared signal range."""
        lower, upper = self.value_range
        return (values - lower) / (upper - lower)


def _resized_shape(width: int, height: int, sidelength: int) -> tuple[int, int]:
    if sidelength <= 0:
        raise ValueError("sidelength must be positive")
    scale = sidelength / max(width, height)
    return max(1, int(height * scale)), max(1, int(width * scale))


def _transform(spatial_shape: tuple[int, int], channels: int):
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    return Compose(
        [
            Resize(spatial_shape),
            ToTensor(),
            Normalize((0.5,) * channels, (0.5,) * channels),
        ]
    )


def _grid(shape: tuple[int, ...]) -> torch.Tensor:
    axes = [torch.linspace(-1, 1, steps=length) for length in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        -1, len(shape)
    )


def load_image_signal(path: str, sidelength: int, channels: int = 1) -> SignalData:
    """Load one image with row-major (y, x) coordinates."""
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    with Image.open(path) as source:
        image = source.convert("RGB" if channels == 3 else "L")
        spatial_shape = _resized_shape(*image.size, sidelength)
        image_tensor = _transform(spatial_shape, channels)(image)

    pixels = image_tensor.permute(1, 2, 0).reshape(-1, channels)
    return SignalData(
        _grid(spatial_shape), pixels, spatial_shape, ("y", "x"), VALUE_RANGE
    )


def load_video_signal(path: str, sidelength: int, channels: int = 1) -> SignalData:
    """Load frames into CPU memory with row-major (t, y, x) coordinates."""
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if sidelength <= 0:
        raise ValueError("sidelength must be positive")

    reader = imageio.get_reader(path)
    frames = []
    spatial_shape = None
    transform = None
    frame_rate = None
    try:
        frame_rate = reader.get_meta_data().get("fps")
        for frame in reader:
            image = Image.fromarray(frame).convert("RGB" if channels == 3 else "L")
            if transform is None:
                spatial_shape = _resized_shape(*image.size, sidelength)
                transform = _transform(spatial_shape, channels)
            frames.append(transform(image).permute(1, 2, 0).reshape(-1, channels))
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"Video contains no frames: {path}")

    frame_count = len(frames)
    pixels = torch.cat(frames)
    return SignalData(
        _grid((frame_count, *spatial_shape)),
        pixels,
        spatial_shape,
        ("t", "y", "x"),
        VALUE_RANGE,
        frame_count=frame_count,
        frame_rate=frame_rate,
    )
