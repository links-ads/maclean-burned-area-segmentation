from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class IndexedBounds:
    index: int
    coords: Tuple[int, int, int, int]


def to_tuple(value: Union[int, tuple]) -> tuple:
    """Convert a value to a tuple."""
    if isinstance(value, int):
        return (value, value)


def compute_padding(shape: Tuple[int, int], tile_size: int, stride: Optional[int] = None) -> Tuple[int, int]:
    """Compute the padding to use for a given shape and tile size."""
    stride = stride or tile_size
    width, height = shape
    pad_w = (width // stride + 1) * stride - width
    pad_h = (height // stride + 1) * stride - height
    return pad_w, pad_h


def pad_shape(shape: Tuple[int, int], padding: Union[int, tuple]) -> Tuple[int, int]:
    """Pad a shape with a given padding."""
    width, height = shape
    pad_w, pad_h = to_tuple(padding)
    new_width = width + pad_w
    new_height = height + pad_h
    return new_width, new_height
