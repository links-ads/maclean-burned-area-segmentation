from abc import ABC, abstractmethod
from typing import Callable, Generator

import numpy as np
import torch

from baseg.tiling.functional import predict_smooth_windowing


class Tiler(ABC):
    """Generic tiling operator"""

    def __init__(self, tile_size: int, channels_first: bool) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.channels_first = channels_first

    @abstractmethod
    def __call__(self, image: np.ndarray) -> Generator[tuple, None, None]:
        return NotImplementedError("Implement in subclass")


class SingleImageTiler(Tiler):
    """'Fake' tiling operator that returns the coordinates for the full image.
    Used to generate the test set and avoid overlapping pixels (tiling will be done at test time).
    """

    def __call__(self, image: np.ndarray) -> Generator[tuple, None, None]:
        if len(image.shape) == 2:
            axis = 0 if self.channels_first else -1
            image = np.expand_dims(image, axis=axis)
        if self.channels_first:
            image = np.moveaxis(image, 0, -1)
        height, width, _ = image.shape
        yield (0, 0), (0, 0, height, width)


class SmoothTiler(Tiler):
    def __init__(
        self,
        tile_size: int,
        channels_first: bool = False,
        subdivisions: int = 2,
        batch_size: int = 4,
        mirrored: bool = True,
    ) -> None:
        super().__init__(tile_size, channels_first)
        self.subdivisions = subdivisions
        self.batch_size = batch_size
        self.mirrored = mirrored

    def __call__(self, image: torch.Tensor, callback: Callable) -> torch.Tensor:
        return predict_smooth_windowing(
            image=image,
            tile_size=self.tile_size,
            subdivisions=self.subdivisions,
            prediction_fn=callback,
            batch_size=self.batch_size,
            channels_first=self.channels_first,
            mirrored=self.mirrored,
        )
