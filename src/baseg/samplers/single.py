import warnings
from abc import ABC

import numpy as np
from torch.utils.data import Dataset, Sampler

from baseg.samplers.utils import IndexedBounds


class FullImageSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        assert hasattr(
            dataset, "image_shapes"
        ), "To use samplers, the dataset must have an `image_shapes` method implementation."
        self.dataset = dataset
        self.shapes = dataset.image_shapes()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, image_size in enumerate(self.shapes):
            yield IndexedBounds(i, (0, 0, *image_size))


class TiledSampler(Sampler, ABC):
    def __init__(
        self,
        dataset: Dataset,
        tile_size: int,
        stride: int = None,
        length: int = None,
    ):
        assert hasattr(
            dataset, "image_shapes"
        ), "To use samplers, the dataset must have an `image_shapes` method implementation."
        self.dataset = dataset
        self.shapes = dataset.image_shapes()
        self.tile_size = tile_size
        self.stride = stride or tile_size
        self.length = length


class SequentialTiledSampler(TiledSampler):
    def __iter__(self):
        for i, image_size in enumerate(self.shapes):
            width, height = image_size
            for x in range(0, width - self.tile_size + 1, self.stride):
                for y in range(0, height - self.tile_size + 1, self.stride):
                    maxx, maxy = x + self.tile_size, y + self.tile_size
                    if maxx > width or maxy > height:
                        warnings.warn(
                            f"Tile at ({x}, {y}) is out of bounds for image {i}, you may want to pad the image."
                        )
                        continue
                    yield IndexedBounds(i, (x, y, maxx, maxy))

    def __len__(self):
        # count the tiles generated
        if self.length:
            return self.length
        return sum(
            (width - self.tile_size + 1) * (height - self.tile_size + 1) // (self.stride**2)
            for width, height in self.shapes
        )


class RandomTiledSampler(TiledSampler):
    def __init__(
        self,
        dataset: Dataset,
        tile_size: int,
        stride: int = None,
        apply_pad: bool = False,
        length: int = None,
    ):
        super().__init__(dataset, tile_size, stride, apply_pad)
        self.length = length or self.estimate_length()
        self.indices = np.random.choice(len(self.dataset), self.length, replace=True)

    def estimate_length(self) -> int:
        # compute the number of tiles in each image and sum them up
        return sum(
            (width - self.tile_size + 1) * (height - self.tile_size + 1) // (self.stride**2)
            for width, height in self.shapes
        )

    def __iter__(self):
        # sample a random image, then sample a random tile from that image
        for i in self.indices:
            width, height = self.shapes[i]
            x = np.random.randint(0, width - self.tile_size + 1, 1)
            y = np.random.randint(0, height - self.tile_size + 1, 1)
            yield IndexedBounds(i, (x, y, self.tile_size, self.tile_size))
