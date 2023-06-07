import numpy as np
from torch.utils.data import Dataset

from baseg.samplers.single import TiledSampler
from baseg.samplers.utils import IndexedBounds


class RandomTiledBatchSampler(TiledSampler):
    def __init__(
        self,
        dataset: Dataset,
        tile_size: int,
        batch_size: int,
        drop_last: bool = False,
        stride: int = None,
        apply_pad: bool = False,
        length: int = None,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))
        super().__init__(dataset, tile_size, stride, apply_pad)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = length or self.estimate_length()
        self.length = self.length // self.batch_size - 1 if self.drop_last else self.length // self.batch_size
        self.indices = np.random.choice(len(self.dataset), self.length, replace=True)

    def __len__(self):
        return self.length

    def estimate_length(self) -> int:
        # compute the number of tiles in each image and sum them up
        return sum(
            (width - self.tile_size + 1) * (height - self.tile_size + 1) // (self.stride**2)
            for width, height in self.image_sizes
        )

    def __iter__(self):
        # sample a random image, then sample a random tile from that image
        for i in self.indices:
            width, height = self.image_sizes[i]
            x = np.random.randint(0, width - self.tile_size + 1, self.batch_size)
            y = np.random.randint(0, height - self.tile_size + 1, self.batch_size)
            yield [
                IndexedBounds(i, (x[j], y[j], x[j] + self.tile_size, y[j] + self.tile_size))
                for j in range(self.batch_size)
            ]
