from pathlib import Path
from typing import Callable
from baseg.io import read_raster, read_raster_profile
from baseg.samplers.utils import IndexedBounds
import numpy as np
from torch.utils.data import Dataset
from rasterio.windows import Window


class EMSImageDataset(Dataset):
    """Dataset using CEMS data. The format is the following:

    root
    ├── EMSR456
    │   ├── AOI01
    │   ├── ...
    │   ├── AOI21
    |   |   ├── EMSR456_AOI21_01
    |   |   ├── ...
    |   |   ├── EMSR456_AOI21_20
    |   |   |   ├── EMSR456_AOI21_20_01_S2L2A.tif
    |   |   |   ├── EMSR456_AOI21_20_01_S2L2A.json
    |   |   |   ├── EMSR456_AOI21_20_01_DEL.tif
    |   |   |   ├── EMSR456_AOI21_20_01_GRA.tif
    |   |   |   ├── EMSR456_AOI21_20_01_ESA_LC.tif
    |   |   |   ├── EMSR456_AOI21_20_01_CM.tif
    |   |   |
    |   |   ├── EMSR456_AOI21_01
    |   |   ├── ...
    |   |   ├── EMSR456_AOI21_20
    |   |
    |   ├── AOI22
    |   ├── ...
    |
    ├── EMSR457

    """

    classes = {
        0: "trees",
        1: "shrubland",
        2: "grassland",
        3: "cropland",
        5: "built-up",
        6: "bare",
        7: "snow",
        8: "water",
        9: "wetland",
        10: "mangroves",
        11: "moss",
    }
    palette = {
        0: (0, 100, 0),
        1: (255, 187, 34),
        2: (255, 255, 76),
        3: (240, 150, 255),
        4: (255, 0, 0),
        5: (180, 180, 180),
        6: (240, 240, 240),
        7: (0, 100, 200),
        8: (0, 150, 160),
        9: (0, 207, 117),
        10: (250, 230, 160),
        255: (0, 0, 0),
    }
    bands = {
        "S2L2A": None,
        "DEL": 1,
        "GRA": 1,
        "ESA_LC": 1,
        "CM": 1,
    }
    dtypes = {
        "S2L2A": "float32",
        "DEL": "uint8",
        "GRA": "uint8",
        "ESA_LC": "uint8",
        "CM": "uint8",
    }

    def __init__(
        self,
        root: Path,
        subset: str,
        modalities: list[str] = ["S2L2A", "DEL", "ESA_LC", "CM"],
        transform: Callable = None,
        check_integrity: bool = True,
    ):
        self.root = Path(root)
        modalities = set(modalities)
        assert self.root.exists(), f"root {self.root} does not exist"
        assert subset in ["train", "val", "test"], f"subset must be one of train, val, test, got {subset}"
        assert all(modality in ["S2L2A", "DEL", "GRA", "ESA_LC", "CM"] for modality in modalities)
        assert len(modalities) > 0, "modalities must not be empty"
        assert "S2L2A" in modalities, "At least S2L2A must be present in the modalities"
        self.modalities = modalities
        self.transform = transform
        self.files = {}
        # gather all the files
        for modality in self.modalities:
            rasters = sorted((self.root / subset).glob(f"**/*_{modality}.tif"))
            assert len(rasters) > 0, f"no rasters found for modality {modality}"
            self.files[modality] = rasters
        # filter out the files that are not present in all modalities, make sure file names match
        self.files = self._filter_files(self.files)
        self.activations = [d for d in (self.root / subset).iterdir() if d.is_dir()]
        # double check that the file IDs match at the same index across modalities
        if check_integrity:
            self._check_integrity()

    def _file_id(self, file_path: Path) -> str:
        """Get the stem, minus the modality.
        From file names like EMSR456_AOI21_20_01_ESA_LC.tif, get EMSR456_AOI21_20_01.
        """
        return "_".join(file_path.stem.split("_", maxsplit=4)[:3])

    def _filter_files(self, files: dict[str, list[Path]]) -> dict[str, list[Path]]:
        """Filter out the files that are not present in all modalities, using the file IDs."""
        # get the file IDs that are present in all modalities
        intersection = set()
        for modality, modality_files in files.items():
            modality_ids = set(self._file_id(file_path) for file_path in modality_files)

            # compute the intersection of the file IDs
            if len(intersection) == 0:
                intersection = modality_ids
            else:
                intersection = intersection.intersection(modality_ids)
        assert len(intersection) > 0, "no common file IDs found"

        # filter out the modalities that do not have the same file IDs
        filtered = {}
        for modality, modality_files in files.items():
            filtered[modality] = sorted(
                file_path for file_path in modality_files if self._file_id(file_path) in intersection
            )
            assert len(filtered[modality]) > 0, f"no files found for modality {modality}"

        return filtered

    def _check_integrity(self):
        """Double check that the file IDs match at the same index across modalities."""
        lengths = [len(modality_files) for modality_files in self.files.values()]
        assert all(length == lengths[0] for length in lengths), "modalities have different number of files"
        # zip the files together
        zipped = zip(*self.files.values())
        for idx, file_paths in enumerate(zipped):
            # assert all file IDs match
            file_ids = [self._file_id(file_path) for file_path in file_paths]
            assert all(
                file_id == file_ids[0] for file_id in file_ids
            ), f"file IDs do not match at index {idx}: {file_ids}"

    def image_shapes(self) -> list[tuple[int, int]]:
        """Get the shapes of all the images in the dataset."""
        shapes = []
        for files in self.files["S2L2A"]:
            profile = read_raster_profile(files)
            shapes.append((profile["width"], profile["height"]))
        return shapes

    def _preprocess(self, sample: dict) -> dict:
        sample["image"] = np.clip(sample.pop("S2L2A").transpose(1, 2, 0), 0, 1)
        sample["mask"] = sample.pop("DEL")
        return sample

    def _postprocess(self, sample: dict) -> dict:
        sample["S2L2A"] = sample.pop("image")
        sample["DEL"] = sample.pop("mask")
        return sample

    def __len__(self) -> int:
        return len(self.files["S2L2A"])

    def __getitem__(self, idx: int) -> dict:
        sample = {}
        metadata = dict(idx=idx)
        for modality, modality_files in self.files.items():
            file_path = modality_files[idx]
            sample[modality] = read_raster(file_path, bands=self.bands[modality]).astype(self.dtypes[modality])
            metadata[modality] = str(file_path)
        if self.transform:
            sample = self._postprocess(self.transform(**self._preprocess(sample)))
        sample["metadata"] = metadata
        return sample


class EMSCropDataset(EMSImageDataset):
    def __getitem__(self, bounds: IndexedBounds) -> dict:
        idx, coords = bounds.index, bounds.coords
        sample = {}
        metadata = dict(idx=idx, coords=coords)
        for modality, modality_files in self.files.items():
            file_path = modality_files[idx]
            sample[modality] = read_raster(
                file_path,
                bands=self.bands[modality],
                window=Window(*coords),
            )
            metadata[modality] = str(file_path)
        if self.transform:
            sample = self._postprocess(self.transform(**self._preprocess(sample)))
        sample["metadata"] = metadata
        return sample
