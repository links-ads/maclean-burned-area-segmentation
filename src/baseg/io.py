from pathlib import Path
import numpy as np
import rasterio as rio


def read_raster(
    path: Path,
    bands: list[int] = None,
    window: tuple[int, int, int, int] = None,
    return_profile: bool = False,
) -> np.ndarray:
    """Read a raster file using rasterio.

    Args:
        path (Path): Path to the raster file.
        bands (list[int], optional): List of bands to read. Defaults to None.
        window (tuple[int, int, int, int], optional): Window to read. Defaults to None.
        return_profile (bool, optional): Whether to return the profile. Defaults to False.

    Returns:
        np.ndarray: Raster data.
    """
    with rio.open(path) as dataset:
        options = {}
        if window is not None:
            fill_value = 0 if bands is None else 255
            options.update(window=window, boundless=True, fill_value=fill_value)
        if bands is not None:
            data = dataset.read(bands, **options)
        else:
            data = dataset.read(**options)
        if return_profile:
            return data, dataset.profile
        return data


def read_raster_profile(path: Path) -> dict:
    """Read a raster file profile using rasterio.

    Args:
        path (Path): Path to the raster file.

    Returns:
        dict: Raster profile.
    """
    with rio.open(path) as dataset:
        return dataset.profile


def write_raster(
    path: Path,
    data: np.ndarray,
    profile: dict,
    window: tuple[int, int, int, int] = None,
    bands: list[int] = None,
):
    """Write a raster file using rasterio.

    Args:
        path (Path): Path to the raster file.
        data (np.ndarray): Raster data.
        profile (dict): Raster profile.
        window (tuple[int, int, int, int], optional): Window to write. Defaults to None.
        bands (list[int], optional): List of bands to write. Defaults to None.
    """
    with rio.open(path, "w", **profile) as dataset:
        options = {}
        if window is not None:
            options.update(window=window, boundless=True)
        if bands is not None:
            options.update(bands=bands)
        dataset.write(data, **options)
