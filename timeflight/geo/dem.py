"""DEM sampling into the local ENU frame (pure numpy; reading GeoTIFFs happens in scripts/).

A DEM is a 2D height array plus a north-up geotransform (lon0, dlon, lat0, dlat) giving the
top-left corner of pixel (0, 0) and the pixel size in degrees (dlat < 0), as in GDAL/rasterio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .enu import Origin, enu_to_geodetic, geodetic_to_enu


@dataclass(frozen=True)
class Dem:
    heights: np.ndarray  # (rows, cols), meters
    lon0: float
    dlon: float
    lat0: float
    dlat: float  # negative for north-up rasters

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear height at lat/lon (pixel centers sit at +0.5 pixel)."""
        col = (np.asarray(lon) - self.lon0) / self.dlon - 0.5
        row = (np.asarray(lat) - self.lat0) / self.dlat - 0.5
        h = self.heights
        c0 = np.clip(np.floor(col).astype(int), 0, h.shape[1] - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, h.shape[0] - 2)
        fc = np.clip(col - c0, 0.0, 1.0)
        fr = np.clip(row - r0, 0.0, 1.0)
        top = h[r0, c0] * (1 - fc) + h[r0, c0 + 1] * fc
        bot = h[r0 + 1, c0] * (1 - fc) + h[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr


def _filter(a: np.ndarray, size: int, fn) -> np.ndarray:
    pad = size // 2
    p = np.pad(a, pad, mode="edge")
    return fn(sliding_window_view(p, (size, size)), axis=(-2, -1))


def grey_opening(heights: np.ndarray, size: int) -> np.ndarray:
    """Min filter then max filter. Removes bumps narrower than `size` pixels (buildings, trees)
    from a surface model while keeping broad terrain, a cheap DSM -> DTM approximation."""
    if size < 2:
        return heights.copy()
    return _filter(_filter(heights, size, np.min), size, np.max)


def terrain_grid(dem: Dem, origin: Origin, half_size_m: float, step_m: float) -> np.ndarray:
    """Terrain vertices (ny, nx, 3) in ENU meters on a square around the origin.

    The grid is regular in the local east/north plane; each vertex is moved to its true ENU
    position, so Earth curvature (about 30 m drop at 20 km) is included.
    """
    n = int(round(2 * half_size_m / step_m)) + 1
    e = np.linspace(-half_size_m, half_size_m, n)
    ee, nn = np.meshgrid(e, e)
    lat, lon, _ = enu_to_geodetic(np.stack([ee, nn, np.zeros_like(ee)], axis=-1), origin)
    return geodetic_to_enu(lat, lon, dem.sample(lat, lon), origin)
