"""WGS84 geodetic <-> local East-North-Up (ENU) coordinates.

ENU is the scene frame for Blender: x = east, y = north, z = up, in meters, origin on the
ground at the place's origin. All functions are vectorized over numpy arrays.

Heights: Copernicus DEM heights are orthometric (EGM2008), not ellipsoidal. Over a few tens
of kilometers the geoid undulation is almost constant, so we use orthometric heights as `h`
everywhere and the constant offset cancels out in the local frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A = 6378137.0  # WGS84 semi-major axis, m
F = 1 / 298.257223563
E2 = F * (2 - F)  # first eccentricity squared


@dataclass(frozen=True)
class Origin:
    lat: float
    lon: float
    h: float = 0.0


def geodetic_to_ecef(lat, lon, h):
    lat, lon, h = (np.asarray(v, dtype=np.float64) for v in (lat, lon, h))
    phi, lam = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    x = (n + h) * np.cos(phi) * np.cos(lam)
    y = (n + h) * np.cos(phi) * np.sin(lam)
    z = (n * (1 - E2) + h) * np.sin(phi)
    return x, y, z


def ecef_to_geodetic(x, y, z, iterations: int = 5):
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    lam = np.arctan2(y, x)
    p = np.hypot(x, y)
    phi = np.arctan2(z, p * (1 - E2))
    for _ in range(iterations):
        n = A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
        h = p / np.cos(phi) - n
        phi = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    h = p / np.cos(phi) - n
    return np.degrees(phi), np.degrees(lam), h


def _rotation(origin: Origin) -> np.ndarray:
    phi, lam = np.radians(origin.lat), np.radians(origin.lon)
    sp, cp, sl, cl = np.sin(phi), np.cos(phi), np.sin(lam), np.cos(lam)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def geodetic_to_enu(lat, lon, h, origin: Origin) -> np.ndarray:
    """Returns an array of shape (..., 3) with east, north, up in meters."""
    x, y, z = geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = geodetic_to_ecef(origin.lat, origin.lon, origin.h)
    d = np.stack([x - x0, y - y0, z - z0], axis=-1)
    return d @ _rotation(origin).T


def enu_to_geodetic(enu, origin: Origin):
    """Inverse of geodetic_to_enu. `enu` has shape (..., 3). Returns (lat, lon, h)."""
    enu = np.asarray(enu, dtype=np.float64)
    d = enu @ _rotation(origin)
    x0, y0, z0 = geodetic_to_ecef(origin.lat, origin.lon, origin.h)
    return ecef_to_geodetic(d[..., 0] + x0, d[..., 1] + y0, d[..., 2] + z0)
