# -*- coding: utf-8 -*-
"""
Driscoll-Healy spherical grid and numerical operators.

Geometry
--------
* Grid is stored North-to-South with `Nlat = 2*(LMAX+1)` rows and
  `Nlon = 4*(LMAX+1)` columns.
* Latitudes range from +90° (row 0) down to -90° (row Nlat-1), exclusive
  of the south pole (DH convention).
* Longitudes are equally spaced 0 to 360°, exclusive of 360.

Operators
---------
* ``d_lon(q)`` — exact zonal derivative via FFT.
* ``d_lat(q)`` — centered finite-difference meridional derivative
  (with sign flip for N→S storage).
* ``sh_filter(q, alpha, p)`` — exponential filter in spherical harmonics:
  attenuates degree n by ``exp(-alpha * (n/Lmax)^p)``. Removes aliasing
  and polar noise.
* ``make_sponge(...)`` — multiplicative damping near the poles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import pyshtools as pysh


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------
@dataclass
class SphereGrid:
    """Driscoll-Healy spherical grid metadata.

    Attributes
    ----------
    LMAX : int
        Spectral truncation (T-LMAX).
    Nlat, Nlon : int
        Grid dimensions in latitude and longitude.
    a : float
        Earth radius (m).
    Omega : float
        Earth angular velocity (rad/s).
    g : float
        Gravitational acceleration (m/s^2).
    LON_ns, LAT_ns : ndarray
        Meshgrids of longitude and latitude in radians, stored N→S.
    cosL, sinL, f_cor : ndarray
        Geometric quantities (cos φ, sin φ, Coriolis parameter).
    cos_safe : ndarray
        cos φ with a small minimum value to avoid polar singularity.
    kx : ndarray
        Zonal wavenumbers for FFT.
    dlat : float
        Meridional grid spacing in radians.
    """
    LMAX: int
    Nlat: int
    Nlon: int
    a: float
    Omega: float
    g: float
    LON_ns: np.ndarray
    LAT_ns: np.ndarray
    cosL: np.ndarray
    sinL: np.ndarray
    f_cor: np.ndarray
    cos_safe: np.ndarray
    kx: np.ndarray
    dlat: float

    @classmethod
    def make(
        cls,
        LMAX: int = 32,
        a: float = 6.371e6,
        Omega: float = 7.292e-5,
        g: float = 9.81,
        cos_floor: float = 0.08,
    ) -> "SphereGrid":
        Nlat = 2 * (LMAX + 1)
        Nlon = 4 * (LMAX + 1)
        lats = 90 - np.arange(Nlat) * 180 / Nlat            # degrees, N→S
        lons = np.arange(Nlon) * 360 / Nlon
        LON_ns, LAT_ns = np.meshgrid(np.radians(lons), np.radians(lats))
        cosL = np.cos(LAT_ns)
        sinL = np.sin(LAT_ns)
        f_cor = 2 * Omega * sinL
        cos_safe = np.where(np.abs(cosL) < cos_floor,
                            np.sign(cosL + 1e-10) * cos_floor, cosL)
        kx = np.fft.rfftfreq(Nlon, d=1.0 / Nlon)
        dlat = np.radians(180 / Nlat)
        return cls(LMAX=LMAX, Nlat=Nlat, Nlon=Nlon, a=a, Omega=Omega, g=g,
                   LON_ns=LON_ns, LAT_ns=LAT_ns, cosL=cosL, sinL=sinL,
                   f_cor=f_cor, cos_safe=cos_safe, kx=kx, dlat=dlat)


# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------
def d_lon(q: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Exact zonal derivative ∂q/∂λ via FFT."""
    return np.fft.irfft(1j * grid.kx * np.fft.rfft(q, axis=1),
                        n=grid.Nlon, axis=1)


def d_lat(q: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Centered finite-difference meridional derivative ∂q/∂φ.

    The grid is stored N→S, so dq/d(row index) has opposite sign to
    dq/d(latitude) — we flip the sign.
    """
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2 * grid.dlat)
    dq[0] = (q[1] - q[0]) / grid.dlat
    dq[-1] = (q[-1] - q[-2]) / grid.dlat
    return -dq


def sh_filter(
    q: np.ndarray,
    grid: SphereGrid,
    alpha: float = 36.0,
    p: int = 16,
) -> np.ndarray:
    """Exponential filter in spherical harmonics.

    Attenuates spherical-harmonic degree ``n`` by ``exp(-alpha*(n/LMAX)^p)``.
    With ``p=16`` this is essentially a brick-wall filter that passes
    ``n < LMAX`` and cuts ``n ~ LMAX``. Removes aliasing and polar noise
    that builds up under explicit time integration.
    """
    L = grid.LMAX
    c = pysh.expand.SHExpandDH(q, norm=1, sampling=2, lmax_calc=L)
    n_arr = np.arange(L + 1)
    decay = np.exp(-alpha * (n_arr / max(L, 1)) ** p)
    c = c * decay[None, :, None]
    return pysh.expand.MakeGridDH(c, norm=1, sampling=2, lmax=L)


def make_sponge(Nlat: int, n_pole_rows: int = 10,
                strength: float = 0.98) -> np.ndarray:
    """Build a multiplicative damping mask near the poles.

    Returns an array of shape ``(Nlat, 1)`` that is < 1 within
    ``n_pole_rows`` of each pole, ramping linearly from full damping
    (``1 - strength``) at the pole to 1 at the edge of the sponge band.
    """
    sponge = np.ones((Nlat, 1))
    for i in range(n_pole_rows):
        w = (n_pole_rows - i) / n_pole_rows * strength
        sponge[i] = 1 - w
        sponge[-1 - i] = 1 - w
    return sponge
