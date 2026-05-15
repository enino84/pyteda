# -*- coding: utf-8 -*-
"""
Vector-invariant form of the shallow-water equations on the sphere.

State variables
---------------
* ``u(λ, φ)`` — zonal velocity (m/s)
* ``v(λ, φ)`` — meridional velocity (m/s)
* ``h(λ, φ)`` — layer thickness (m)

Equations (Hack & Jakob 1992 form)
----------------------------------
    ∂u/∂t = η v − (1 / a cosφ) ∂B/∂λ
    ∂v/∂t = −η u − (1 / a) ∂B/∂φ
    ∂h/∂t = −h δ

where:
    ζ = (1 / a cosφ) ∂v/∂λ − (1/a) ∂u/∂φ + (tanφ/a) u    (relative vorticity)
    η = ζ + f                                            (absolute vorticity)
    B = g h + ½(u² + v²)                                 (Bernoulli potential)
    δ = (1 / a cosφ) ∂u/∂λ + (1 / a cosφ) ∂(v cosφ)/∂φ   (divergence)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._grid import SphereGrid, d_lon, d_lat


def rhs(
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    grid: SphereGrid,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(du/dt, dv/dt, dh/dt)`` for the SWE system."""
    cos = grid.cos_safe
    a = grid.a

    # Relative vorticity ζ
    zeta = (d_lon(v, grid) / (a * cos)
            - d_lat(u, grid) / a
            + grid.sinL / cos * u / a)

    # Absolute vorticity η
    eta = zeta + grid.f_cor

    # Bernoulli potential
    B = grid.g * h + 0.5 * (u ** 2 + v ** 2)

    # Horizontal divergence δ
    div = (d_lon(u, grid) / (a * cos)
           + d_lat(v * grid.cosL, grid) / (a * cos))

    # Tendencies
    du = eta * v - d_lon(B, grid) / (a * cos)
    dv = -eta * u - d_lat(B, grid) / a
    dh = -h * div

    return du, dv, dh


def relative_vorticity(u: np.ndarray, v: np.ndarray,
                       grid: SphereGrid) -> np.ndarray:
    """Diagnostic: relative vorticity ζ from (u, v)."""
    cos = grid.cos_safe
    a = grid.a
    return (d_lon(v, grid) / (a * cos)
            - d_lat(u, grid) / a
            + grid.sinL / cos * u / a)


def divergence(u: np.ndarray, v: np.ndarray,
               grid: SphereGrid) -> np.ndarray:
    """Diagnostic: horizontal divergence δ from (u, v)."""
    cos = grid.cos_safe
    a = grid.a
    return (d_lon(u, grid) / (a * cos)
            + d_lat(v * grid.cosL, grid) / (a * cos))
