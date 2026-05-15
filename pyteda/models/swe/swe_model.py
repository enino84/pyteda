# -*- coding: utf-8 -*-
"""
Shallow-water equations on the sphere — TEDA-facing model.

Implements Hack & Jakob (1992) form of the SWE on a Driscoll-Healy grid,
integrated with explicit Euler stabilized by an exponential
spherical-harmonic filter (every ``filter_every`` steps) plus a polar
sponge layer. This is the same recipe used in the upstream visualization
script that this module derives from.

State vector
------------
The TEDA `state vector` is the flat concatenation of one or more of the
fields ``u``, ``v``, ``h``, in that canonical order. The set of fields
is configurable via ``state_vars``::

    SWEModel(state_vars=['u', 'v', 'h'])   # 3 fields, default
    SWEModel(state_vars=['h'])             # height only
    SWEModel(state_vars=['u', 'v'])        # winds only

For variants with fewer than 3 fields, the model keeps a *latent cache*
of the missing fields (``self._latent``) initialised by the IC.
``propagate`` reads the visible fields from the input state, fills in
the missing fields from the cache, integrates with the full system, and
returns only the requested fields. The cache is updated to the integrated
values so that subsequent calls remain consistent.

This means: you can asimilate only ``h`` (small state, altimeter-like)
while the model still integrates the full primitive equations.

Variable blocks
---------------
``var_blocks`` exposes the field layout for localization::

    {'u': slice(0, n_grid),
     'v': slice(n_grid, 2*n_grid),
     'h': slice(2*n_grid, 3*n_grid)}

(Slices adjust if some fields are absent from ``state_vars``.)

This lets you specify per-field localization radii::

    LETKF(model, r={'u': 4, 'v': 4, 'h': 6})
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

from ..model import Model
from .._localization import resolve_radius, pairwise_radius, RadiusSpec
from ._grid import SphereGrid, sh_filter, make_sponge
from ._dynamics import rhs, relative_vorticity, divergence
from ._initial_conditions import default_ic, perturbed_ic


_VALID_VARS = ("u", "v", "h")


class SWEModel(Model):
    """Shallow-water model on the sphere with configurable state vector.

    Parameters
    ----------
    LMAX : int
        Spectral truncation. T-LMAX → grid is ``2(L+1) × 4(L+1)``.
        Default 32 (T32, ~400 km horizontal resolution).
    dt : float
        Integration time step in seconds. Default 120.
    state_vars : list[str]
        Subset and ordering of {'u', 'v', 'h'} that defines the TEDA
        state vector. Default ``['u', 'v', 'h']``.
    U0, H0 : float
        Williamson TC2 parameters: jet speed (m/s) and base height (m).
    filter_every : int
        Apply spherical-harmonic filter every N integration steps.
    filter_alpha, filter_p : float, int
        Exponential filter parameters (see ``sh_filter``).
    sponge_pole_rows : int
        Number of latitude rows near each pole within the sponge band.
    sponge_strength : float
        Sponge damping strength (0 = none, 1 = full damping at pole).
    a, Omega, g : float
        Earth radius, angular velocity, gravity.

    Notes
    -----
    The model uses ``pyshtools`` for the spherical-harmonic filter, which
    is a hard dependency. If you cannot install it on your platform,
    consider using ``Lorenz96`` or ``QGModel`` instead.
    """

    def __init__(
        self,
        LMAX: int = 32,
        dt: float = 120.0,
        state_vars: Optional[List[str]] = None,
        U0: float = 38.0,
        H0: float = 2800.0,
        filter_every: int = 3,
        filter_alpha: float = 36.0,
        filter_p: int = 16,
        sponge_pole_rows: int = 10,
        sponge_strength: float = 0.98,
        a: float = 6.371e6,
        Omega: float = 7.292e-5,
        g: float = 9.81,
    ):
        # ----- state_vars validation -----
        if state_vars is None:
            state_vars = ["u", "v", "h"]
        state_vars = [s.lower() for s in state_vars]
        bad = [s for s in state_vars if s not in _VALID_VARS]
        if bad:
            raise ValueError(
                f"Unknown state variables: {bad}. "
                f"Allowed: {list(_VALID_VARS)}."
            )
        if len(set(state_vars)) != len(state_vars):
            raise ValueError(f"state_vars has duplicates: {state_vars}")
        # Canonical ordering: u, v, h (always; user order is just a label).
        self.state_vars: List[str] = [s for s in _VALID_VARS if s in state_vars]

        # ----- grid + numerical operators -----
        self.grid = SphereGrid.make(LMAX=LMAX, a=a, Omega=Omega, g=g)
        self.dt = float(dt)
        self.U0 = float(U0)
        self.H0 = float(H0)
        self.filter_every = int(filter_every)
        self.filter_alpha = float(filter_alpha)
        self.filter_p = int(filter_p)
        self._sponge = make_sponge(
            self.grid.Nlat,
            n_pole_rows=sponge_pole_rows,
            strength=sponge_strength,
        )

        # Sizes
        self.field_size = self.grid.Nlat * self.grid.Nlon
        self.n_visible = len(self.state_vars)  # number of state variables
        self.dim = self.n_visible * self.field_size

        # var_blocks for localization dispatch
        self.var_blocks = {
            name: slice(i * self.field_size, (i + 1) * self.field_size)
            for i, name in enumerate(self.state_vars)
        }

        # Decorrelation matrix cache
        self._L: Optional[np.ndarray] = None

        # Latent cache for hidden fields. Filled at first
        # `get_initial_condition` call; refreshed inside `propagate`.
        self._latent: dict = {}

    # ------------------------------------------------------------------
    # Model API
    # ------------------------------------------------------------------
    def get_number_of_variables(self) -> int:
        return self.dim

    def get_initial_condition(self, seed: int = 0,
                              T: Optional[np.ndarray] = None) -> np.ndarray:
        """Build a TEDA-compatible initial state vector.

        For ``seed == 0`` the deterministic default IC (TC2 + Rossby
        perturbations) is used. For ``seed != 0`` a small random
        perturbation is added on top, so different seeds yield different
        ensemble members starting on the slow manifold.

        ``T`` is optional spinup; if given, the model is propagated from
        ``T[0]`` to ``T[-1]`` and the final state is returned.
        """
        if seed == 0:
            u, v, h = default_ic(self.grid, U0=self.U0, H0=self.H0)
        else:
            u, v, h = perturbed_ic(self.grid, seed=int(seed),
                                   U0=self.U0, H0=self.H0)

        # Initialise latent cache for hidden fields.
        self._latent = {"u": u, "v": v, "h": h}

        x0 = self._pack(u, v, h)

        if T is not None and len(T) > 1:
            x0 = self.propagate(x0, T)
        return x0

    def propagate(self, x0: np.ndarray, T: np.ndarray,
                  just_final_state: bool = True) -> np.ndarray:
        """Integrate from T[0] to T[-1] using ``self.dt``.

        The integrator is explicit Euler with sponge damping every step
        and a spherical-harmonic filter every ``filter_every`` steps.
        """
        if not just_final_state:
            raise NotImplementedError(
                "SWEModel.propagate(..., just_final_state=False) not supported."
            )

        u, v, h = self._unpack(x0)

        t0, t1 = float(T[0]), float(T[-1])
        n_steps = max(1, int(round((t1 - t0) / self.dt)))

        for step in range(1, n_steps + 1):
            du, dv, dh = rhs(u, v, h, self.grid)
            u = u + self.dt * du
            v = v + self.dt * dv
            h = h + self.dt * dh

            # Sponge near poles
            u = u * self._sponge
            v = v * self._sponge

            # Spectral filter
            if step % self.filter_every == 0:
                u = sh_filter(u, self.grid,
                              alpha=self.filter_alpha, p=self.filter_p)
                v = sh_filter(v, self.grid,
                              alpha=self.filter_alpha, p=self.filter_p)
                h = sh_filter(h, self.grid,
                              alpha=self.filter_alpha, p=self.filter_p)

            if not np.isfinite(h).all():
                raise RuntimeError(
                    f"SWE integration blew up at step {step}/{n_steps}. "
                    f"Try a smaller dt or stronger spectral filter."
                )

        # Update latent cache.
        self._latent["u"] = u
        self._latent["v"] = v
        self._latent["h"] = h

        return self._pack(u, v, h)

    # ------------------------------------------------------------------
    # State vector packing/unpacking — supports state_vars subset
    # ------------------------------------------------------------------
    def _pack(self, u: np.ndarray, v: np.ndarray,
              h: np.ndarray) -> np.ndarray:
        """Pack the requested fields into a flat state vector."""
        fields = {"u": u, "v": v, "h": h}
        out = np.empty(self.dim, dtype=float)
        for i, name in enumerate(self.state_vars):
            sl = slice(i * self.field_size, (i + 1) * self.field_size)
            out[sl] = fields[name].ravel()
        return out

    def _unpack(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack a flat state vector to ``(u, v, h)`` 2-D fields.

        Variables not in ``state_vars`` are read from the latent cache.
        """
        if not self._latent:
            # User called propagate before get_initial_condition.
            # Initialise the latent cache from the default IC so that
            # hidden fields have sensible values.
            u0, v0, h0 = default_ic(self.grid, U0=self.U0, H0=self.H0)
            self._latent = {"u": u0, "v": v0, "h": h0}

        Nlat, Nlon = self.grid.Nlat, self.grid.Nlon
        fields = {}
        for i, name in enumerate(self.state_vars):
            sl = slice(i * self.field_size, (i + 1) * self.field_size)
            fields[name] = x[sl].reshape(Nlat, Nlon)
        # Fill hidden fields from cache.
        for name in _VALID_VARS:
            if name not in fields:
                fields[name] = self._latent[name].copy()
        return fields["u"], fields["v"], fields["h"]

    # ------------------------------------------------------------------
    # Diagnostics — exposed for plotting / observation operators
    # ------------------------------------------------------------------
    def vorticity(self, x: np.ndarray) -> np.ndarray:
        """Return relative vorticity ζ from a state vector."""
        u, v, _ = self._unpack(x)
        return relative_vorticity(u, v, self.grid)

    def divergence(self, x: np.ndarray) -> np.ndarray:
        """Return horizontal divergence δ from a state vector."""
        u, v, _ = self._unpack(x)
        return divergence(u, v, self.grid)

    def get_field(self, x: np.ndarray, name: str) -> np.ndarray:
        """Return one of {'u', 'v', 'h'} as a 2-D field."""
        if name not in _VALID_VARS:
            raise ValueError(f"Unknown field '{name}'.")
        u, v, h = self._unpack(x)
        return {"u": u, "v": v, "h": h}[name]

    # ------------------------------------------------------------------
    # Localization — same dispatch contract as Lorenz96 / QGModel
    # ------------------------------------------------------------------
    def _resolve(self, r: RadiusSpec) -> np.ndarray:
        return resolve_radius(r, n_state=self.dim, var_blocks=self.var_blocks)

    def _index_to_grid(self, i: int) -> Tuple[int, int, int]:
        """Map flat index ``i`` to ``(field_id, iy, ix)``.

        ``field_id`` is the position of the field in ``state_vars``.
        """
        field_id = i // self.field_size
        local = i % self.field_size
        iy, ix = divmod(local, self.grid.Nlon)
        return field_id, iy, ix

    def get_ngb(
        self,
        i: int,
        r: RadiusSpec,
        cross: bool = False,
    ) -> np.ndarray:
        """Cyclic 2-D neighbours of ``i`` on the spherical grid.

        The grid is periodic in longitude only (latitude has poles), so
        neighbours wrap in x but reflect at the latitude edges.
        """
        r_arr = self._resolve(r)
        ri = int(round(r_arr[i]))
        field_id, iy, ix = self._index_to_grid(i)
        Nlat, Nlon = self.grid.Nlat, self.grid.Nlon

        out = []
        for dy in range(-ri, ri + 1):
            y = iy + dy
            if y < 0 or y >= Nlat:
                continue
            for dx in range(-ri, ri + 1):
                x = (ix + dx) % Nlon  # cyclic in longitude
                base = y * Nlon + x
                if cross:
                    for fid in range(self.n_visible):
                        out.append(base + fid * self.field_size)
                else:
                    out.append(base + field_id * self.field_size)
        return np.array(sorted(set(out)), dtype=int)

    def get_pre(self, i: int, r: RadiusSpec, cross: bool = False) -> np.ndarray:
        ngb = self.get_ngb(i, r, cross=cross)
        return ngb[ngb < i]

    def create_decorrelation_matrix(
        self,
        r: RadiusSpec,
        cross: bool = False,
        cross_scale: float = 1.0,
        combine: str = "mean",
    ) -> None:
        """2-D periodic-in-longitude gaussian decorrelation matrix.

        Latitude distance is non-cyclic; longitude is cyclic.

        Note: builds a dense n × n matrix. Not suitable for very high
        dimensional models (e.g. SWE LMAX=21 with n ~ 10^4 needs ~1 GB).
        For such cases, use full-rank shrinkage filters (EnKF-LW,
        EnKF-RBLW, EnKF-Shrinkage-Binv) which avoid this matrix entirely.
        """
        r_arr = self._resolve(r)
        Nlat, Nlon = self.grid.Nlat, self.grid.Nlon
        N = self.dim

        idx = np.arange(N)
        field_id = idx // self.field_size
        local = idx % self.field_size
        iy = local // Nlon
        ix = local % Nlon

        # Latitude distance (non-cyclic)
        dy = np.abs(iy[:, None] - iy[None, :])
        # Longitude distance (cyclic)
        dx = np.abs(ix[:, None] - ix[None, :])
        dx = np.minimum(dx, Nlon - dx)
        d2 = dx ** 2 + dy ** 2

        R = pairwise_radius(r_arr, combine=combine)

        same_block = (field_id[:, None] == field_id[None, :])
        if cross:
            scale = np.where(same_block, 1.0, cross_scale)
        else:
            scale = same_block.astype(float)

        self._L = scale * np.exp(-d2 / (2.0 * R ** 2))

    def get_decorrelation_matrix(self) -> np.ndarray:
        if self._L is None:
            raise RuntimeError(
                "Decorrelation matrix not built. "
                "Call create_decorrelation_matrix(r) first."
            )
        return self._L

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"SWEModel(LMAX={self.grid.LMAX}, "
                f"grid={self.grid.Nlat}x{self.grid.Nlon}, "
                f"state_vars={self.state_vars}, dim={self.dim}, dt={self.dt})")
