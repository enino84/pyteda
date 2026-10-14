# -*- coding: utf-8 -*-
"""
Shallow-water equations on the sphere — TEDA-facing adapter.

The model itself is **not** implemented here. The physics, the grid, the time
integrator, the spectral filter, the polar sponge, the forcing and the
documented flow regimes all live in the external ``swesphere`` package::

    pip install pyteda[swe]

This module is the adapter that makes ``swesphere.SWEModel`` satisfy the TEDA
``Model`` contract: a flat state vector, ``propagate``, variable blocks and the
localization helpers. It adds no physics of its own.

Flow regimes
------------
``swesphere`` ships three characterized presets, each with measured statistics:

``waves``
    Solid-body rotation plus Rossby wave packets, unforced. Quasi-linear and
    slowly decaying.
``one_jet``
    A Galewsky jet at 45 N with relaxation of the zonal mean. Stationary,
    fairly regular eddies in the north.
``two_jets``
    Galewsky jets at +-45 deg with the same relaxation. Statistically
    stationary irregular turbulence in both hemispheres. This is the regime to
    use when an experiment needs a stationary turbulent flow.

Select one with ``preset=``::

    SWEModel(preset="two_jets")               # forced, stationary turbulence
    SWEModel(preset="waves")                  # unforced wave packets
    SWEModel(preset=None, U0=38.0, H0=2800.0) # bare TC2 + Rossby, no forcing

State vector
------------
The TEDA state vector is the flat concatenation of one or more of the fields
``u``, ``v``, ``h``, in that canonical order. The set of fields is configurable
via ``state_vars``::

    SWEModel(state_vars=['u', 'v', 'h'])   # 3 fields, default
    SWEModel(state_vars=['h'])             # height only (altimeter-like)
    SWEModel(state_vars=['u', 'v'])        # winds only

A strict subset is a smaller state in which the absent fields are **not
estimated**: they are held at a fixed reference (``self._reference``, set by
the initial condition) that every call reads and none modifies, so the result
does not depend on the order in which ensemble members are propagated. The
constructor warns when you ask for one.

**To assimilate ``h`` and still estimate ``u`` and ``v``, do not use a subset.**
Keep the full state and observe only the ``h`` block with an observation
operator; the filter then updates the wind through the ensemble cross
covariances, which is the multivariate behaviour this model exists to exercise::

    from pyteda.observation import LinearSelection

    model = SWEModel(preset="two_jets")                 # state = [u, v, h]
    h = model.var_blocks["h"]
    sites = np.arange(h.start, h.stop, 10)              # observe h only
    obs = LinearSelection(indices=sites)

A subset is for the different question of a genuinely smaller state, for
instance a single-trajectory altimeter-like run in which the wind is not part
of the estimation problem at all.

Variable blocks
---------------
``var_blocks`` exposes the field layout for localization::

    {'u': slice(0, n_grid),
     'v': slice(n_grid, 2*n_grid),
     'h': slice(2*n_grid, 3*n_grid)}

(The slices adjust if some fields are absent from ``state_vars``.) This is what
per-field localization radii refer to::

    LETKF(model, r={'u': 4, 'v': 4, 'h': 6})
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

from ..model import Model
from .._localization import resolve_radius, pairwise_radius, RadiusSpec

_SWESPHERE_HINT = (
    "SWEModel needs the 'swesphere' package, which is not installed.\n"
    "    pip install pyteda[swe]\n"
    "swesphere provides the shallow-water model on the sphere; pyteda only "
    "adapts it to the TEDA Model interface."
)

try:  # pragma: no cover - exercised by the import test
    import swesphere as _swe
    from swesphere import presets as _presets
    from swesphere.dynamics import relative_vorticity as _rel_vort
    from swesphere.dynamics import divergence as _divergence
except ImportError as exc:  # pragma: no cover
    raise ImportError(_SWESPHERE_HINT) from exc


_VALID_VARS = ("u", "v", "h")
_PRESETS = ("waves", "one_jet", "two_jets")


class SWEModel(Model):
    """Shallow-water model on the sphere, adapted to the TEDA Model interface.

    Parameters
    ----------
    preset : str or None
        One of ``'waves'``, ``'one_jet'``, ``'two_jets'``, or ``None`` for a
        bare unforced model started from the Rossby initial condition.
        Default ``'two_jets'``.
    LMAX : int
        Spectral truncation. The grid is ``2(L+1) x 4(L+1)``. Default 32
        (66 x 132, about 2.7 degrees).
    dt : float
        Integration time step in seconds. Default 120.
    state_vars : list[str]
        Subset and ordering of {'u', 'v', 'h'} that defines the TEDA state
        vector. Default ``['u', 'v', 'h']``.
    U0, H0 : float
        Parameters of the Rossby initial condition. Only used when
        ``preset is None``; the Galewsky presets set their own.
    filter_every : int
        Apply the spherical-harmonic filter every N steps. Note that the
        dissipation, and therefore the eddy amplitude of a regime, is set by
        the interval ``dt * filter_every`` rather than by ``dt`` alone; keep
        that product fixed if you change the time step.
    filter_alpha, filter_p : float, int
        Exponential filter parameters.
    sponge_pole_rows : int
        Number of latitude rows inside the polar sponge at each pole.
    scheme : str
        Time-stepping scheme registered in ``swesphere.integrators``.
        Default ``'rk4'``.
    a, Omega, g : float
        Earth radius, rotation rate, gravity.

    Notes
    -----
    ``swesphere`` uses ``pyshtools`` for the spherical-harmonic filter and
    installs it as a dependency. If it cannot be installed on your platform,
    use ``Lorenz96`` or ``QGModel`` instead.
    """

    def __init__(
        self,
        preset: Optional[str] = "two_jets",
        LMAX: int = 32,
        dt: float = 120.0,
        state_vars: Optional[List[str]] = None,
        U0: float = 38.0,
        H0: float = 2800.0,
        filter_every: int = 3,
        filter_alpha: float = 36.0,
        filter_p: int = 16,
        sponge_pole_rows: int = 10,
        scheme: str = "rk4",
        a: float = 6.371e6,
        Omega: float = 7.292e-5,
        g: float = 9.81,
        **legacy,
    ):
        # ----- state_vars validation -----
        if state_vars is None:
            state_vars = ["u", "v", "h"]
        state_vars = [s.lower() for s in state_vars]
        bad = [s for s in state_vars if s not in _VALID_VARS]
        if bad:
            raise ValueError(
                f"Unknown state variables: {bad}. Allowed: {list(_VALID_VARS)}."
            )
        if len(set(state_vars)) != len(state_vars):
            raise ValueError(f"state_vars has duplicates: {state_vars}")
        # Canonical ordering: u, v, h (the user's order is just a label).
        self.state_vars: List[str] = [s for s in _VALID_VARS if s in state_vars]

        # ----- parameters dropped when the vendored model was removed -----
        for name in ("sponge_strength",):
            if name in legacy:
                legacy.pop(name)
                warnings.warn(
                    f"SWEModel: '{name}' is no longer configurable; swesphere "
                    "fixes the sponge profile. The argument is ignored.",
                    stacklevel=2,
                )
        if legacy:
            raise TypeError(
                f"SWEModel got unexpected keyword arguments: {sorted(legacy)}"
            )

        if preset is not None and preset not in _PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {list(_PRESETS)}, or "
                "None for a bare unforced model."
            )
        self.preset = preset

        # ----- build the swesphere model -----
        overrides = dict(
            LMAX=LMAX,
            dt=dt,
            scheme=scheme,
            filter_every=filter_every,
            filter_alpha=filter_alpha,
            filter_p=filter_p,
            n_pole_rows=sponge_pole_rows,
            a=a,
            Omega=Omega,
            g=g,
        )
        if preset is None:
            self._core = _swe.SWEModel(**overrides)
            self._x0_core = self._core.initial_condition("rossby", U0=U0, H0=H0)
        else:
            self._core, self._x0_core = _presets.make(preset, **overrides)

        self.U0 = float(U0)
        self.H0 = float(H0)
        self.dt = float(dt)
        self.grid = self._core.grid

        # Sizes
        self.field_size = self._core.field_size
        self.n_visible = len(self.state_vars)
        self.dim = self.n_visible * self.field_size

        # var_blocks for localization dispatch
        self.var_blocks = {
            name: slice(i * self.field_size, (i + 1) * self.field_size)
            for i, name in enumerate(self.state_vars)
        }

        self._L: Optional[np.ndarray] = None

        # Fixed reference for the fields outside state_vars. Read by every
        # call, written only by get_initial_condition / set_reference_fields,
        # so a forecast never depends on the order in which ensemble members
        # were propagated.
        u0, v0, h0 = self._core.unpack(np.asarray(self._x0_core, dtype=float))
        self._reference = {"u": u0, "v": v0, "h": h0}

        if self.n_visible < len(_VALID_VARS):
            hidden = [f for f in _VALID_VARS if f not in self.state_vars]
            warnings.warn(
                f"SWEModel(state_vars={self.state_vars}): the fields "
                f"{hidden} are not part of the state, so they are not "
                "estimated; they are held at a fixed reference while the "
                "model still integrates the full system. To assimilate one "
                "field and estimate the others, keep the full state and "
                "observe only that field with an observation operator "
                "instead.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Model API
    # ------------------------------------------------------------------
    def get_number_of_variables(self) -> int:
        return self.dim

    def _perturb(self, u, v, h, seed: int, strength: float):
        """Seeded smooth perturbation of a state, for ensemble generation.

        Generating ensemble members is the assimilation library's job, not the
        model's, so this lives here and not in ``swesphere``. The perturbation
        is a sum of low zonal wavenumbers (1 to 8) with a Gaussian envelope in
        latitude, scaled to a fraction of the field's spatial standard
        deviation. It is deliberately not balanced: the model radiates the
        unbalanced part as gravity waves within a few hours, so members should
        be spun up before they are sampled.

        ``u`` and ``v`` share a single scale, taken from both components
        together. Scaling each by its own standard deviation would leave ``v``
        unperturbed in a purely zonal preset such as ``two_jets``, where
        ``v == 0`` everywhere; that member would then carry no ensemble
        spread in ``v``, and no filter could ever correct it.
        """
        rng = np.random.default_rng(int(seed))
        G = self._core.grid
        lat_mod = np.exp(-(np.asarray(G.LAT_ns) / np.radians(45.0)) ** 2)
        wind_scale = float(np.std(np.concatenate([np.ravel(u), np.ravel(v)])))
        scales = (wind_scale, wind_scale, float(np.std(h)))
        out = []
        for field, scale in zip((u, v, h), scales):
            if scale == 0.0:                      # a field with no structure
                out.append(field.copy())          # anywhere stays as it is
                continue
            pert = np.zeros_like(field)
            for k in range(1, 9):
                amp = rng.standard_normal() / k
                phase = 2.0 * np.pi * rng.random()
                pert += amp * np.cos(k * np.asarray(G.LON_ns) + phase)
            pert *= lat_mod
            pert *= strength * scale / max(float(np.std(pert)), 1e-12)
            out.append(field + pert)
        return tuple(out)

    def get_initial_condition(
        self,
        seed: int = 0,
        T: Optional[np.ndarray] = None,
        pert_strength: float = 0.05,
    ) -> np.ndarray:
        """Build a TEDA-compatible initial state vector.

        ``seed == 0`` gives the deterministic initial state of the preset.
        ``seed != 0`` adds the seeded perturbation of :meth:`_perturb`, so
        ensemble members start from different states; since that perturbation
        is unbalanced, pass ``T`` or propagate before sampling.

        ``T`` is an optional spin-up: when given, the model is propagated from
        ``T[0]`` to ``T[-1]`` and the final state is returned.
        """
        u, v, h = self._core.unpack(np.asarray(self._x0_core, dtype=float))
        if seed != 0:
            u, v, h = self._perturb(u, v, h, seed, pert_strength)

        self._reference = {"u": u, "v": v, "h": h}

        x0 = self._pack(u, v, h)
        if T is not None and len(T) > 1:
            x0 = self.propagate(x0, T)
        return x0

    def propagate(
        self, x0: np.ndarray, T: np.ndarray, just_final_state: bool = True
    ) -> np.ndarray:
        """Integrate from ``T[0]`` to ``T[-1]`` with the model's own stepper.

        ``T`` is in seconds and the running time is passed through to
        swesphere, not reset to zero, because from 0.4.0 its spectral filter
        fires on absolute model time. That is what makes a forecast window of
        any length give the same trajectory as one long integration, so the
        observation frequency of an experiment does not change the model.

        The interval must be a whole number of the backend's time steps
        (``model.dt``, 120 s by default); swesphere raises otherwise instead
        of quietly returning a state at another time. Choose an ``obs_freq``
        that is a multiple of ``dt``, or build the model with a ``dt`` that
        divides it.
        """
        if not just_final_state:
            raise NotImplementedError(
                "SWEModel.propagate(..., just_final_state=False) not supported."
            )

        u, v, h = self._unpack(x0)
        x_end = self._core.propagate(
            self._core.pack(u, v, h), [float(T[0]), float(T[-1])]
        )
        # The reference is deliberately NOT updated here: ensemble members are
        # propagated one after another on the same model object, so writing it
        # would make each member's forecast depend on the previous one.
        return self._pack(*self._core.unpack(x_end))

    # ------------------------------------------------------------------
    # State vector packing/unpacking — supports a state_vars subset
    # ------------------------------------------------------------------
    def _pack(self, u: np.ndarray, v: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Pack the requested fields into a flat state vector."""
        fields = {"u": u, "v": v, "h": h}
        out = np.empty(self.dim, dtype=float)
        for i, name in enumerate(self.state_vars):
            sl = slice(i * self.field_size, (i + 1) * self.field_size)
            out[sl] = fields[name].ravel()
        return out

    def _unpack(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack a flat state vector to ``(u, v, h)`` 2-D fields.

        Fields outside ``state_vars`` come from the fixed reference.
        """
        Nlat, Nlon = self._core.Nlat, self._core.Nlon
        fields = {}
        for i, name in enumerate(self.state_vars):
            sl = slice(i * self.field_size, (i + 1) * self.field_size)
            fields[name] = x[sl].reshape(Nlat, Nlon)
        for name in _VALID_VARS:
            if name not in fields:
                fields[name] = self._reference[name].copy()
        return fields["u"], fields["v"], fields["h"]

    def set_reference_fields(self, u=None, v=None, h=None) -> None:
        """Set the fixed reference used for fields outside ``state_vars``.

        Only needed when ``state_vars`` is a strict subset and you want to
        advance the unestimated fields deliberately, for instance to a state
        taken from a separate full-state run. It is a no-op for a full state.
        """
        for name, field in (("u", u), ("v", v), ("h", h)):
            if field is not None:
                self._reference[name] = np.asarray(field, dtype=float).reshape(
                    self._core.Nlat, self._core.Nlon
                )

    # ------------------------------------------------------------------
    # Diagnostics — for plotting and observation operators
    # ------------------------------------------------------------------
    def vorticity(self, x: np.ndarray) -> np.ndarray:
        """Relative vorticity from a state vector."""
        u, v, _ = self._unpack(x)
        return _rel_vort(u, v, self.grid)

    def divergence(self, x: np.ndarray) -> np.ndarray:
        """Horizontal divergence from a state vector."""
        u, v, _ = self._unpack(x)
        return _divergence(u, v, self.grid)

    def get_field(self, x: np.ndarray, name: str) -> np.ndarray:
        """Return one of {'u', 'v', 'h'} as a 2-D field."""
        if name not in _VALID_VARS:
            raise ValueError(f"Unknown field '{name}'.")
        u, v, h = self._unpack(x)
        return {"u": u, "v": v, "h": h}[name]

    def interior_mask(self) -> np.ndarray:
        """Boolean mask over one field: True outside the polar sponge.

        These are the rows the model is meant to be observed and scored on.
        """
        return self._core.interior_mask()

    # ------------------------------------------------------------------
    # Localization — same dispatch contract as Lorenz96 / QGModel
    # ------------------------------------------------------------------
    def _resolve(self, r: RadiusSpec) -> np.ndarray:
        return resolve_radius(r, n_state=self.dim, var_blocks=self.var_blocks)

    def _index_to_grid(self, i: int) -> Tuple[int, int, int]:
        """Map a flat index to ``(field_id, iy, ix)``."""
        field_id = i // self.field_size
        local = i % self.field_size
        iy, ix = divmod(local, self._core.Nlon)
        return field_id, iy, ix

    def get_ngb(
        self, i: int, r: RadiusSpec, cross: bool = False
    ) -> np.ndarray:
        """Neighbours of ``i`` on the spherical grid.

        The grid is periodic in longitude only, so neighbours wrap in x and
        stop at the latitude edges.
        """
        r_arr = self._resolve(r)
        ri = int(round(r_arr[i]))
        field_id, iy, ix = self._index_to_grid(i)
        Nlat, Nlon = self._core.Nlat, self._core.Nlon

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

    def get_pre(
        self, i: int, r: RadiusSpec, cross: bool = False
    ) -> np.ndarray:
        ngb = self.get_ngb(i, r, cross=cross)
        return ngb[ngb < i]

    def create_decorrelation_matrix(
        self,
        r: RadiusSpec,
        cross: bool = False,
        cross_scale: float = 1.0,
        combine: str = "mean",
    ) -> None:
        """2-D decorrelation matrix, periodic in longitude only.

        Builds a dense ``n x n`` matrix, so it is not suitable for high
        dimensions (LMAX 32 with three fields needs several GB). For those,
        use the full-rank shrinkage filters, which never form it.
        """
        r_arr = self._resolve(r)
        Nlon = self._core.Nlon
        N = self.dim

        idx = np.arange(N)
        field_id = idx // self.field_size
        local = idx % self.field_size
        iy = local // Nlon
        ix = local % Nlon

        dy = np.abs(iy[:, None] - iy[None, :])          # latitude: not cyclic
        dx = np.abs(ix[:, None] - ix[None, :])          # longitude: cyclic
        dx = np.minimum(dx, Nlon - dx)
        d2 = dx ** 2 + dy ** 2

        R = pairwise_radius(r_arr, combine=combine)

        same_block = field_id[:, None] == field_id[None, :]
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
    def __repr__(self) -> str:
        return (
            f"SWEModel(preset={self.preset!r}, LMAX={self.grid.LMAX}, "
            f"grid={self._core.Nlat}x{self._core.Nlon}, "
            f"state_vars={self.state_vars}, dim={self.dim}, dt={self.dt}, "
            f"backend=swesphere {_swe.__version__})"
        )
