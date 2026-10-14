# -*- coding: utf-8 -*-
"""
1.5-layer quasi-geostrophic model — TEDA-facing adapter.

The model itself is **not** implemented here. The physics, the Helmholtz
solver, the time integrators, the initial conditions and the documented
stationary regime all live in the external ``qg-python`` package::

    pip install pyteda[qg]

This module is the adapter that makes ``qgpython.QGModel`` satisfy the TEDA
``Model`` contract: a flat state vector, ``propagate``, variable blocks and the
localization helpers. It adds no physics of its own.

State layout
------------
The state vector is the flattened concatenation of the two physical fields
``[q, psi]``, both of shape ``(m, n)``, so the dimension is ``2 * m * n``::

    var_blocks = {'q': slice(0, m*n), 'psi': slice(m*n, 2*m*n)}

Those names are what ``r={'q': 2, 'psi': 4}`` refers to. ``psi`` is a
diagnostic of ``q`` through the Helmholtz inverse; it travels in the vector for
convenience and is recomputed from ``q`` at the start of every propagation, so
an analysis that touches only the ``psi`` block has no effect on the forecast.

Localization radius
-------------------
The radius ``r`` accepted by ``get_ngb``, ``get_pre`` and
``create_decorrelation_matrix`` may be:

* ``int`` / ``float`` — one radius for every component (legacy);
* ``dict``            — one per block, e.g. ``{'q': 2, 'psi': 4}``;
* ``np.ndarray``      — one per component (length ``2*m*n``).

See ``pyteda.models._localization``.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .model import Model
from ._localization import resolve_radius, pairwise_radius, RadiusSpec

_QG_HINT = (
    "QGModel needs the 'qg-python' package, which is not installed.\n"
    "    pip install pyteda[qg]\n"
    "qg-python provides the quasi-geostrophic model; pyteda only adapts it to "
    "the TEDA Model interface."
)

try:  # pragma: no cover - exercised by the import test
    import qgpython as _qg
    from qgpython import api as _api
    from qgpython import presets as _presets
    from qgpython.model import QGModel as _QGCore, QGParams
except ImportError as exc:  # pragma: no cover
    raise ImportError(_QG_HINT) from exc


#: Time-stepping schemes offered by qg-python.
_SCHEMES = ("order1", "order2", "rk4", "dp5", "leapfrog")

#: Initial conditions offered by qg-python.
_ICS = ("zero", "fourier", "vortex", "restart")

#: Arguments the vendored model accepted and qg-python does not provide.
_DROPPED = {
    "bc": "qg-python implements the Dirichlet basin only",
    "raw_beta": "qg-python uses the standard RAW beta",
}


class QGModel(Model):
    """1.5-layer QG model, adapted to the TEDA Model interface.

    Parameters
    ----------
    preset : str or None
        ``'sakov_oke_stationary'`` for the regime used in data-assimilation
        testbeds (``rkh2 = 1e-11``, RK4, ``dt = 0.3125``), which is the one
        whose statistics are documented; ``'qgc_default'`` for the parameters
        of the original QG-C model; ``None`` to build from the arguments
        below. Default ``'sakov_oke_stationary'``.
    mrefin, nx1, ny1, lx : grid parameters.
    rkb, rkh, rkh2, f, r, a, k : physics parameters.
    scheme : str
        One of :data:`_SCHEMES`. Default ``'rk4'``.
    dt : float
        Time step.
    ra_alpha, raw_filter : Robert-Asselin filter options for ``leapfrog``.
    ic_kind : str
        Initial condition, one of :data:`_ICS`. Used by
        ``get_initial_condition``.
    ic_kwargs : dict, optional
        Keyword arguments passed to the initial condition.
    verbose : bool
        Print solver-build progress.

    Notes
    -----
    Unlike the shallow-water model, a point perturbation in this flow does not
    advect appreciably, so flow-dependent localization structures gain nothing
    here; it is the companion case to ``SWEModel``, where they do.
    """

    def __init__(
        self,
        preset: Optional[str] = "sakov_oke_stationary",
        mrefin: int = 5,
        nx1: int = 3,
        ny1: int = 3,
        lx: float = 1.0,
        rkb: float = 3.0e-4,
        rkh: float = -5.0e-8,
        rkh2: float = 1.0e-12,
        f: float = 1600.0,
        r: float = 1.0e-5,
        a: float = 6.29,
        k: float = 1.0,
        scheme: str = "rk4",
        dt: float = 1.0,
        ra_alpha: float = 0.1,
        raw_filter: bool = False,
        ic_kind: str = "fourier",
        ic_kwargs: Optional[dict] = None,
        verbose: bool = False,
        **legacy,
    ):
        # ----- arguments the vendored model had and qg-python has not -----
        for name, why in _DROPPED.items():
            if name in legacy:
                value = legacy.pop(name)
                warnings.warn(
                    f"QGModel: '{name}={value!r}' is not supported by the "
                    f"qg-python backend ({why}). The argument is ignored.",
                    stacklevel=2,
                )
        if legacy:
            raise TypeError(
                f"QGModel got unexpected keyword arguments: {sorted(legacy)}"
            )

        if scheme not in _SCHEMES:
            raise ValueError(
                f"Unknown scheme {scheme!r}. Available: {list(_SCHEMES)}."
            )

        # ----- build the qg-python params -----
        if preset == "sakov_oke_stationary":
            self.prm = _presets.sakov_oke_stationary(
                mrefin=mrefin, dt=dt, scheme=scheme, verbose=verbose
            )
        elif preset == "qgc_default":
            self.prm = _presets.qgc_default(
                mrefin=mrefin, scheme=scheme, dt=dt, verbose=verbose
            )
        elif preset is None:
            self.prm = QGParams(
                mrefin=mrefin, nx1=nx1, ny1=ny1, lx=lx,
                rkb=rkb, rkh=rkh, rkh2=rkh2, f=f, r=r, a=a, k=k,
                scheme=scheme, dt=dt,
                ra_alpha=ra_alpha, raw_filter=raw_filter,
                verbose=verbose, save_q=False,
            )
        else:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: "
                "'sakov_oke_stationary', 'qgc_default', or None."
            )
        self.preset = preset

        with warnings.catch_warnings():
            # The Helmholtz solver emits a harmless SparseEfficiencyWarning.
            warnings.simplefilter("ignore")
            self._core = _QGCore(self.prm)

        # Grid dimensions (m = rows, n = cols)
        self.m_grid = self.prm.m
        self.n_grid = self.prm.n
        self.field_size = self.m_grid * self.n_grid
        self.dim = 2 * self.field_size           # state = [q, psi]

        self.var_blocks = {
            "q": slice(0, self.field_size),
            "psi": slice(self.field_size, self.dim),
        }

        self.ic_kind = ic_kind
        self.ic_kwargs = dict(ic_kwargs) if ic_kwargs else {}
        self._L: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def list_available_integrators() -> list:
        """Time-stepping schemes offered by the qg-python backend."""
        return list(_SCHEMES)

    @staticmethod
    def list_available_ics() -> list:
        """Initial conditions offered by the qg-python backend."""
        return list(_ICS)

    # ------------------------------------------------------------------
    # Model API
    # ------------------------------------------------------------------
    def get_number_of_variables(self) -> int:
        return self.dim

    def get_initial_condition(
        self, seed: int = 0, T: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Build a flattened ``[q, psi]`` initial state.

        ``seed`` is forwarded to stochastic initial conditions such as
        ``fourier``; the ones that ignore it are unaffected. ``T``, if given,
        is a spin-up applied after the initial condition is set, which the
        stationary preset needs (``qgpython.presets.spinup_time``).
        """
        kwargs = dict(self.ic_kwargs)
        kwargs.setdefault("seed", int(seed))
        try:
            self._core.set_ic(self.ic_kind, **kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            self._core.set_ic(self.ic_kind, **kwargs)

        x0 = _api.pack(self._core.q, self._core.psi)
        if T is not None and len(T) > 1:
            x0 = self.propagate(x0, T)
        return x0

    def propagate(
        self, x0: np.ndarray, T: np.ndarray, just_final_state: bool = True
    ) -> np.ndarray:
        """Propagate ``x0 = [q; psi]`` from ``T[0]`` to ``T[-1]``.

        Only the end points of ``T`` are used; the number of steps is
        ``round((T[-1] - T[0]) / dt)``. Multistep memory is reset between
        calls, so consecutive forecasts are independent, which is what an
        ensemble needs.
        """
        if not just_final_state:
            raise NotImplementedError(
                "QGModel.propagate(..., just_final_state=False) is not "
                "supported. Use qgpython.QGModel.run() for trajectory output."
            )
        return _api.propagate(self._core, np.asarray(x0, dtype=float),
                              [float(T[0]), float(T[-1])])

    # ------------------------------------------------------------------
    # Helpers mirroring the SWE adapter
    # ------------------------------------------------------------------
    def unpack(self, x: np.ndarray):
        """Return ``(q, psi)`` as 2-D fields from a state vector."""
        return _api.unpack(self._core, np.asarray(x, dtype=float))

    def interior_mask(self) -> np.ndarray:
        """Boolean mask over one field: True off the Dirichlet boundary."""
        return _api.interior_mask(self.prm)

    def psi_from_q(self, x: np.ndarray) -> np.ndarray:
        """Recompute the ``psi`` block as the Helmholtz inverse of ``q``."""
        return _api.psi_from_q(self._core, np.asarray(x, dtype=float))

    # ------------------------------------------------------------------
    # Localization — same dispatch as Lorenz96 / SWEModel
    # ------------------------------------------------------------------
    def _resolve(self, r: RadiusSpec) -> np.ndarray:
        return resolve_radius(r, n_state=self.dim, var_blocks=self.var_blocks)

    def _index_to_grid(self, i: int):
        """Map a flat state index to ``(var, iy, ix)``; var is 0 for q, 1 for psi."""
        if i < self.field_size:
            var, j = 0, i
        else:
            var, j = 1, i - self.field_size
        iy, ix = divmod(j, self.n_grid)
        return var, iy, ix

    def get_ngb(
        self, i: int, r: RadiusSpec, cross: bool = False
    ) -> np.ndarray:
        """Cyclic 2-D neighbours of component ``i`` within radius ``r_i``.

        ``cross=True`` also includes the matching component of the other
        variable block (q <-> psi).
        """
        r_arr = self._resolve(r)
        ri = int(round(r_arr[i]))
        var, iy, ix = self._index_to_grid(i)
        m, n = self.m_grid, self.n_grid

        out = []
        for dy in range(-ri, ri + 1):
            y = (iy + dy) % m
            for dx in range(-ri, ri + 1):
                x = (ix + dx) % n
                base = y * n + x
                out.append(base + var * self.field_size)
                if cross:
                    out.append(base + (1 - var) * self.field_size)
        return np.array(sorted(set(out)), dtype=int)

    def get_pre(
        self, i: int, r: RadiusSpec, cross: bool = False
    ) -> np.ndarray:
        """Neighbours of ``i`` with a strictly smaller index (Cholesky pre-set)."""
        ngb = self.get_ngb(i, r, cross=cross)
        return ngb[ngb < i]

    def create_decorrelation_matrix(
        self,
        r: RadiusSpec,
        cross: bool = False,
        cross_scale: float = 1.0,
        combine: str = "mean",
    ) -> None:
        """Build the 2-D periodic Gaussian decorrelation matrix.

        Each entry is ``L[i, j] = scale * exp(-d_ij^2 / (2 r_ij^2))``, with
        ``d_ij`` the periodic 2-D distance, ``r_ij`` combining ``r_i`` and
        ``r_j`` per ``combine``, and ``scale`` equal to 1 within a variable
        block, ``cross_scale`` across blocks when ``cross=True``, and 0
        otherwise.
        """
        r_arr = self._resolve(r)
        m, n = self.m_grid, self.n_grid
        N = self.dim

        idx = np.arange(N)
        var = (idx >= self.field_size).astype(int)
        flat = idx % self.field_size
        iy = flat // n
        ix = flat % n

        dy = np.abs(iy[:, None] - iy[None, :])
        dy = np.minimum(dy, m - dy)
        dx = np.abs(ix[:, None] - ix[None, :])
        dx = np.minimum(dx, n - dx)
        d2 = dx ** 2 + dy ** 2

        R = pairwise_radius(r_arr, combine=combine)

        same_block = var[:, None] == var[None, :]
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
            f"QGModel(preset={self.preset!r}, "
            f"grid={self.m_grid}x{self.n_grid}, dim={self.dim}, "
            f"scheme={self.prm.scheme!r}, dt={self.prm.dt}, "
            f"backend=qg-python {_qg.__version__})"
        )
