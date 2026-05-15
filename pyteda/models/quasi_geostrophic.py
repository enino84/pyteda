# -*- coding: utf-8 -*-
"""
QGModel — TEDA-facing wrapper around the vendored 1.5-layer QG core.

The underlying physics, integrators (Euler, RK4, DP5, AB2/3/4, SSPRK3,
Leapfrog+RA, Leapfrog+RAW), boundary conditions (``dirichlet``,
``channel``), and initial-condition library are vendored under
``pyteda.models.qg`` from the ``qg-integrators`` SoftwareX package.

State layout
------------
The state vector is the flattened concatenation of the two physical
fields ``[q, psi]``, both of shape ``(m, n)``. Total state dimension is
``2 * m * n``. The variable blocks are::

    var_blocks = {'q': slice(0, m*n), 'psi': slice(m*n, 2*m*n)}

These names are what `r={'q': 2, 'psi': 4}` refers to.

Localization radius
-------------------
The localization radius `r` accepted by `get_ngb`, `get_pre`, and
`create_decorrelation_matrix` may be:

* ``int`` / ``float`` — single radius for every component (legacy).
* ``dict``            — per block, e.g. ``{'q': 2, 'psi': 4}``.
* ``np.ndarray``      — per component (length ``2*m*n``).

See ``pyteda.models._localization`` for details.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .model import Model
from ._localization import resolve_radius, pairwise_radius, RadiusSpec

with warnings.catch_warnings():
    # The vendored Helmholtz solver triggers a SparseEfficiencyWarning on
    # build that is harmless but noisy.
    warnings.simplefilter("ignore", category=Warning)
    from .qg._qg_core import QGModel as _QGCore, QGParams
from .qg.integrators import list_integrators
from .qg.initial_conditions import list_ics


class QGModel(Model):
    """1.5-layer QG model with pluggable integrators and BCs.

    Parameters
    ----------
    mrefin, nx1, ny1, lx : grid parameters (see QGParams).
    rkb, rkh, rkh2, f, r, a, k : physics parameters.
    bc : {'dirichlet', 'channel'}.
    scheme : str
        Time integrator. One of ``list_available_integrators()``.
    dt : float
        Time step.
    ra_alpha, raw_filter, raw_beta : leapfrog filter options.
    ic_kind : str
        Initial-condition kind. One of ``list_available_ics()``. Used by
        ``get_initial_condition``.
    ic_kwargs : dict, optional
        Keyword arguments passed to the IC constructor.
    verbose : bool
        If True, the underlying core prints solver-build progress.
    """

    def __init__(
        self,
        mrefin: int = 7,
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
        bc: str = "dirichlet",
        scheme: str = "rk4",
        dt: float = 1.0,
        ra_alpha: float = 0.1,
        raw_filter: bool = False,
        raw_beta: float = 0.5,
        ic_kind: str = "fourier",
        ic_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ):
        # Build the core QG params and instance
        self.prm = QGParams(
            mrefin=mrefin, nx1=nx1, ny1=ny1, lx=lx,
            rkb=rkb, rkh=rkh, rkh2=rkh2, f=f, r=r, a=a, k=k,
            bc=bc, scheme=scheme, dt=dt,
            ra_alpha=ra_alpha, raw_filter=raw_filter, raw_beta=raw_beta,
            verbose=verbose,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._core = _QGCore(self.prm)

        # Grid dimensions (m=rows, n=cols, both in physical-grid sense)
        self.m_grid = self.prm.m  # rows (y)
        self.n_grid = self.prm.n  # cols (x)
        self.field_size = self.m_grid * self.n_grid
        self.dim = 2 * self.field_size  # state = [q, psi]

        # Variable blocks for resolve_radius dispatch
        self.var_blocks = {
            "q":   slice(0, self.field_size),
            "psi": slice(self.field_size, self.dim),
        }

        # IC config
        self.ic_kind = ic_kind
        self.ic_kwargs = dict(ic_kwargs) if ic_kwargs else {}

        # Decorrelation matrix cache
        self._L: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Static helpers — exposed for users
    # ------------------------------------------------------------------
    @staticmethod
    def list_available_integrators() -> list:
        """Return the names of all registered time-integration schemes."""
        return list_integrators()

    @staticmethod
    def list_available_ics() -> list:
        """Return the names of all available initial conditions."""
        return list_ics()

    # ------------------------------------------------------------------
    # Model API
    # ------------------------------------------------------------------
    def get_number_of_variables(self) -> int:
        return self.dim

    def get_initial_condition(self, seed: int = 0,
                              T: Optional[np.ndarray] = None) -> np.ndarray:
        """Build a flattened ``[q, psi]`` initial state.

        ``seed`` is forwarded to stochastic ICs (e.g. ``band_noise``); ICs
        that don't use it ignore it. ``T`` (if given) is a propagation
        time vector applied AFTER setting the IC, useful as a spinup.
        """
        kwargs = dict(self.ic_kwargs)
        # Common stochastic ICs accept a 'seed' kw; only inject if absent.
        kwargs.setdefault("seed", int(seed))
        try:
            self._core.set_ic(self.ic_kind, **kwargs)
        except TypeError:
            # IC doesn't take 'seed' — drop it and retry.
            kwargs.pop("seed", None)
            self._core.set_ic(self.ic_kind, **kwargs)

        x0 = np.concatenate(
            [self._core.q.ravel(), self._core.psi.ravel()]
        )
        if T is not None and len(T) > 1:
            x0 = self.propagate(x0, T)
        return x0

    def propagate(self, x0: np.ndarray, T: np.ndarray,
                  just_final_state: bool = True) -> np.ndarray:
        """Propagate state ``x0 = [q; psi]`` over the time vector ``T``.

        Only ``T[0]`` and ``T[-1]`` are used (start and end times). The
        time step is ``self.prm.dt``; the number of integrator calls is
        ``round((T[-1] - T[0]) / dt)``.
        """
        q = x0[:self.field_size].reshape(self.m_grid, self.n_grid).copy()

        # Reset core state to (q, t=T[0])
        self._core.q = q
        self._core.psi = self._core._calc_psi(q)
        self._core.t = float(T[0])
        # Multistep / leapfrog integrators must be reset between calls.
        self._core._integrator.reset()

        t_end = float(T[-1])
        dt = self.prm.dt
        n_steps = max(1, int(round((t_end - self._core.t) / dt)))

        for _ in range(n_steps):
            self._core._do_step()

        x_end = np.concatenate(
            [self._core.q.ravel(), self._core.psi.ravel()]
        )

        if just_final_state:
            return x_end
        # The core does not store intermediate states by default; for full
        # trajectory storage, use the core's run() API directly.
        raise NotImplementedError(
            "QGModel.propagate(..., just_final_state=False) is not supported. "
            "Use the underlying core.run() if you need full trajectory output."
        )

    # ------------------------------------------------------------------
    # Localization helpers — same r-dispatch as Lorenz96
    # ------------------------------------------------------------------
    def _resolve(self, r: RadiusSpec) -> np.ndarray:
        return resolve_radius(r, n_state=self.dim, var_blocks=self.var_blocks)

    def _index_to_grid(self, i: int):
        """Map flat state index ``i`` to ``(var, iy, ix)``.

        ``var`` is 0 for q and 1 for psi.
        """
        if i < self.field_size:
            var, j = 0, i
        else:
            var, j = 1, i - self.field_size
        iy, ix = divmod(j, self.n_grid)
        return var, iy, ix

    def get_ngb(
        self,
        i: int,
        r: RadiusSpec,
        cross: bool = False,
    ) -> np.ndarray:
        """Cyclic 2-D neighbours of state component ``i`` within radius ``r_i``.

        Parameters
        ----------
        i : int
            State index.
        r : RadiusSpec
            Localization radius.
        cross : bool
            If True, also include the corresponding component in the
            *other* variable block (q <-> psi).
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
        self,
        i: int,
        r: RadiusSpec,
        cross: bool = False,
    ) -> np.ndarray:
        """Neighbours of ``i`` with index strictly less than ``i`` (Cholesky pre-set)."""
        ngb = self.get_ngb(i, r, cross=cross)
        return ngb[ngb < i]

    def create_decorrelation_matrix(
        self,
        r: RadiusSpec,
        cross: bool = False,
        cross_scale: float = 1.0,
        combine: str = "mean",
    ) -> None:
        """Build the 2-D periodic gaussian decorrelation matrix.

        For each pair (i, j) the entry is::

            L[i, j] = scale * exp(-d_ij^2 / (2 * r_ij^2))

        where ``d_ij`` is the periodic 2-D distance, ``r_ij`` combines
        ``r_i`` and ``r_j`` per ``combine``, and ``scale`` is 1 if i and
        j are in the same variable block, ``cross_scale`` if they are in
        different blocks and ``cross=True``, or 0 otherwise.
        """
        r_arr = self._resolve(r)
        m, n = self.m_grid, self.n_grid
        N = self.dim

        # Pre-compute grid coordinates per state index, vectorized.
        idx = np.arange(N)
        var = (idx >= self.field_size).astype(int)         # 0 or 1
        flat = idx % self.field_size
        iy = flat // n
        ix = flat % n

        # Pairwise periodic distances (vectorized, dense).
        dy = np.abs(iy[:, None] - iy[None, :])
        dy = np.minimum(dy, m - dy)
        dx = np.abs(ix[:, None] - ix[None, :])
        dx = np.minimum(dx, n - dx)
        d2 = dx ** 2 + dy ** 2

        R = pairwise_radius(r_arr, combine=combine)

        # Same-block scale, cross-block scale.
        same_block = (var[:, None] == var[None, :])
        if cross:
            scale = np.where(same_block, 1.0, cross_scale)
        else:
            scale = same_block.astype(float)

        L = scale * np.exp(-d2 / (2.0 * R ** 2))
        self._L = L

    def get_decorrelation_matrix(self) -> np.ndarray:
        if self._L is None:
            raise RuntimeError(
                "Decorrelation matrix not built. "
                "Call create_decorrelation_matrix(r) first."
            )
        return self._L
