# -*- coding: utf-8 -*-
"""
Lorenz 96 model with heterogeneous-radius localization support.

The localization radius `r` accepted by `get_ngb`, `get_pre`, and
`create_decorrelation_matrix` may be:

* ``int`` / ``float``  — a single radius for all components (legacy).
* ``dict``             — one radius per variable block. Lorenz96 has a
                        single block named ``'x'``.
* ``np.ndarray``       — one radius per state component (length ``n``).

See ``pyteda.models._localization`` for details.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from .model import Model
from ._localization import resolve_radius, pairwise_radius, RadiusSpec


class Lorenz96(Model):
    """Lorenz 96 model.

    Parameters
    ----------
    n : int, optional
        Number of variables (default 40).
    F : float, optional
        Forcing constant (default 8).
    """

    def __init__(self, n: int = 40, F: float = 8.0):
        self.n = int(n)
        self.F = float(F)
        self._L: np.ndarray | None = None
        # Single block of variables — used by resolve_radius for dict specs.
        self.var_blocks = {"x": slice(0, self.n)}

    # ------------------------------------------------------------------
    # Dynamical core
    # ------------------------------------------------------------------
    def lorenz96(self, x, t):
        n, F = self.n, self.F
        return [(x[(i + 1) % n] - x[i - 2]) * x[i - 1] - x[i] + F
                for i in range(n)]

    def get_number_of_variables(self) -> int:
        return self.n

    def get_initial_condition(self, seed: int = 10,
                              T: np.ndarray = np.arange(0, 10, 0.1)) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal(self.n)
        return self.propagate(x0, T)

    def propagate(self, x0: np.ndarray, T: np.ndarray,
                  just_final_state: bool = True) -> np.ndarray:
        x1 = odeint(self.lorenz96, x0, T)
        return x1[-1, :] if just_final_state else x1

    # ------------------------------------------------------------------
    # Localization helpers — accept r as int, dict, or ndarray
    # ------------------------------------------------------------------
    def _resolve(self, r: RadiusSpec) -> np.ndarray:
        return resolve_radius(r, n_state=self.n, var_blocks=self.var_blocks)

    def create_decorrelation_matrix(
        self,
        r: RadiusSpec,
        combine: str = "mean",
    ) -> None:
        """Build the decorrelation matrix L using the chosen ``r`` spec.

        Distance is taken modulo ``n`` (cyclic boundary). The kernel is
        gaussian: ``L[i, j] = exp(-d_ij^2 / (2 * r_ij^2))`` where
        ``r_ij`` combines ``r_i`` and ``r_j`` per ``combine``.
        """
        r_arr = self._resolve(r)
        n = self.n

        # Cyclic pairwise distances (vectorized)
        i = np.arange(n)
        diff = np.abs(i[:, None] - i[None, :])
        d = np.minimum(diff, n - diff)

        R = pairwise_radius(r_arr, combine=combine)
        L = np.exp(-(d ** 2) / (2.0 * R ** 2))
        self._L = L

    def get_decorrelation_matrix(self) -> np.ndarray:
        if self._L is None:
            raise RuntimeError(
                "Decorrelation matrix not built. "
                "Call create_decorrelation_matrix(r) first."
            )
        return self._L

    def get_ngb(self, i: int, r: RadiusSpec) -> np.ndarray:
        """Return cyclic neighbours of ``i`` within radius ``r_i``."""
        r_arr = self._resolve(r)
        ri = int(round(r_arr[i]))
        return np.arange(i - ri, i + ri + 1) % self.n

    def get_pre(self, i: int, r: RadiusSpec) -> np.ndarray:
        """Return neighbours of ``i`` with index < i (used by Cholesky-style filters)."""
        ngb = self.get_ngb(i, r)
        return ngb[ngb < i]
