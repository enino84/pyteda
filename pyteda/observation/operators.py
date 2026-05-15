# -*- coding: utf-8 -*-
"""
Observation operators.

An observation operator h maps a state vector x in R^n to an observation
vector y in R^m. This module separates *what is observed* (the operator)
from *how it is corrupted* (the noise) and from *what is generated*
(the Observation object that bundles everything for a given assimilation
step).

Three concrete operators are provided:

* LinearSelection : selects m components from x (rows of identity).
                    This is the original behavior of TEDA.
* LinearMatrix    : a generic linear operator y = H x for any H.
* NonlinearOperator : y = h(x) for a user-supplied callable h, with an
                    optional analytical Jacobian.

All operators expose a uniform API:

    op.apply(x)          -> h(x). Works for x of shape (n,) or (n, N_ens).
    op.linearize(x)      -> Jacobian H of h at x (m, n). For linear ops
                            this is constant; for nonlinear ops it is
                            evaluated at x (analytical or finite-diff).
    op.is_linear         -> bool.
    op.dim_obs           -> m.
    op.dim_state         -> n.
    op.indices           -> ndarray of selected indices (LinearSelection
                            only). Used by local filters (LEnKF/LETKF).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class ObservationOperator(ABC):
    """Abstract base class for observation operators."""

    is_linear: bool = False

    @property
    @abstractmethod
    def dim_obs(self) -> int:
        """Number of observations m."""

    @property
    @abstractmethod
    def dim_state(self) -> int:
        """Number of state variables n."""

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Evaluate h(x).

        Parameters
        ----------
        x : ndarray
            State vector of shape (n,) or ensemble of shape (n, N_ens).

        Returns
        -------
        ndarray
            Observation vector of shape (m,) or ensemble of obs of shape
            (m, N_ens).
        """

    @abstractmethod
    def linearize(self, x: np.ndarray) -> np.ndarray:
        """Return the Jacobian dh/dx evaluated at x.

        For linear operators this is a constant matrix; the argument x is
        ignored. For nonlinear operators the Jacobian is evaluated at x.
        """


class LinearSelection(ObservationOperator):
    """Linear selection operator: y = x[indices].

    The Jacobian is a fixed matrix H whose rows are rows of the identity
    matrix selected by the index set. This recovers the original behavior
    of TEDA.

    Parameters
    ----------
    m : int
        Number of observations.
    n_state : int
        State dimension.
    indices : ndarray, optional
        Length-m sorted array of indices to observe. If None, indices are
        drawn at random (without replacement) using `rng`.
    rng : np.random.Generator, optional
        Random generator for index sampling. Required if `indices` is None.
    """

    is_linear = True

    def __init__(
        self,
        m: int,
        n_state: int,
        indices: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if m > n_state:
            raise ValueError(f"Cannot observe {m} of {n_state} state variables.")
        self._m = int(m)
        self._n = int(n_state)

        if indices is None:
            if rng is None:
                rng = np.random.default_rng()
            indices = rng.choice(n_state, size=m, replace=False)
            indices = np.sort(indices)
        else:
            indices = np.asarray(indices, dtype=int)
            if indices.size != m:
                raise ValueError(f"`indices` must have length m={m}.")
            if not np.all((indices >= 0) & (indices < n_state)):
                raise ValueError("`indices` out of range.")

        self._indices = indices
        # The matrix form H is m × n_state with exactly one 1 per row and
        # zeros elsewhere — so we store it as a sparse matrix. For large
        # state dimensions this is essential: a dense H at LMAX=32 SWE
        # (m≈18000, n≈26000) would take 3.5 GB; the sparse form is ≈150 KB.
        # Built lazily because many filters only need indices via apply().
        self._H = None    # built on demand by linearize()

    @property
    def dim_obs(self) -> int:
        return self._m

    @property
    def dim_state(self) -> int:
        return self._n

    @property
    def indices(self) -> np.ndarray:
        """Selected state indices (sorted). Used by local filters."""
        return self._indices

    def apply(self, x: np.ndarray) -> np.ndarray:
        # Works for x of shape (n,) or (n, N_ens).
        return x[self._indices] if x.ndim == 1 else x[self._indices, :]

    def linearize(self, x: np.ndarray = None):
        """Return the m × n Jacobian H as a sparse CSR matrix.

        H is a row-selection matrix (one 1 per row, zeros elsewhere). It
        supports the standard `@` operator with dense arrays so all
        downstream filter math works without modification:

            H @ x        # dense vector
            H @ X        # dense matrix (n, N_ens) -> (m, N_ens)
            H.T @ y      # dense (n,)
            H @ Pb @ H.T # dense (m, m)

        Storage cost is O(m), not O(m * n), which is essential at high
        state dimensions.
        """
        if self._H is None:
            from scipy.sparse import csr_matrix
            data = np.ones(self._m, dtype=float)
            row_ind = np.arange(self._m)
            col_ind = self._indices
            self._H = csr_matrix(
                (data, (row_ind, col_ind)),
                shape=(self._m, self._n),
            )
        return self._H


class LinearMatrix(ObservationOperator):
    """Generic linear operator y = H x.

    Parameters
    ----------
    H : ndarray
        Observation matrix of shape (m, n).
    """

    is_linear = True

    def __init__(self, H: np.ndarray):
        H = np.asarray(H)
        if H.ndim != 2:
            raise ValueError("H must be 2D.")
        self._H = H
        self._m, self._n = H.shape

    @property
    def dim_obs(self) -> int:
        return self._m

    @property
    def dim_state(self) -> int:
        return self._n

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self._H @ x

    def linearize(self, x: np.ndarray = None) -> np.ndarray:
        return self._H


class NonlinearOperator(ObservationOperator):
    """Nonlinear observation operator y = h(x).

    Parameters
    ----------
    h : callable
        Function that maps a state vector x of shape (n,) to an observation
        vector of shape (m,). It must also accept an ensemble of shape
        (n, N_ens) and return shape (m, N_ens) (apply column-wise if not
        vectorized; see `vectorized`).
    n_state : int
        State dimension n.
    dim_obs : int
        Observation dimension m.
    jacobian : callable, optional
        Function that returns the (m, n) Jacobian dh/dx at a given x. If
        None, a finite-difference approximation is used.
    vectorized : bool
        If True, h is assumed to support ensemble inputs (n, N_ens) -> (m, N_ens).
        If False, apply h column-by-column on ensembles.
    fd_eps : float
        Step size for finite-difference Jacobian when `jacobian` is None.
    """

    is_linear = False

    def __init__(
        self,
        h: Callable[[np.ndarray], np.ndarray],
        n_state: int,
        dim_obs: int,
        jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        vectorized: bool = False,
        fd_eps: float = 1e-6,
    ):
        self._h = h
        self._n = int(n_state)
        self._m = int(dim_obs)
        self._jac = jacobian
        self._vectorized = bool(vectorized)
        self._fd_eps = float(fd_eps)

    @property
    def dim_obs(self) -> int:
        return self._m

    @property
    def dim_state(self) -> int:
        return self._n

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.asarray(self._h(x))
        # Ensemble case
        if self._vectorized:
            return np.asarray(self._h(x))
        out = np.empty((self._m, x.shape[1]), dtype=float)
        for k in range(x.shape[1]):
            out[:, k] = self._h(x[:, k])
        return out

    def linearize(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian at x (m, n)."""
        if x.ndim != 1:
            # Default to ensemble mean
            x = np.mean(x, axis=1)
        if self._jac is not None:
            return np.asarray(self._jac(x))
        # Finite-difference fallback (central differences).
        n = self._n
        m = self._m
        eps = self._fd_eps
        J = np.empty((m, n), dtype=float)
        h0 = self._h(x)
        for j in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            J[:, j] = (self._h(xp) - self._h(xm)) / (2.0 * eps)
        # Suppress unused-variable warning
        del h0
        return J
