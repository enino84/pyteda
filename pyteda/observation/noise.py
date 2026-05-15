# -*- coding: utf-8 -*-
"""
Observation noise models.

Each noise model defines the data error covariance R and provides:

    noise.R              -> (m, m) covariance matrix
    noise.R_inv          -> (m, m) precision matrix (inverse of R)
    noise.sample(rng)    -> draw a noise vector of shape (m,)
    noise.sample_many(N, rng) -> draw N noise vectors, shape (m, N)

Three families are provided:

* IsotropicDiagonal     : R = sigma^2 * I  (the original behavior).
* HeterogeneousDiagonal : R = diag(sigma_1^2, ..., sigma_m^2).
* BlockDiagonal         : per-variable-block isotropic noise.
* DenseCovariance       : an arbitrary symmetric positive-definite R.

For diagonal R variants, R and R_inv switch automatically between dense
ndarrays (small dim) and scipy.sparse CSR matrices (dim >= 500) — see
SPARSE_THRESHOLD. Filters interact with both seamlessly via the ``@``
operator. R_diag / R_inv_diag give the diagonal as a 1-D vector for
filters that don't want a matrix at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


SPARSE_THRESHOLD = 500
"""Observation dim at which we switch from dense to scipy.sparse storage.

Below this threshold, dense numpy is faster despite touching every
matrix entry. Above it, scipy.sparse becomes essential to avoid
materialising matrices that would consume gigabytes.
"""


def _diag(values: np.ndarray):
    """Build a diagonal matrix in the right format for its length.

    Returns dense ``np.diag`` when ``len(values) < SPARSE_THRESHOLD``,
    or scipy.sparse CSR otherwise. The result supports the standard
    ``@`` operator with dense vectors / matrices in both cases.
    """
    values = np.asarray(values, dtype=float)
    if values.size < SPARSE_THRESHOLD:
        return np.diag(values)
    from scipy.sparse import diags
    return diags(values, format="csr")


class ObservationNoise(ABC):
    """Abstract base class for observation noise models."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Observation dimension m."""

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Data error covariance, shape (m, m)."""

    @property
    @abstractmethod
    def R_inv(self) -> np.ndarray:
        """Precision matrix, shape (m, m)."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one noise vector of shape (m,)."""

    def sample_many(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n noise vectors stacked column-wise, shape (m, n)."""
        return np.stack([self.sample(rng) for _ in range(n)], axis=1)


class IsotropicDiagonal(ObservationNoise):
    """R = std**2 * I_m. The original TEDA noise model.

    Parameters
    ----------
    std : float
        Common standard deviation across all observations.
    dim : int, optional
        Observation dimension m. May be omitted at construction time and
        bound later by Observation; if omitted, attempts to access R raise.
    """

    def __init__(self, std: float, dim: Optional[int] = None):
        self._std = float(std)
        self._dim = dim

    def bind_dim(self, m: int):
        """Bind the dimension if it was unknown at construction time."""
        self._dim = int(m)

    def _require_dim(self):
        if self._dim is None:
            raise RuntimeError(
                "IsotropicDiagonal needs `dim` either at construction or via bind_dim()."
            )

    @property
    def dim(self) -> int:
        self._require_dim()
        return self._dim

    @property
    def R(self):
        """Covariance matrix R = σ² I.

        Returns dense ``np.eye * σ²`` when dim is small, or sparse CSR
        when dim is large. Supports ``R @ x``, ``R @ X`` either way.
        """
        self._require_dim()
        return _diag(np.full(self._dim, self._std ** 2))

    @property
    def R_diag(self) -> np.ndarray:
        """The diagonal of R as a 1-D ndarray (cheap regardless of size)."""
        self._require_dim()
        return np.full(self._dim, self._std ** 2)

    @property
    def R_inv(self):
        """Inverse covariance R⁻¹ = (1/σ²) I; dense or sparse like R."""
        self._require_dim()
        return _diag(np.full(self._dim, 1.0 / self._std ** 2))

    @property
    def R_inv_diag(self) -> np.ndarray:
        """The diagonal of R⁻¹ as a 1-D ndarray."""
        self._require_dim()
        return np.full(self._dim, 1.0 / self._std ** 2)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        self._require_dim()
        return self._std * rng.standard_normal(self._dim)

    def sample_many(self, n: int, rng: np.random.Generator) -> np.ndarray:
        self._require_dim()
        return self._std * rng.standard_normal((self._dim, n))

    def sample_many_legacy(self, n: int) -> np.ndarray:
        """Like sample_many() but uses numpy's GLOBAL RNG (np.random.*),
        which is what filters need so Simulation's np.random.seed(...) call
        keeps results reproducible across runs."""
        self._require_dim()
        return self._std * np.random.standard_normal((self._dim, n))


class HeterogeneousDiagonal(ObservationNoise):
    """R = diag(stds**2). Per-component standard deviations."""

    def __init__(self, stds: np.ndarray):
        stds = np.asarray(stds, dtype=float)
        if stds.ndim != 1:
            raise ValueError("`stds` must be 1D.")
        if np.any(stds <= 0):
            raise ValueError("All standard deviations must be positive.")
        self._stds = stds
        self._dim = stds.size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def R(self):
        return _diag(self._stds ** 2)

    @property
    def R_diag(self) -> np.ndarray:
        return self._stds ** 2

    @property
    def R_inv(self):
        return _diag(1.0 / self._stds ** 2)

    @property
    def R_inv_diag(self) -> np.ndarray:
        return 1.0 / self._stds ** 2

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self._stds * rng.standard_normal(self._dim)

    def sample_many(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._stds[:, None] * rng.standard_normal((self._dim, n))

    def sample_many_legacy(self, n: int) -> np.ndarray:
        """Same as sample_many() but uses numpy's global RNG."""
        return self._stds[:, None] * np.random.standard_normal((self._dim, n))


class BlockDiagonal(HeterogeneousDiagonal):
    """Per-variable-block isotropic noise — the realistic case.

    For multi-variable models (SWE with [u, v, h], QG with [q, psi]),
    observation noise is typically a single std *per physical variable*,
    not heterogeneous per-component. This class lets you write that
    naturally::

        noise = BlockDiagonal(
            stds_per_block={"u": 1.0, "v": 1.0, "h": 5.0},
            var_blocks=model.var_blocks,    # {"u": slice(0, m_u), ...}
        )

    Internally it concatenates the per-block stds into a single 1-D
    array and reuses the (efficient, sparse-aware) HeterogeneousDiagonal
    machinery — so R is a sparse CSR for high-dim setups, dense for
    small ones, with no extra plumbing.

    The ``var_blocks`` mapping must give the slice (relative to the
    *observation* vector) for each named block. For a typical setup
    where you observe a fixed selection of state components and the
    selection respects the variable order of the model, you can build
    these slices once and pass them.
    """

    def __init__(
        self,
        stds_per_block: Dict[str, float],
        var_blocks: Dict[str, slice],
    ):
        if not stds_per_block:
            raise ValueError("stds_per_block must not be empty.")
        if set(stds_per_block.keys()) != set(var_blocks.keys()):
            raise ValueError(
                "stds_per_block and var_blocks must reference the same "
                f"block names. Got stds={list(stds_per_block)}, "
                f"blocks={list(var_blocks)}."
            )
        # Build the full per-component stds vector by laying down each
        # block's std into its slice.
        # Determine the total length from the largest block.stop.
        total = max(sl.stop for sl in var_blocks.values())
        stds = np.empty(total, dtype=float)
        for name, sl in var_blocks.items():
            std = float(stds_per_block[name])
            if std <= 0:
                raise ValueError(
                    f"std for block '{name}' must be positive (got {std})."
                )
            stds[sl] = std
        super().__init__(stds=stds)
        self._stds_per_block = dict(stds_per_block)
        self._var_blocks = dict(var_blocks)

    @property
    def stds_per_block(self) -> Dict[str, float]:
        return dict(self._stds_per_block)

    @property
    def var_blocks(self) -> Dict[str, slice]:
        return dict(self._var_blocks)


class DenseCovariance(ObservationNoise):
    """R is an arbitrary symmetric positive-definite covariance matrix.

    Sampling uses the Cholesky factor of R; the precision is computed via
    Cholesky solve.
    """

    def __init__(self, R: np.ndarray):
        R = np.asarray(R, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError("R must be a square 2D array.")
        self._R = R
        self._dim = R.shape[0]
        try:
            self._L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError as e:
            raise ValueError("R must be symmetric positive-definite.") from e
        self._R_inv = np.linalg.inv(R)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def R_inv(self) -> np.ndarray:
        return self._R_inv

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self._L @ rng.standard_normal(self._dim)

    def sample_many(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._L @ rng.standard_normal((self._dim, n))

    def sample_many_legacy(self, n: int) -> np.ndarray:
        """Same as sample_many() but uses numpy's global RNG."""
        return self._L @ np.random.standard_normal((self._dim, n))
