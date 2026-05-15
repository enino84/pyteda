# -*- coding: utf-8 -*-
"""
Observation container.

`Observation` bundles an `ObservationOperator` and an `ObservationNoise`
model and exposes the API that the analysis filters consume.

Backwards-compatible API used by the existing analysis filters:

    obs.get_observation()              -> y
    obs.get_observation_operator()     -> H matrix (Jacobian; constant if linear)
    obs.get_observation_operator_index()-> indices (LinearSelection only)
    obs.get_data_error_covariance()    -> R
    obs.get_precision_error_covariance()-> R^{-1}
    obs.y, obs.R, obs.H, obs.H_index   -> attributes used by LEnKF/LETKF

New API:

    obs.apply(x)        -> h(x), works for vectors and ensembles
    obs.linearize(x)    -> Jacobian at x
    obs.is_linear       -> bool
    obs.operator        -> the ObservationOperator object
    obs.noise           -> the ObservationNoise object

Two ways to construct:

1) Original style (still works):

       obs = Observation(m=400, std_obs=0.01)
       obs.set_observation_operator(n_state)   # builds a LinearSelection
       obs.generate_observation(x_true)        # samples y = H x + noise

2) New style (recommended):

       op    = LinearSelection(m=400, n_state=1024, rng=rng)
       noise = IsotropicDiagonal(std=0.01, dim=400)
       obs   = Observation(operator=op, noise=noise, rng=rng)
       obs.generate_observation(x_true)

   Or directly with already-sampled values (used by Scenario):

       obs = Observation.from_arrays(y, op, noise)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .operators import ObservationOperator, LinearSelection
from .noise import ObservationNoise, IsotropicDiagonal


class Observation:
    """Observation container: composes an operator and a noise model."""

    def __init__(
        self,
        m: Optional[int] = None,
        std_obs: float = 0.01,
        obs_operator_fixed: bool = False,
        H: Optional[np.ndarray] = None,
        operator: Optional[ObservationOperator] = None,
        noise: Optional[ObservationNoise] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self._rng = rng if rng is not None else np.random.default_rng()

        if operator is not None:
            # New API path
            self._operator: Optional[ObservationOperator] = operator
            if noise is None:
                noise = IsotropicDiagonal(std=std_obs, dim=operator.dim_obs)
            elif isinstance(noise, IsotropicDiagonal) and noise._dim is None:
                noise.bind_dim(operator.dim_obs)
            if noise.dim != operator.dim_obs:
                raise ValueError(
                    f"Noise dim ({noise.dim}) does not match operator dim_obs "
                    f"({operator.dim_obs})."
                )
            self._noise: ObservationNoise = noise
            self.m = operator.dim_obs
            self.obs_operator_fixed = True
        else:
            # Legacy API path: defer operator construction.
            if m is None:
                raise ValueError(
                    "Observation needs either `m` (legacy) or `operator` (new API)."
                )
            self.m = int(m)
            self._operator = None
            if H is not None:
                from .operators import LinearMatrix
                self._operator = LinearMatrix(H)
            self._noise = IsotropicDiagonal(std=std_obs, dim=self.m)
            self.obs_operator_fixed = bool(obs_operator_fixed)

        # Backwards-compat attributes for the analysis filters.
        self.y: Optional[np.ndarray] = None
        if self._operator is not None and self._operator.is_linear:
            self.H = self._operator.linearize(None)
        else:
            self.H = None
        if isinstance(self._noise, IsotropicDiagonal) and self._noise._dim is None:
            self.R = None
        else:
            self.R = self._noise.R

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        y: np.ndarray,
        operator: ObservationOperator,
        noise: ObservationNoise,
    ) -> "Observation":
        """Build an Observation from a pre-sampled y and an operator+noise.

        Used by `Scenario` to replay frozen observations into the filters.
        """
        obs = cls.__new__(cls)
        obs._rng = np.random.default_rng()
        obs._operator = operator
        obs._noise = noise
        obs.m = operator.dim_obs
        obs.obs_operator_fixed = True
        obs.y = np.asarray(y)
        # NOTE: do NOT materialise R here. For high-dimensional setups
        # (e.g. SWE LMAX=32 with ~18000 obs) a dense m×m matrix would
        # cost gigabytes. Filters that actually need R get it via
        # `get_data_error_covariance()` (or via the noise object), and
        # only the non-localised filters touch the full R; localised
        # filters work in observation subspaces that are tiny.
        obs.H = operator.linearize(None) if operator.is_linear else None
        return obs

    # ------------------------------------------------------------------
    # New high-level API
    # ------------------------------------------------------------------
    @property
    def operator(self) -> ObservationOperator:
        if self._operator is None:
            raise RuntimeError(
                "No operator set. Call set_observation_operator(n) or pass `operator=` at construction."
            )
        return self._operator

    @property
    def noise(self) -> ObservationNoise:
        return self._noise

    @property
    def R(self):
        """Observation-error covariance, fetched from the noise model.

        Materialised lazily (and only when actually requested) so that
        `from_arrays`-built observations don't pay the cost of building
        an m×m matrix for huge m.
        """
        return self._noise.R

    @R.setter
    def R(self, value):
        # Legacy setter: some old code path assigns obs.R directly. We
        # honour the assignment by storing it in an instance attribute
        # that shadows the property — but the canonical source is the
        # noise model.
        self.__dict__["R"] = value

    @property
    def is_linear(self) -> bool:
        return self.operator.is_linear

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.operator.apply(x)

    def linearize(self, x: np.ndarray) -> np.ndarray:
        return self.operator.linearize(x)

    # ------------------------------------------------------------------
    # Legacy API (preserved from the original class)
    # ------------------------------------------------------------------
    def set_observation_operator(self, n: int):
        op = LinearSelection(m=self.m, n_state=n, rng=self._rng)
        self._operator = op
        self.H = op.linearize(None)

    def generate_observation(self, x: np.ndarray):
        if not self.obs_operator_fixed or self._operator is None:
            self.set_observation_operator(x.size)

        # Bind noise dim if needed
        if isinstance(self._noise, IsotropicDiagonal) and self._noise._dim is None:
            self._noise.bind_dim(self._operator.dim_obs)

        self.R = self._noise.R
        self.y = self._operator.apply(x) + self._noise.sample(self._rng)

    def get_observation(self) -> np.ndarray:
        return self.y

    def get_observation_operator(self) -> np.ndarray:
        if self.is_linear:
            return self.operator.linearize(None)
        return self.operator.linearize(np.zeros(self.operator.dim_state))

    def get_observation_operator_index(self) -> np.ndarray:
        if isinstance(self._operator, LinearSelection):
            return self._operator.indices
        raise AttributeError(
            "Index-based access is only defined for LinearSelection operators."
        )

    @property
    def H_index(self) -> np.ndarray:
        if isinstance(self._operator, LinearSelection):
            return self._operator.indices
        raise AttributeError(
            "H_index is only defined for LinearSelection operators."
        )

    @H_index.setter
    def H_index(self, value):
        # No-op for backwards compat: LinearSelection.indices is the source of truth.
        pass

    def get_data_error_covariance(self) -> np.ndarray:
        return self._noise.R

    def get_precision_error_covariance(self) -> np.ndarray:
        return self._noise.R_inv
