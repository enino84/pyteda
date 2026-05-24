# -*- coding: utf-8 -*-
"""
Observation submodule.

Public API:
    Observation              — container that bundles operator + noise

Operators:
    ObservationOperator      — abstract base
    LinearSelection          — y = x[indices] (original TEDA behavior)
    LinearMatrix             — y = H x for arbitrary H
    NonlinearOperator        — y = h(x) with optional analytical Jacobian

Noise models:
    ObservationNoise         — abstract base
    IsotropicDiagonal        — R = sigma^2 * I (original TEDA behavior)
    HeterogeneousDiagonal    — R = diag(sigma_i^2)
    DenseCovariance          — arbitrary symmetric positive-definite R
"""

from .observation_core import Observation
from .operators import (
    ObservationOperator,
    LinearSelection,
    LinearMatrix,
    NonlinearOperator,
)
from .noise import (
    ObservationNoise,
    IsotropicDiagonal,
    HeterogeneousDiagonal,
    BlockDiagonal,
    DenseCovariance,
)
from .grid_masks import checkerboard_indices, strided_indices

__all__ = [
    "Observation",
    "ObservationOperator",
    "LinearSelection",
    "LinearMatrix",
    "NonlinearOperator",
    "ObservationNoise",
    "IsotropicDiagonal",
    "HeterogeneousDiagonal",
    "BlockDiagonal",
    "DenseCovariance",
    "checkerboard_indices",
    "strided_indices",
]
