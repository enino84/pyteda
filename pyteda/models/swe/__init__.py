# -*- coding: utf-8 -*-
"""Shallow-water equations on the sphere — TEDA model package.

Public API:

    from pyteda.models import SWEModel

Components (also available individually):

    pyteda.models.swe._grid                 — Driscoll-Healy grid + operators
    pyteda.models.swe._dynamics             — vector-invariant SWE rhs
    pyteda.models.swe._initial_conditions   — Williamson TC2 + Rossby perturbations
    pyteda.models.swe.swe_model             — TEDA-facing SWEModel class
"""

from .swe_model import SWEModel

__all__ = ["SWEModel"]
