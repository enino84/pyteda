# -*- coding: utf-8 -*-
"""Localization plumbing shared by the localized filters.

A localized filter decides *where* in the state each analysis looks, and it
asks the model for that geometry through three methods::

    model.get_ngb(i, r, cross=...)                  # neighbours of i
    model.get_pre(i, r, cross=...)                  # neighbours with index < i
    model.create_decorrelation_matrix(r, cross=..., cross_scale=...)

For a model with more than one variable block (SWE: ``u``, ``v``, ``h``; QG:
``q``, ``psi``) ``cross`` decides whether the localized neighbourhood of a
point in one block reaches into the other blocks:

``cross=False`` (the default)
    Each block is localized on its own. The precision factor and the
    decorrelation matrix come out block diagonal by variable, so the analysis
    is **univariate**: observing ``h`` moves ``h`` and leaves ``u`` and ``v``
    exactly where the forecast put them.

``cross=True``
    The neighbourhood of a point includes the same grid cell in the other
    blocks, so the analysis is **multivariate**: an observation of ``h``
    reaches ``u`` and ``v`` through the ensemble cross covariances.
    ``cross_scale`` in ``[0, 1]`` tapers those cross-block entries; 1.0 trusts
    them as much as the within-block ones.

Single-variable models (Lorenz96) accept ``cross`` and ignore it.

These helpers exist so a filter can pass ``cross`` to any model without
knowing which model it has. A model written before ``cross`` existed, whose
methods take only ``(i, r)``, still works with ``cross=False``; asking it for
``cross=True`` raises a ``TypeError`` that names the problem instead of
silently returning a univariate neighbourhood.
"""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ["supports_cross", "call_ngb", "call_pre", "build_decorrelation"]


def supports_cross(func: Any) -> bool:
    """True when ``func`` accepts a ``cross`` keyword."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):       # builtins, C extensions
        return False
    if "cross" in params:
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _require(func: Any, cross: bool, what: str) -> bool:
    """Decide whether to forward ``cross``; complain if it is needed but absent."""
    if supports_cross(func):
        return True
    if cross:
        raise TypeError(
            f"cross=True was requested but this model's {what}() does not "
            f"accept a `cross` keyword, so a cross-variable neighbourhood "
            f"cannot be built. Either use cross=False (univariate analysis) "
            f"or add `cross` to the model's localization methods; "
            f"pyteda.models.SWEModel and QGModel are examples."
        )
    return False


def call_ngb(model: Any, i: int, r: Any, cross: bool = False):
    """``model.get_ngb(i, r)``, forwarding ``cross`` when the model takes it."""
    if _require(model.get_ngb, cross, "get_ngb"):
        return model.get_ngb(i, r, cross=cross)
    return model.get_ngb(i, r)


def call_pre(model: Any, i: int, r: Any, cross: bool = False):
    """``model.get_pre(i, r)``, forwarding ``cross`` when the model takes it."""
    if _require(model.get_pre, cross, "get_pre"):
        return model.get_pre(i, r, cross=cross)
    return model.get_pre(i, r)


def build_decorrelation(
    model: Any, r: Any, cross: bool = False, cross_scale: float = 1.0
) -> None:
    """``model.create_decorrelation_matrix(r)``, forwarding the cross options."""
    func = model.create_decorrelation_matrix
    if _require(func, cross, "create_decorrelation_matrix"):
        func(r, cross=cross, cross_scale=cross_scale)
    else:
        func(r)
