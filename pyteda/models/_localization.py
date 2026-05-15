# -*- coding: utf-8 -*-
"""
Localization-radius dispatch for TEDA models.

The localization radius `r` may be specified in three ways:

* ``int`` or ``float``   — a single radius applied to every component.
* ``dict``               — one radius per variable block, e.g.
                          ``{'q': 2, 'psi': 4}``. Requires the model to
                          declare its variable blocks via ``var_blocks``.
* ``np.ndarray``         — one radius per state component (length ``n``).

`resolve_radius` normalizes all three forms to a 1-D array of length
``n``, which the model then uses uniformly in `get_ngb`,
`create_decorrelation_matrix`, and `get_pre`.

When `r` is heterogeneous, an entry ``L[i, j]`` of the decorrelation
matrix combines ``r_i`` and ``r_j`` according to a `combine` rule:

* ``'mean'`` — ``r_ij = (r_i + r_j) / 2``  (default; standard in the
              literature, smooths the transition between regions).
* ``'min'``  — ``r_ij = min(r_i, r_j)``  (the more restrictive component
              dominates the pair).

When ``r_i == r_j`` the two rules give the same result, so for a uniform
``r`` the choice is irrelevant.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import numpy as np


# Public type alias — what users pass to filters and models
RadiusSpec = Union[int, float, Mapping[str, float], np.ndarray]


def resolve_radius(
    r: RadiusSpec,
    n_state: int,
    var_blocks: Optional[Dict[str, slice]] = None,
) -> np.ndarray:
    """Return a length-``n_state`` array of per-component radii.

    Parameters
    ----------
    r : int | float | dict | ndarray
        Radius specification (see module docstring).
    n_state : int
        State dimension.
    var_blocks : dict, optional
        Mapping ``block_name -> slice`` (or ``-> ndarray`` of indices)
        defining which state components belong to each block. Required
        when ``r`` is a dict.

    Returns
    -------
    r_array : ndarray of float, shape (n_state,)
        Per-component radius array.

    Raises
    ------
    ValueError
        If ``r`` is a dict but ``var_blocks`` is None or if a key in
        ``r`` is not declared in ``var_blocks``; if ``r`` is an array of
        the wrong length; if any radius is non-positive.
    TypeError
        If ``r`` has an unsupported type.
    """
    # Scalar -----------------------------------------------------------
    if isinstance(r, (int, float, np.integer, np.floating)):
        if r <= 0:
            raise ValueError(f"radius must be positive, got r={r}.")
        return np.full(n_state, float(r))

    # Array ------------------------------------------------------------
    if isinstance(r, np.ndarray):
        if r.shape != (n_state,):
            raise ValueError(
                f"radius array must have shape ({n_state},), got {r.shape}."
            )
        if np.any(r <= 0):
            raise ValueError("All entries of radius array must be positive.")
        return r.astype(float, copy=True)

    # Dict (per-block) -------------------------------------------------
    if isinstance(r, Mapping):
        if var_blocks is None:
            raise ValueError(
                "Per-block radius (dict) requires the model to declare "
                "`var_blocks`. This model does not."
            )
        unknown = set(r.keys()) - set(var_blocks.keys())
        if unknown:
            raise ValueError(
                f"Unknown variable block(s) in radius dict: {sorted(unknown)}. "
                f"Known blocks: {sorted(var_blocks.keys())}."
            )
        missing = set(var_blocks.keys()) - set(r.keys())
        if missing:
            raise ValueError(
                f"Radius dict is missing block(s): {sorted(missing)}."
            )

        out = np.empty(n_state, dtype=float)
        for name, sel in var_blocks.items():
            ri = float(r[name])
            if ri <= 0:
                raise ValueError(
                    f"radius for block '{name}' must be positive, got {ri}."
                )
            out[sel] = ri
        return out

    raise TypeError(
        f"Unsupported radius type: {type(r).__name__}. "
        f"Expected int, float, dict, or ndarray."
    )


def pairwise_radius(
    r_array: np.ndarray,
    combine: str = "mean",
) -> np.ndarray:
    """Combine per-component radii into a pairwise (n, n) matrix.

    Parameters
    ----------
    r_array : ndarray, shape (n,)
        Per-component radii (typically from ``resolve_radius``).
    combine : {'mean', 'min'}
        How to combine ``r_i`` and ``r_j``:
          * ``'mean'`` — ``(r_i + r_j) / 2`` (default).
          * ``'min'``  — ``min(r_i, r_j)``.

    Returns
    -------
    R : ndarray, shape (n, n)
        Pairwise radius matrix. ``R[i, j]`` is the radius used for the
        pair ``(i, j)`` in the gaussian decorrelation kernel.
    """
    if combine == "mean":
        return 0.5 * (r_array[:, None] + r_array[None, :])
    if combine == "min":
        return np.minimum(r_array[:, None], r_array[None, :])
    raise ValueError(
        f"combine must be 'mean' or 'min', got '{combine}'."
    )


def radius_at(r_array: np.ndarray, i: int) -> float:
    """Return the localization radius for component ``i``.

    Convenience accessor used inside ``get_ngb`` / ``get_pre``.
    """
    return float(r_array[i])
