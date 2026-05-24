# -*- coding: utf-8 -*-
"""
Grid-based observation index helpers.

These build the index sets consumed by :class:`LinearSelection` for
models laid out on a 2-D grid (e.g. the shallow-water model on a
``Nlat × Nlon`` Gaussian grid, or the quasi-geostrophic model on an
``m × n`` grid). A model is "grid-like" here if it exposes

    * ``field_size``  — number of grid points per variable, and
    * ``var_blocks``  — dict mapping variable name -> slice in the state
                        vector,

and if each field is stored row-major (C order) as a flattened
``(rows, cols)`` array, which is the convention used by the SWE and QG
models in pyteda.

Public functions
----------------
checkerboard_indices
    Regularly spaced "checkerboard" sampling: observe every ``spacing``-th
    grid point in each direction (optionally a different spacing per axis,
    and optionally an offset). Same pattern applied to every variable.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np


def _grid_shape(model, n_rows, n_cols):
    """Resolve the 2-D (rows, cols) shape of a single field, or None if 1-D.

    Priority: explicit (n_rows, n_cols) args > model.grid.Nlat/Nlon >
    model.m_grid/n_grid. Returns None (rather than raising) when the model
    exposes no 2-D grid, signalling that the 1-D strided sampler applies.
    """
    if n_rows is not None and n_cols is not None:
        return int(n_rows), int(n_cols)
    grid = getattr(model, "grid", None)
    if grid is not None and hasattr(grid, "Nlat") and hasattr(grid, "Nlon"):
        return int(grid.Nlat), int(grid.Nlon)
    if hasattr(model, "m_grid") and hasattr(model, "n_grid"):
        return int(model.m_grid), int(model.n_grid)
    return None


def _field_length(model):
    """Length of a single field: field_size if present, else state dim n."""
    fs = getattr(model, "field_size", None)
    if fs is not None:
        return int(fs)
    for attr in ("dim", "n", "n_state"):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    raise ValueError(
        "Cannot determine field length: model exposes neither "
        "`field_size` nor `dim`/`n`/`n_state`."
    )


def strided_indices(
    model,
    spacing: int = 2,
    variables: Optional[Sequence[str]] = None,
    offset: int = 0,
    n: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Indices of a regularly strided 1-D observation set.

    The 1-D analogue of :func:`checkerboard_indices`: observe every
    ``spacing``-th component of a 1-D field, i.e. positions
    ``offset, offset+spacing, offset+2*spacing, ...``. This is the
    standard "observe every other variable" network for Lorenz-96
    (``spacing=2``) and its coarser variants.

    Parameters
    ----------
    model : object
        1-D model. Field length is taken from ``field_size`` if present,
        else from ``dim``/``n``/``n_state``. If ``var_blocks`` is present
        the pattern is applied within each requested variable block;
        otherwise the whole state is treated as a single field.
    spacing : int, default 2
        Stride. ``spacing=1`` observes everything.
    variables : sequence of str, optional
        Variable blocks to observe (needs ``model.var_blocks``). Default:
        all blocks, or the whole state if no blocks are declared.
    offset : int, default 0
        Starting offset, taken modulo ``spacing``.
    n : int, optional
        Explicit field length, overriding model introspection.

    Returns
    -------
    indices : ndarray of int
        Sorted state-vector indices to observe.
    n_per_field : int
        Number of observed points per field.
    """
    if spacing < 1:
        raise ValueError("spacing must be >= 1.")
    L = int(n) if n is not None else _field_length(model)
    off = int(offset) % int(spacing)
    pos = np.arange(off, L, int(spacing))
    if pos.size == 0:
        raise ValueError(
            f"spacing {spacing} with offset {off} selects no points "
            f"on a field of length {L}."
        )
    n_per_field = pos.size

    var_blocks = getattr(model, "var_blocks", None)
    if var_blocks is None:
        return np.sort(pos), n_per_field

    if variables is None:
        variables = list(var_blocks.keys())
    parts = []
    for var in variables:
        if var not in var_blocks:
            raise ValueError(
                f"variable {var!r} not in model.var_blocks "
                f"({sorted(var_blocks.keys())})."
            )
        parts.append(pos + var_blocks[var].start)
    indices = np.concatenate(parts)
    indices.sort()
    return indices, n_per_field



def checkerboard_indices(
    model,
    spacing: Union[int, Sequence[int]] = 2,
    variables: Optional[Sequence[str]] = None,
    offset: Union[int, Sequence[int]] = 0,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Indices of a regularly spaced grid (checkerboard) observation set.

    Observes every ``spacing``-th grid point in each direction, i.e. the
    points ``(i, j)`` with ``i % s_row == off_row`` and
    ``j % s_col == off_col``. The same sub-grid is observed in every
    requested variable, and the per-variable index sets are concatenated
    in the order of ``variables`` (default: all of ``model.var_blocks``).

    Parameters
    ----------
    model : object
        Grid-like model exposing ``field_size`` and ``var_blocks`` (and a
        2-D field shape, see module docstring).
    spacing : int or (int, int), default 2
        Sampling stride. A single int applies to both axes (a square
        checkerboard). A pair ``(s_row, s_col)`` sets the row (latitude)
        and column (longitude) strides independently. ``spacing=1``
        observes every point (full coverage).
    variables : sequence of str, optional
        Which variable blocks to observe. Default: all blocks in
        ``model.var_blocks``.
    offset : int or (int, int), default 0
        Starting offset ``(off_row, off_col)`` of the sampling pattern,
        each taken modulo the corresponding spacing.
    n_rows, n_cols : int, optional
        Explicit field shape, overriding model introspection.

    Returns
    -------
    indices : ndarray of int
        Sorted state-vector indices to observe, concatenated across the
        requested variables.
    n_per_field : int
        Number of observed points in a single field (same for each
        variable).

    Examples
    --------
    Observe every other grid point in both directions of an SWE state::

        idx, k = checkerboard_indices(model, spacing=2)
        op = LinearSelection(m=idx.size, n_state=model.dim, indices=idx)

    A coarser zonal sampling than meridional::

        idx, _ = checkerboard_indices(model, spacing=(2, 4))
    """
    nrow_ncol = _grid_shape(model, n_rows, n_cols)
    if nrow_ncol is None:
        # 1-D model (e.g. Lorenz-96): fall back to strided sampling.
        # A scalar spacing maps directly; a pair uses its first entry.
        s = spacing if np.isscalar(spacing) else spacing[0]
        o = offset if np.isscalar(offset) else offset[0]
        return strided_indices(model, spacing=int(s), variables=variables,
                               offset=int(o), n=n_cols)
    nrow, ncol = nrow_ncol

    # normalise spacing / offset to per-axis pairs
    if np.isscalar(spacing):
        s_row = s_col = int(spacing)
    else:
        s_row, s_col = (int(spacing[0]), int(spacing[1]))
    if s_row < 1 or s_col < 1:
        raise ValueError("spacing must be >= 1 in each direction.")

    if np.isscalar(offset):
        off_row = off_col = int(offset)
    else:
        off_row, off_col = (int(offset[0]), int(offset[1]))
    off_row %= s_row
    off_col %= s_col

    # sampled rows/cols of one field
    rows = np.arange(off_row, nrow, s_row)
    cols = np.arange(off_col, ncol, s_col)
    if rows.size == 0 or cols.size == 0:
        raise ValueError(
            f"spacing ({s_row},{s_col}) with offset ({off_row},{off_col}) "
            f"selects no points on a {nrow}x{ncol} grid."
        )

    # flat (row-major) indices within a single field
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    flat_field = (rr * ncol + cc).ravel()
    flat_field.sort()
    n_per_field = flat_field.size

    # map to full state vector for each requested variable
    var_blocks = getattr(model, "var_blocks", None)
    if var_blocks is None:
        raise ValueError("model has no `var_blocks`; cannot place indices.")
    if variables is None:
        variables = list(var_blocks.keys())

    field_size = getattr(model, "field_size", None)
    parts = []
    for var in variables:
        if var not in var_blocks:
            raise ValueError(
                f"variable {var!r} not in model.var_blocks "
                f"({sorted(var_blocks.keys())})."
            )
        start = var_blocks[var].start
        # sanity: the field must actually be `field_size` long if known
        if field_size is not None and flat_field.max() >= field_size:
            raise ValueError(
                "checkerboard indices exceed field_size; the grid shape "
                "does not match the variable block length."
            )
        parts.append(flat_field + start)

    indices = np.concatenate(parts)
    indices.sort()
    return indices, n_per_field
