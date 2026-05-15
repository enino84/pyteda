# -*- coding: utf-8 -*-
"""Internal helpers for IO: format dispatch by extension."""

from __future__ import annotations

import os


def detect_format(path: str) -> str:
    """Return 'netcdf' or 'npz' based on the file extension.

    Recognized extensions:
      * ``.nc``, ``.netcdf``, ``.cdf``  → netcdf
      * ``.npz``                         → npz
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nc", ".netcdf", ".cdf"):
        return "netcdf"
    if ext == ".npz":
        return "npz"
    raise ValueError(
        f"Cannot infer file format from extension '{ext}'. "
        f"Use one of .nc, .netcdf, .cdf (netCDF) or .npz (numpy)."
    )
