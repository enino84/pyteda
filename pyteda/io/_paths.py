# -*- coding: utf-8 -*-
"""
Helpers for choosing where to store TEDA artifacts on disk.

The rule is intentionally minimal:

* If the user passes an explicit ``path``, use it.
* Otherwise, default to ``./scenarios`` (relative to the current working
  directory).

The directory is created if it does not exist.

Examples
--------

Use the default location, relative to the notebook or script::

    from pyteda.io import get_data_dir
    scen_dir = get_data_dir()
    # → './scenarios'

Pass an absolute or any custom path::

    scen_dir = get_data_dir('/data/runs/lorenz96')
    scen_dir = get_data_dir('~/teda_results')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def get_data_dir(path: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Return a directory where TEDA artifacts (scenarios, X0, results) are stored.

    Parameters
    ----------
    path : str or PathLike, optional
        Explicit destination. If None (default), use ``./scenarios`` relative
        to the current working directory. ``~`` is expanded to the user
        home directory.

    Returns
    -------
    Path
        The resolved directory path. The directory is created if missing.
    """
    if path is None:
        path = "scenarios"
    p = Path(os.path.expanduser(str(path)))
    p.mkdir(parents=True, exist_ok=True)
    return p
