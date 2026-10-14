# -*- coding: utf-8 -*-
"""Shallow-water equations on the sphere — TEDA adapter.

The model is the external ``swesphere`` package (``pip install pyteda[swe]``);
this subpackage only adapts it to the TEDA ``Model`` interface.

    from pyteda.models import SWEModel
"""

from .swe_model import SWEModel

__all__ = ["SWEModel"]
