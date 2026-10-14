# -*- coding: utf-8 -*-
"""Models for TEDA experiments.

``Lorenz96`` is implemented here and always available. The two geophysical
models are **not** implemented in pyteda: each is an external package that
pyteda only adapts to the ``Model`` interface, and each is an optional extra::

    pip install pyteda[qg]     # QGModel  -> qg-python
    pip install pyteda[swe]    # SWEModel -> swesphere
    pip install pyteda[all]    # both

The adapters are imported lazily, so ``import pyteda`` works with neither
extra installed and asking for a model you have not installed raises an
ImportError that names the command to fix it.
"""

from .model import Model
from .lorenz96 import Lorenz96

__all__ = ["Model", "Lorenz96", "QGModel", "SWEModel"]

#: model -> (adapter module, extra name, distribution name, import name)
_OPTIONAL = {
    "QGModel": ("pyteda.models.quasi_geostrophic", "qg", "qg-python", "qgpython"),
    "SWEModel": ("pyteda.models.swe", "swe", "swesphere", "swesphere"),
}


def __getattr__(name: str):
    """Import the optional model adapters on first use (PEP 562)."""
    if name not in _OPTIONAL:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, extra, package, _import_name = _OPTIONAL[name]
    from importlib import import_module

    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"{name} needs the '{package}' package, which is not installed.\n"
            f"    pip install pyteda[{extra}]\n"
            f"{package} provides the model; pyteda only adapts it to the TEDA "
            f"Model interface."
        ) from exc

    value = getattr(module, name)
    globals()[name] = value          # cache, so this runs once
    return value


def __dir__():
    return sorted(__all__)


def available_models() -> dict:
    """Report which models can be built in this installation.

    Returns a dict mapping each model name to True, or to the pip command
    that would make it available.
    """
    from importlib.util import find_spec

    out = {"Lorenz96": True}
    for name, (_, extra, _package, import_name) in _OPTIONAL.items():
        out[name] = (
            True if find_spec(import_name) else f"pip install pyteda[{extra}]"
        )
    return out
