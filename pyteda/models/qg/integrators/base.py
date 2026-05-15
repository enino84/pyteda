"""
integrators/base.py
===================
Abstract base class and global registry for time integration schemes.

All integrators share a single interface:

    new_state = integrator.step(state, t, dt, rhs_fn)

where
    state   : np.ndarray  — current model state (q field)
    t       : float       — current time
    dt      : float       — time step
    rhs_fn  : callable    — rhs_fn(state) -> tendency, same shape as state

This design fully decouples the numerical method from the physical model.
Any model that exposes an rhs_fn callable is immediately compatible with
every integrator in this library.

Adding a new scheme
-------------------
1. Subclass TimeIntegrator, implement step().
2. Decorate with @register_integrator('myname').
3. It becomes available via get_integrator('myname').
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

_REGISTRY: dict[str, type] = {}


def register_integrator(name: str):
    """Class decorator — registers the integrator under the given name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_integrator(name: str, **kwargs) -> 'TimeIntegrator':
    """
    Instantiate a registered integrator by name.

    Parameters
    ----------
    name    : registered name, e.g. 'rk4', 'ab3', 'leapfrog_ra'
    **kwargs: passed to the integrator constructor

    Returns
    -------
    TimeIntegrator instance
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown integrator '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_integrators() -> list[str]:
    """Return sorted list of all registered integrator names."""
    return sorted(_REGISTRY.keys())


class TimeIntegrator(ABC):
    """
    Abstract base for all time integration schemes.

    Subclasses must implement step(). Multistep methods should also
    override reset() to clear their internal history buffers.
    """

    @abstractmethod
    def step(
        self,
        state:  np.ndarray,
        t:      float,
        dt:     float,
        rhs_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Advance state by one time step.

        Parameters
        ----------
        state  : current state array (not modified in-place)
        t      : current time
        dt     : time step size
        rhs_fn : callable(state) -> tendency, same shape as state

        Returns
        -------
        new_state : np.ndarray, same shape as state
        """

    def reset(self) -> None:
        """Reset internal memory (override in multistep / leapfrog methods)."""

    @property
    def rhs_evals_per_step(self) -> int:
        """RHS evaluations consumed per step (informational)."""
        return 1

    @property
    def order(self) -> int:
        """Formal order of accuracy (informational)."""
        return 1

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(order={self.order}, rhs_evals={self.rhs_evals_per_step})")
