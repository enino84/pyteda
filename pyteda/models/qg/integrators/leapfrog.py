"""
integrators/leapfrog.py
=======================
Leapfrog family of time integration schemes with Robert-Asselin (RA)
and Robert-Asselin-Williams (RAW) filters.

Schemes
-------
LeapfrogRA   — Leapfrog + Robert-Asselin filter (Asselin 1972)
LeapfrogRAW  — Leapfrog + Robert-Asselin-Williams filter (Williams 2009)

Both use a single RK4 step to bootstrap the first step and avoid
exciting the computational mode from the initial condition.

Physics
-------
Standard centred Leapfrog (2nd-order):

    q^{n+1} = q^{n-1} + 2*dt * F(q^n)

Neutrally stable for purely imaginary eigenvalues, but sustains a
spurious computational mode phase-separated from the physical mode.
Filters suppress the computational mode:

RA filter (Asselin 1972):
    q^n <- q^n + alpha/2 * (q^{n-1} - 2*q^n + q^{n+1})

RAW filter (Williams 2009), splitting parameter beta=0.5:
    q^n    <- q^n    + alpha*(1-beta)/2 * delta
    q^{n+1}<- q^{n+1} - alpha*beta/2   * delta
    where delta = q^{n-1} - 2*q^n + q^{n+1}

The RAW filter is 3rd-order accurate in the filter correction
(vs 1st-order for RA) and dramatically reduces spurious amplitude
damping. See the README for a quantitative comparison.

References
----------
Asselin, R. (1972). Frequency filter for time integrations.
    Mon. Wea. Rev. 100, 487-490.

Williams, P.D. (2009). A proposed modification to the Robert-Asselin
    time filter. Mon. Wea. Rev. 137, 2538-2546.
"""

from __future__ import annotations
import numpy as np
from .base import TimeIntegrator, register_integrator
from .explicit import RK4


class _LeapfrogBase(TimeIntegrator):
    """Internal base for Leapfrog variants."""

    def __init__(self, alpha: float = 0.1):
        """
        Parameters
        ----------
        alpha : float
            Filter strength. Typical range 0.05-0.20.
            alpha=0 disables filtering (not recommended for long runs).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha   = alpha
        self._q_prev = None
        self._rk4    = RK4()

    def reset(self):
        self._q_prev = None

    @property
    def order(self): return 2

    @property
    def rhs_evals_per_step(self): return 1   # after bootstrap

    def _apply_filter(self, q_nm1, q_n, q_np1, delta):
        raise NotImplementedError

    def step(self, state, t, dt, rhs_fn):
        # Bootstrap: use RK4 for the very first step
        if self._q_prev is None:
            self._q_prev = state.copy()
            return self._rk4.step(state, t, dt, rhs_fn)

        q_nm1 = self._q_prev
        q_n   = state
        F_n   = rhs_fn(q_n)

        # Leapfrog leap
        q_np1 = q_nm1 + 2.0 * dt * F_n

        # Filter
        delta       = q_nm1 - 2.0 * q_n + q_np1
        q_n_f, q_np1_f = self._apply_filter(q_nm1, q_n, q_np1, delta)

        self._q_prev = q_n_f
        return q_np1_f


@register_integrator('leapfrog_ra')
class LeapfrogRA(_LeapfrogBase):
    """
    Leapfrog + Robert-Asselin (RA) filter (Asselin 1972).

    The correction is applied to q^n only; q^{n+1} is unmodified.
    Introduces 1st-order amplitude damping. For long integrations,
    this causes measurable bias in kinetic energy (~-21%) and enstrophy
    (~-45%) relative to a reference RK4 run. Consider LeapfrogRAW instead.
    """

    def _apply_filter(self, q_nm1, q_n, q_np1, delta):
        return q_n + 0.5 * self.alpha * delta, q_np1


@register_integrator('leapfrog_raw')
class LeapfrogRAW(_LeapfrogBase):
    """
    Leapfrog + Robert-Asselin-Williams (RAW) filter (Williams 2009).

    Splits the correction between q^n and q^{n+1} via the splitting
    parameter beta (default 0.5, as recommended by Williams 2009).
    Preserves 2nd-order accuracy while suppressing the computational mode.
    Energy and enstrophy biases are reduced to ~-2% and ~+4% respectively.

    Parameters
    ----------
    alpha : float
        Filter strength (same meaning as in RA).
    beta  : float
        Splitting parameter. 0.5 recommended. beta=0 -> plain RA.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.5):
        super().__init__(alpha=alpha)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta

    def _apply_filter(self, q_nm1, q_n, q_np1, delta):
        a, b = self.alpha, self.beta
        return (q_n   + 0.5 * a * (1.0 - b) * delta,
                q_np1 - 0.5 * a * b          * delta)
