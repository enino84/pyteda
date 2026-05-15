"""
integrators/explicit.py
=======================
Explicit time integration schemes.

Schemes
-------
Euler    — 1st-order explicit Euler
Midpoint — 2nd-order explicit midpoint (RK2)
RK4      — 4th-order classical Runge-Kutta
DP5      — 5th-order Dormand-Prince
SSPRK3   — 3rd-order Strong Stability Preserving RK (Shu & Osher 1988)
AB2      — 2nd-order Adams-Bashforth  (multistep; RK4 bootstrap)
AB3      — 3rd-order Adams-Bashforth  (multistep; RK4 bootstrap)
AB4      — 4th-order Adams-Bashforth  (multistep; RK4 bootstrap)

References
----------
Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially
    non-oscillatory shock-capturing schemes. J. Comput. Phys. 77, 439-471.

Dormand, J.R. & Prince, P.J. (1980). A family of embedded Runge-Kutta
    formulae. J. Comput. Appl. Math. 6, 19-26.

Adams, J.C. & Bashforth, F. (1883). An Attempt to Test the Theories of
    Capillary Action. Cambridge University Press.
"""

from __future__ import annotations
import numpy as np
from .base import TimeIntegrator, register_integrator


# ── 1st order ─────────────────────────────────────────────────────────────────

@register_integrator('euler')
class Euler(TimeIntegrator):
    """Explicit Euler (1st-order). Unconditionally unstable for imaginary eigenvalues."""

    @property
    def order(self): return 1

    @property
    def rhs_evals_per_step(self): return 1

    def step(self, state, t, dt, rhs_fn):
        return state + dt * rhs_fn(state)


# ── 2nd order ─────────────────────────────────────────────────────────────────

@register_integrator('midpoint')
class Midpoint(TimeIntegrator):
    """Explicit midpoint / RK2 (2nd-order)."""

    @property
    def order(self): return 2

    @property
    def rhs_evals_per_step(self): return 2

    def step(self, state, t, dt, rhs_fn):
        k1 = rhs_fn(state)
        k2 = rhs_fn(state + 0.5 * dt * k1)
        return state + dt * k2


# ── 4th order ─────────────────────────────────────────────────────────────────

@register_integrator('rk4')
class RK4(TimeIntegrator):
    """Classical 4th-order Runge-Kutta."""

    @property
    def order(self): return 4

    @property
    def rhs_evals_per_step(self): return 4

    def step(self, state, t, dt, rhs_fn):
        k1 = rhs_fn(state)
        k2 = rhs_fn(state + 0.5 * dt * k1)
        k3 = rhs_fn(state + 0.5 * dt * k2)
        k4 = rhs_fn(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ── 5th order ─────────────────────────────────────────────────────────────────

@register_integrator('dp5')
class DP5(TimeIntegrator):
    """
    5th-order Dormand-Prince (fixed-step, 5th-order weights).

    The 6-stage FSAL structure is preserved; the 5th-order solution is
    used directly without embedded error control.
    """

    @property
    def order(self): return 5

    @property
    def rhs_evals_per_step(self): return 6

    def step(self, state, t, dt, rhs_fn):
        k1 = rhs_fn(state)
        k2 = rhs_fn(state + dt * (1/5)*k1)
        k3 = rhs_fn(state + dt * (3/40*k1     + 9/40*k2))
        k4 = rhs_fn(state + dt * (44/45*k1    - 56/15*k2   + 32/9*k3))
        k5 = rhs_fn(state + dt * (19372/6561*k1 - 25360/2187*k2
                                   + 64448/6561*k3 - 212/729*k4))
        k6 = rhs_fn(state + dt * (9017/3168*k1  - 355/33*k2
                                   + 46732/5247*k3 + 49/176*k4
                                   - 5103/18656*k5))
        return state + dt * (35/384*k1  + 500/1113*k3 + 125/384*k4
                              - 2187/6784*k5 + 11/84*k6)


# ── SSP-RK3 ───────────────────────────────────────────────────────────────────

@register_integrator('ssprk3')
class SSPRK3(TimeIntegrator):
    """
    Strong Stability Preserving Runge-Kutta, 3rd-order (Shu & Osher 1988).

    Preserves the total variation diminishing (TVD) property of the
    spatial discretisation under the CFL constraint. Well-suited for
    problems with sharp fronts or near-discontinuities.

    Shu-Osher form:
        u^(1)   = u^n + dt * L(u^n)
        u^(2)   = 3/4 u^n + 1/4 [ u^(1) + dt * L(u^(1)) ]
        u^{n+1} = 1/3 u^n + 2/3 [ u^(2) + dt * L(u^(2)) ]
    """

    @property
    def order(self): return 3

    @property
    def rhs_evals_per_step(self): return 3

    def step(self, state, t, dt, rhs_fn):
        u1 = state + dt * rhs_fn(state)
        u2 = 0.75 * state + 0.25 * (u1 + dt * rhs_fn(u1))
        return (1.0/3.0) * state + (2.0/3.0) * (u2 + dt * rhs_fn(u2))


# ── Adams-Bashforth family ────────────────────────────────────────────────────

class _AdamsBashforthBase(TimeIntegrator):
    """
    Base for explicit Adams-Bashforth multistep methods.

    Startup uses RK4 for the first (order-1) steps to avoid exciting
    transient errors from a lower-order bootstrap.

    After startup, only one rhs evaluation per step is needed (the
    history stores past tendencies at no additional cost).
    """

    _coeffs: tuple   # AB coefficients from newest to oldest tendency

    def __init__(self):
        self._history: list[np.ndarray] = []
        self._rk4 = RK4()

    def reset(self):
        self._history = []

    @property
    def rhs_evals_per_step(self): return 1   # after startup

    def step(self, state, t, dt, rhs_fn):
        k = rhs_fn(state)
        order = len(self._coeffs)

        if len(self._history) < order - 1:
            # Startup phase: use RK4 and collect tendencies
            new_state = self._rk4.step(state, t, dt, rhs_fn)
        else:
            # Full multistep update
            tendencies = [k] + list(self._history[:order - 1])
            increment  = sum(c * f for c, f in zip(self._coeffs, tendencies))
            new_state  = state + dt * increment

        # Prepend current tendency; keep only what is needed
        self._history.insert(0, k)
        self._history = self._history[:order - 1]
        return new_state


@register_integrator('ab2')
class AB2(_AdamsBashforthBase):
    """
    2nd-order Adams-Bashforth.

        u^{n+1} = u^n + dt * (3/2 F^n - 1/2 F^{n-1})
    """
    _coeffs = (3/2, -1/2)

    @property
    def order(self): return 2


@register_integrator('ab3')
class AB3(_AdamsBashforthBase):
    """
    3rd-order Adams-Bashforth.

        u^{n+1} = u^n + dt * (23/12 F^n - 16/12 F^{n-1} + 5/12 F^{n-2})
    """
    _coeffs = (23/12, -16/12, 5/12)

    @property
    def order(self): return 3


@register_integrator('ab4')
class AB4(_AdamsBashforthBase):
    """
    4th-order Adams-Bashforth.

        u^{n+1} = u^n + dt * (55/24 F^n - 59/24 F^{n-1}
                               + 37/24 F^{n-2} - 9/24 F^{n-3})
    """
    _coeffs = (55/24, -59/24, 37/24, -9/24)

    @property
    def order(self): return 4
