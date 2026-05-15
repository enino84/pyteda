"""
integrators — time integration library for the QG model.

Quick start
-----------
    from integrators import get_integrator, list_integrators

    ig = get_integrator('rk4')
    ig = get_integrator('leapfrog_ra',  alpha=0.1)
    ig = get_integrator('leapfrog_raw', alpha=0.1, beta=0.5)
    ig = get_integrator('ab3')
    ig = get_integrator('ssprk3')

    new_state = ig.step(state, t, dt, rhs_fn)
    ig.reset()   # required when re-starting a simulation

    print(list_integrators())
    # ['ab2','ab3','ab4','dp5','euler','leapfrog_ra','leapfrog_raw',
    #  'midpoint','rk4','ssprk3']
"""

from .base     import TimeIntegrator, get_integrator, list_integrators, register_integrator
from .explicit import Euler, Midpoint, RK4, DP5, SSPRK3, AB2, AB3, AB4
from .leapfrog import LeapfrogRA, LeapfrogRAW

__all__ = [
    'TimeIntegrator',
    'get_integrator', 'list_integrators', 'register_integrator',
    'Euler', 'Midpoint', 'RK4', 'DP5', 'SSPRK3',
    'AB2', 'AB3', 'AB4',
    'LeapfrogRA', 'LeapfrogRAW',
]
