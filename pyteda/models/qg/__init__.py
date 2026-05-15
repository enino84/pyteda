# -*- coding: utf-8 -*-
"""
Vendored 1.5-layer QG model from the qg-integrators SoftwareX package.

This subpackage contains:

* ``_qg_core.py``         — the original ``qg_model_v2.py`` (renamed).
* ``boundary_conditions`` — Helmholtz solver and finite-difference operators
                            for ``dirichlet`` and ``channel`` BCs.
* ``integrators``         — pluggable time-stepping schemes (Euler, RK4,
                            DP5, AB2/3/4, SSPRK3, Leapfrog+RA, Leapfrog+RAW).
* ``initial_conditions``  — IC library (zero, fourier, vortex, dipole,
                            rossby_wave, band_noise, restart_npz, restart_nc).

End users should import the TEDA-facing ``QGModel`` from
``pyteda.models.quasi_geostrophic`` (also re-exported from
``pyteda.models``); this submodule is the implementation backend.

License notice
--------------
The vendored code originates from ``qg-integrators`` and retains its
upstream LICENSE (see ``LICENSE_qg_integrators``). The vendored copy is
preserved with minimal modifications (only relative-import adjustments
needed to make it a TEDA subpackage).
"""
