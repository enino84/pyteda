# qg-integrators

**A modular testbed for time integration schemes in a 1.5-layer quasi-geostrophic model**

Extended from [qg-leapfrog-stability](https://github.com/your-repo/qg-leapfrog-stability)
for submission to *Geoscience Data Journal*.

---

## Overview

`qg-integrators` extends the original QG-Python port of Pavel Sakov's QG-C model
with two new axes of flexibility:

1. **Pluggable time integrators** — 10 schemes via a common interface
2. **Selectable boundary conditions** — closed basin (Dirichlet) and zonal channel (periodic-x)
3. **Extended initial conditions** — 8 IC types, all BC-aware

Any integrator works with any boundary condition and any initial condition
out of the box, with no changes to the model code.

---

## Repository structure

```
qg-integrators/
│
├── qg_model.py                   # Original model (v1, backward compatible)
├── qg_model_v2.py                # Extended model (v2, this work)
├── boundary_conditions.py        # Dirichlet and channel BC handlers
│
├── integrators/
│   ├── __init__.py
│   ├── base.py                   # Abstract base + registry
│   ├── explicit.py               # Euler, Midpoint, RK4, DP5, SSPRK3, AB2/3/4
│   └── leapfrog.py               # LeapfrogRA, LeapfrogRAW
│
├── initial_conditions/
│   ├── __init__.py
│   └── ic_library.py             # zero, fourier, vortex, dipole,
│                                 # rossby_wave, band_noise, restart_npz/nc
│
├── analysis/
│   ├── convergence.py            # Temporal convergence test suite (NEW)
│   ├── jacobian_spectrum.py
│   ├── lyapunov.py
│   ├── stability_regions.py
│   ├── uncertainty.py
│   ├── plot_fields.py
│   ├── plot_alpha_sensitivity.py
│   └── plot_style.py
│
├── tests/
│   └── test_integrators.py       # pytest suite (all schemes x both BCs)
│
├── examples/
│   └── example_quickstart.py     # Three runnable examples
│
├── requirements.txt
└── README.md
```

---

## Time integration schemes

| Name | Order | RHS/step | Notes |
|------|-------|----------|-------|
| `euler` | 1 | 1 | Explicit Euler |
| `midpoint` | 2 | 2 | Explicit RK2 |
| `rk4` | 4 | 4 | Classical Runge-Kutta (QG-C default) |
| `dp5` | 5 | 6 | Dormand-Prince |
| `ssprk3` | 3 | 3 | Strong Stability Preserving RK |
| `ab2` | 2 | 1* | Adams-Bashforth 2-step |
| `ab3` | 3 | 1* | Adams-Bashforth 3-step |
| `ab4` | 4 | 1* | Adams-Bashforth 4-step |
| `leapfrog_ra` | 2 | 1* | + Robert-Asselin filter |
| `leapfrog_raw` | 2 | 1* | + Robert-Asselin-Williams filter |

*After RK4 bootstrap (multistep and leapfrog methods).

---

## Boundary conditions

| Name | Geometry | East/West | North/South |
|------|----------|-----------|-------------|
| `dirichlet` | Closed basin | psi = 0 | psi = 0 |
| `channel` | Zonal channel | Periodic | psi = 0 |

---

## Initial conditions

| Name | Description | Dirichlet | Channel |
|------|-------------|-----------|---------|
| `zero` | Rest state | yes | yes |
| `fourier` | Random spectral modes | sine basis | Fourier (periodic) |
| `vortex` | Gaussian vortex patches | yes | yes (min-image wrap) |
| `dipole` | Counter-rotating pair | yes | yes |
| `rossby_wave` | Linear Rossby wave | sine in x | cosine in x |
| `band_noise` | Band-limited random noise | yes | yes |
| `restart_npz` | Load from .npz | yes | yes |
| `restart_nc` | Load from NetCDF | yes | yes |

---

## Installation

```bash
git clone https://github.com/your-repo/qg-integrators
cd qg-integrators
pip install -r requirements.txt
```

No compiled extensions required.

---

## Quick start

### Python API

```python
from qg_model_v2 import QGModel, QGParams

# Closed basin, RK4 (backward compatible with v1)
prm = QGParams(scheme='rk4', bc='dirichlet', tend=5000)
m   = QGModel(prm)
psi, q = m.run('zero')

# Zonal channel, Adams-Bashforth 3rd-order
prm = QGParams(scheme='ab3', bc='channel', tend=10000, dt=0.5)
m   = QGModel(prm)
psi, q = m.run('fourier', amplitude=0.5, kmax=5)

# Diagnostics
print('KE =', m.kinetic_energy())
print('Z  =', m.enstrophy())
```

### Adding a new integrator (5 lines)

```python
from integrators.base import TimeIntegrator, register_integrator

@register_integrator('my_scheme')
class MyScheme(TimeIntegrator):
    @property
    def order(self): return 2
    def step(self, state, t, dt, rhs_fn):
        k1 = rhs_fn(state)
        k2 = rhs_fn(state + dt * k1)
        return state + 0.5 * dt * (k1 + k2)
```

### Command line

```bash
python qg_model_v2.py --scheme ab3 --bc channel --tend 10000 \
    --ic fourier --out qg_ab3_channel.npz

python qg_model_v2.py --scheme leapfrog_raw --bc channel \
    --ra-alpha 0.1 --raw --tend 50000 --ic zero --out qg_lf_raw.npz
```

### Convergence analysis

```bash
python analysis/convergence.py --both-bc --ic rossby_wave --out figures/
python analysis/convergence.py --schemes rk4 ab3 ssprk3 --bc channel
```

### Tests

```bash
pytest tests/test_integrators.py -v
```

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
pytest >= 7.0
netCDF4        (optional)
```

---

## Reference model

> Sakov, P. (2024). *QG-C: A quasi-geostrophic model in C*. https://github.com/sakov/qg-c

Key numerical methods:

> Arakawa, A. (1966). J. Comput. Phys. 1, 119-143.
> Asselin, R. (1972). Mon. Wea. Rev. 100, 487-490.
> Williams, P.D. (2009). Mon. Wea. Rev. 137, 2538-2546.
> Shu, C.-W. & Osher, S. (1988). J. Comput. Phys. 77, 439-471.
> Dormand, J.R. & Prince, P.J. (1980). J. Comput. Appl. Math. 6, 19-26.

---

## License

MIT — see LICENSE.
