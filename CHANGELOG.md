# Changelog

> Numbering note: this release is 0.6.0, not 0.4.0. PyPI already carries a
> pyteda 0.5.0, uploaded on 2026-06-19, whose code is byte-identical to the
> `0.3.0` written in the repository's `pyproject.toml`: the version was
> bumped at upload time and never written back. Anything numbered below
> 0.5.0 would be refused by PyPI and would never be picked up by
> `pip install -U pyteda`.

## 0.6.0 (2026-10-12)

**The geophysical models moved out of pyteda.** `QGModel` and `SWEModel` used
to carry a vendored copy of the physics inside the package. They are now thin
adapters over the published packages, and pyteda implements no model physics
beyond `Lorenz96`:

| Model | Backend | Extra |
|---|---|---|
| `QGModel` | `qg-python` >= 1.1.0 | `pip install pyteda[qg]` |
| `SWEModel` | `swesphere` >= 0.4.0 | `pip install pyteda[swe]` |

Removed from the source tree: `pyteda/models/qg/` (the vendored QG core,
boundary conditions, integrator registry and IC library) and
`pyteda/models/swe/{_dynamics,_grid,_initial_conditions}.py`.

### Correctness

- **The localized filters were univariate and there was no way to say
  otherwise.** `SWEModel` and `QGModel` have always accepted a `cross` flag on
  `get_ngb`, `get_pre` and `create_decorrelation_matrix`, but no filter passed
  it, so every localized filter ran with the default `cross=False`: the
  Cholesky factor and the decorrelation matrix came out block diagonal by
  variable, and observing one variable left the others at exactly their
  forecast value. Measured on `two_jets` at LMAX 4 (state 600, ensemble 40)
  with a 50-point network inside the `h` block, the increment in `u` and `v`
  was identically zero for LETKF, LEnKF, EnKF-BLoc, EnKF-MC and
  shrinkage-precision, while the unlocalized `enkf` moved them by about
  3e-2 m/s. `cross` is now a filter argument and reaches the model, so the
  choice between a univariate and a multivariate analysis is explicit; the
  default stays `False` so existing results are unchanged. **Any conclusion
  drawn about cross-variable updates under a localized filter should be
  rechecked with `cross=True`.**
- **Plain `enkf` raised `ValueError` for every state of dimension >= 500.**
  Its matrix-free branch applied the increment as `DX @ (H.T @ Z)` instead of
  `DX @ ((H DX).T @ Z)`; the shapes only conform when the ensemble size equals
  the state dimension, so the filter crashed on every SWE and QG
  configuration rather than returning a wrong answer. The two branches are
  now pinned to each other by a test. This defect predates this release.

- **The shallow-water continuity equation was wrong in the vendored copy.** It
  integrated `dh/dt = -h*div(V)`, which omits the advection of `h`; the
  equation is `dh/dt = -div(h V)`. `swesphere` discretizes it in flux form.
  Measured here on the unforced regime, the relative drift of the global mass
  over two days falls from about 1e-3 to 5e-5. **Any result produced with the
  vendored model should be regenerated**, including the assimilation
  experiments that used `SWEModel`.
- The shallow-water integrator changes from explicit Euler to RK4, which is
  what `swesphere` ships and tests.

### Fixed

- **A subset state leaked the unestimated fields between ensemble members.**
  With `state_vars` a strict subset (for instance `['h']`), the adapter kept
  the absent fields in a cache that `propagate` overwrote on every call. Since
  pyteda forecasts an ensemble by looping `model.propagate(Xb[:, e], T)` on one
  model object, member *e* was integrated with member *e-1*'s wind and the
  result depended on the order of the loop. Measured before the fix on a
  one-hour forecast of `two_jets` at LMAX 8, reversing the order changed a
  member's forecast by up to 8.4 m. The absent fields are now a fixed
  reference that `propagate` reads and never writes, so forecasts are
  member-independent and reproducible, and `set_reference_fields()` is there
  to advance them deliberately. The constructor warns when a subset is
  requested and points to the observation-operator route. **This defect
  predates this release**; results from cycled runs that used a subset state should
  be regenerated.

- `tests/test_benchmark.py::TestExportCSV::test_error_curves_csv_long_format`
  asserted `2 * n_steps` rows in `error_curves.csv`, which stopped being true
  when per-variable error curves were added: the file carries a `global` row
  plus one row per state-variable block for every step. The export was right
  and the assertion was stale; the test now checks the real long format
  (`runs x steps x (1 + blocks)`, the `variable` column, and the global series
  on its own). This failure predates this release.

- The `swe` extra now requires `swesphere>=0.4.0`. Up to 0.3.0 the backend's
  spectral filter restarted its schedule on every `propagate` call, so the
  length of the forecast window silently changed the model: a cycling
  experiment with a short observation frequency ran different physics from one
  with a long frequency, and at the default resolution a short enough window
  blew the integration up. The adapter already passed the running time, so
  installing 0.4.0 is the whole fix. Any cycled SWE result produced with
  swesphere 0.3.0 should be regenerated.

### Added

- `SWEModel(preset=...)`: the three characterized `swesphere` regimes,
  `waves`, `one_jet` and `two_jets`, with `two_jets` the default. `preset=None`
  reproduces the previous bare setup (Williamson TC2 + Rossby, unforced).
- `QGModel(preset=...)`: `sakov_oke_stationary` (the default, the regime used
  in assimilation testbeds), `qgc_default`, or `None` to build from arguments.
- `SWEModel.interior_mask()`, `QGModel.interior_mask()`, `QGModel.unpack()`
  and `QGModel.psi_from_q()`, forwarded from the backends.
- `pyteda.models.available_models()` reports which models this installation
  can build and, for the others, the pip command that would enable them.
- Optional-dependency extras `qg`, `swe` and `all` in `pyproject.toml`.
- `cross` on every localized filter (`letkf`, `lenkf`, `enkf-b-loc`,
  `enkf-modified-cholesky`, `enkf-shrinkage-precision`,
  `enkf-obs-modified-cholesky`, its local variant, and
  `enkf-obs-woodbury-cholesky`), plus `cross_scale` on `enkf-b-loc`. With
  `cross=False` (the default) each variable block is localized on its own and
  the analysis is univariate; with `cross=True` a local domain or predecessor
  set also contains the same grid cell in the other blocks, so an observation
  of one variable reaches the others through the ensemble cross covariances.
  `pyteda.analysis._loc_utils` documents the choice and forwards it to any
  model, raising a named `TypeError` rather than silently going univariate if
  a model cannot honor `cross=True`.
- `cross` on `Lorenz96.get_ngb`, `get_pre` and `create_decorrelation_matrix`,
  accepted and ignored: one variable block, so there is nothing to cross to.
- `tests/test_cross_variable.py`, which pins both branches of the choice.

### Changed

- `pyteda.models` imports the two adapters lazily, so `import pyteda` works
  with neither extra installed. Asking for a model you have not installed
  raises an `ImportError` that names the install command.
- `pyshtools` is no longer a core dependency; it comes with `pyteda[swe]`
  through `swesphere`.
- Ensemble members (`get_initial_condition(seed=...)` with a non-zero seed) are
  now generated by perturbing the preset's own state with a seeded
  low-wavenumber field scaled to the field's standard deviation. Generating
  ensemble members is the assimilation library's job, so this lives in pyteda
  and not in the model package. The perturbation is unbalanced on purpose; spin
  members up before sampling.
- `SWEModel._perturb` scales `u` and `v` by one wind scale taken from both
  components together, instead of each by its own standard deviation. In a
  purely zonal preset such as `two_jets` the reference has `v == 0` everywhere,
  so the old rule left `v` unperturbed: the ensemble carried no spread in `v`
  and no filter could correct it, whatever was observed.

### Removed

Two `QGModel` arguments are gone because the published `qg-python` does not
implement them. Passing either one warns and continues:

- `bc='channel'` — `qg-python` implements the Dirichlet basin only.
- `raw_beta` — `qg-python` uses the standard RAW beta.

The backend also offers fewer schemes and initial conditions than the vendored
fork did:

| | vendored (0.3.0) | qg-python 1.1.0 |
|---|---|---|
| schemes | euler, midpoint, rk4, dp5, ssprk3, ab2, ab3, ab4, leapfrog_ra, leapfrog_raw | order1, order2, rk4, dp5, leapfrog |
| ICs | zero, fourier, vortex, dipole, rossby_wave, band_noise, restart_npz, restart_nc | zero, fourier, vortex, restart |

Nothing in pyteda's tests or examples used the missing ones. If you need them,
the right fix is to upstream them into `qg-python` rather than to vendor the
code again.
