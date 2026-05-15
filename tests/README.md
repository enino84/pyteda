# TEDA test suite

Run all tests:

```bash
pytest tests/
```

Run a single file:

```bash
pytest tests/test_localization.py -v
```

Run a single test:

```bash
pytest tests/test_models.py::TestLorenz96::test_propagate_shape -v
```

## Test files

| File | What it covers |
|------|----------------|
| `test_localization.py` | `resolve_radius`, `pairwise_radius`, the three forms of `r` (int, dict, ndarray) |
| `test_models.py` | The `Model` contract for `Lorenz96`, `QGModel`, `SWEModel` |
| `test_observation.py` | `LinearSelection`, `LinearMatrix`, `NonlinearOperator`, `IsotropicDiagonal`, `HeterogeneousDiagonal`, `DenseCovariance` |
| `test_io.py` | netCDF and npz roundtrips for state vectors, ensembles, truth, observations; `get_data_dir` |
| `test_scenario.py` | The 3-phase ensemble recipe, pre-computed artifact reuse, legacy aliases, `Scenario.save` / `load` |
| `test_filters.py` | All 10 registered analyses run, `r` dispatch is consistent across forms |
| `test_benchmark.py` | `Benchmark` grid, diagnostics (`spread`, `CRPS`, `rank histogram`) |
| `test_reproducibility.py` | Bit-exact reproducibility under fixed seeds; pre-loaded artifacts match full compute |

The full suite runs in well under 10 seconds on a laptop.
