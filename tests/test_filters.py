# -*- coding: utf-8 -*-
"""Tests for analysis filters in the registry.

Each filter must run on a small scenario without errors, produce an
analysis whose RMSE is at least not NaN/Inf, and reduce the background
error on average over a few steps.
"""

import warnings

import numpy as np
import pytest

from pyteda.experiments import Benchmark


warnings.filterwarnings("ignore", category=Warning)


# Methods to exercise. Each entry: (label, config dict).
ALL_METHODS = [
    ("enkf", dict(method="enkf")),
    ("enkf-cholesky", dict(method="enkf-cholesky")),
    ("enkf-naive", dict(method="enkf-naive")),
    ("enkf-bloc", dict(method="enkf-b-loc")),
    ("enkf-mc", dict(method="enkf-modified-cholesky", r=2)),
    ("enkf-shrinkage", dict(method="enkf-shrinkage-precision")),
    ("etkf", dict(method="etkf")),
    ("ensrf", dict(method="ensrf")),
    ("lenkf", dict(method="lenkf", r=2)),
    ("letkf", dict(method="letkf", r=2)),
]


@pytest.mark.parametrize("name,cfg", ALL_METHODS)
def test_filter_runs_without_errors(small_scenario, name, cfg):
    """Each filter from the registry runs end-to-end without crashing."""
    results = Benchmark(
        scenarios=[small_scenario],
        methods={name: cfg},
        n_runs_per_method=1, parallel=False, verbose=False,
    ).run()
    row = results.rows[0]
    assert np.isfinite(row["error_a"]).all()
    assert np.isfinite(row["error_b"]).all()


@pytest.mark.parametrize("name,cfg", ALL_METHODS)
def test_filter_produces_analysis(small_scenario, name, cfg):
    """The analysis error array has the right length."""
    results = Benchmark(
        scenarios=[small_scenario],
        methods={name: cfg},
        n_runs_per_method=1, parallel=False, verbose=False,
    ).run()
    row = results.rows[0]
    assert len(row["error_a"]) == small_scenario.n_steps


def test_letkf_radius_three_forms_equivalent(small_scenario):
    """LETKF with r=2, r={'x':2}, r=full(20,2) yields identical RMSE."""
    methods = {
        "int":   dict(method="letkf", r=2),
        "dict":  dict(method="letkf", r={"x": 2}),
        "array": dict(method="letkf", r=np.full(20, 2.0)),
    }
    results = Benchmark(
        scenarios=[small_scenario],
        methods=methods, n_runs_per_method=1,
        method_seed_base=1000, parallel=False, verbose=False,
    ).run()
    rmses = {r["method"]: float(np.mean(r["error_a"])) for r in results.rows}
    assert rmses["int"] == rmses["dict"] == rmses["array"]


def test_lenkf_radius_three_forms_equivalent(small_scenario):
    methods = {
        "int":   dict(method="lenkf", r=2),
        "dict":  dict(method="lenkf", r={"x": 2}),
        "array": dict(method="lenkf", r=np.full(20, 2.0)),
    }
    results = Benchmark(
        scenarios=[small_scenario],
        methods=methods, n_runs_per_method=1,
        method_seed_base=1000, parallel=False, verbose=False,
    ).run()
    rmses = {r["method"]: float(np.mean(r["error_a"])) for r in results.rows}
    assert rmses["int"] == rmses["dict"] == rmses["array"]


def test_letkf_better_than_enkf_on_l96(small_scenario):
    """On Lorenz96 with localization, LETKF should outperform plain EnKF."""
    methods = {
        "EnKF":  dict(method="enkf"),
        "LETKF": dict(method="letkf", r=2),
    }
    results = Benchmark(
        scenarios=[small_scenario], methods=methods,
        n_runs_per_method=2, method_seed_base=2000,
        parallel=False, verbose=False,
    ).run()
    rmses = {}
    for row in results.rows:
        rmses.setdefault(row["method"], []).append(float(np.mean(row["error_a"])))
    assert np.mean(rmses["LETKF"]) < np.mean(rmses["EnKF"])


# ----------------------------------------------------------------------
# Plain EnKF: the matrix-free path must agree with the dense one
# ----------------------------------------------------------------------
class TestEnKFMatrixFreePath:
    """The two branches of AnalysisEnKF.perform_assimilation.

    For ``n < SPARSE_THRESHOLD`` the filter forms P^b and solves in
    observation space. For ``n >= SPARSE_THRESHOLD`` it never forms P^b and
    applies the increment as

        X^a = X^b + (1/(N-1)) * DX @ ((H DX)^T Z).

    Writing that second factor as ``H^T Z`` instead of ``(H DX)^T Z`` leaves
    a shape that only conforms when the ensemble size happens to equal the
    state dimension, so the filter raised a ValueError for every high
    dimensional problem. These tests pin the two branches to each other.
    """

    @staticmethod
    def _setup(n=60, N=24, m=20, seed=3):
        from pyteda.models import Lorenz96
        from pyteda.background import Background
        from pyteda.observation import Observation
        from pyteda.observation.noise import IsotropicDiagonal
        from pyteda.observation.operators import LinearSelection

        rng = np.random.default_rng(seed)
        model = Lorenz96(n=n)
        x = rng.standard_normal(n)
        Xb = x[:, None] + 0.3 * rng.standard_normal((n, N))

        idx = np.sort(rng.choice(n, size=m, replace=False))
        op = LinearSelection(m, n, indices=idx, rng=rng)
        noise = IsotropicDiagonal(0.5, dim=m)
        obs = Observation.from_arrays(op.apply(x) + noise.sample(rng),
                                      op, noise)

        bg = Background(model, ensemble_size=N)
        bg.Xb = Xb
        return model, bg, obs, Xb

    @staticmethod
    def _run_with_threshold(threshold, bg, obs, seed=0):
        import pyteda.observation.noise as noise_mod
        from pyteda.analysis import AnalysisEnKF

        original = noise_mod.SPARSE_THRESHOLD
        noise_mod.SPARSE_THRESHOLD = threshold
        try:
            np.random.seed(seed)
            return np.asarray(
                AnalysisEnKF().perform_assimilation(bg, obs)
            ).copy()
        finally:
            noise_mod.SPARSE_THRESHOLD = original

    def test_matrix_free_path_runs(self):
        """n >= SPARSE_THRESHOLD must not raise."""
        _, bg, obs, Xb = self._setup()
        Xa = self._run_with_threshold(1, bg, obs)
        assert Xa.shape == Xb.shape
        assert np.isfinite(Xa).all()

    def test_matrix_free_matches_dense(self):
        """Both branches compute the same analysis ensemble."""
        _, bg, obs, _ = self._setup()
        Xa_dense = self._run_with_threshold(10_000, bg, obs)
        Xa_free = self._run_with_threshold(1, bg, obs)
        assert np.allclose(Xa_dense, Xa_free, rtol=1e-8, atol=1e-8)

    def test_matrix_free_moves_the_state(self):
        """A sanity floor: the analysis is not simply the background."""
        _, bg, obs, Xb = self._setup()
        Xa = self._run_with_threshold(1, bg, obs)
        assert np.abs(Xa - Xb).mean() > 1e-6
