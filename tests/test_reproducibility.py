# -*- coding: utf-8 -*-
"""Reproducibility tests — bit-exactness under fixed seeds.

These are the most important tests for a DA toolbox: two runs with the
same configuration must produce numerically identical results, otherwise
benchmarks are not comparable.
"""

import warnings

import numpy as np
import pytest

from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario, Benchmark


warnings.filterwarnings("ignore", category=Warning)


# ----------------------------------------------------------------------
# Scenario reproducibility
# ----------------------------------------------------------------------
class TestScenarioReproducibility:
    def _make(self, seed):
        m = Lorenz96(n=20)
        n = m.get_number_of_variables()
        return Scenario.generate(
            model=m,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=8,
            spinup_truth=1.0, pert_xb=0.5, spinup_xb=0.2,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.5, seed=seed,
        )

    def test_same_seed_bit_identical(self):
        s1 = self._make(42)
        s2 = self._make(42)
        assert np.array_equal(s1.x0_ref, s2.x0_ref)
        assert np.array_equal(s1.xb, s2.xb)
        assert np.array_equal(s1.initial_ensemble, s2.initial_ensemble)
        for a, b in zip(s1.truth_trajectory, s2.truth_trajectory):
            assert np.array_equal(a, b)
        for a, b in zip(s1.observations, s2.observations):
            assert np.array_equal(a, b)
        assert s1.meta["config_hash"] == s2.meta["config_hash"]

    def test_different_seed_different_observations(self):
        s1 = self._make(42)
        s2 = self._make(43)
        # x0_ref depends only on the model + spinup_truth, NOT on seed.
        assert np.array_equal(s1.x0_ref, s2.x0_ref)
        # But xb, ensemble, observations all differ
        assert not np.array_equal(s1.xb, s2.xb)
        assert not np.array_equal(s1.observations[0], s2.observations[0])


# ----------------------------------------------------------------------
# Benchmark reproducibility (the most important guarantee)
# ----------------------------------------------------------------------
class TestBenchmarkReproducibility:
    def _make_scenario(self):
        m = Lorenz96(n=20)
        n = m.get_number_of_variables()
        return Scenario.generate(
            model=m,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=8,
            spinup_truth=1.0, pert_xb=0.5, spinup_xb=0.2,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.5, seed=42,
        )

    def test_two_identical_benchmarks_match(self):
        scen = self._make_scenario()
        methods = {
            "EnKF":  dict(method="enkf"),
            "LETKF": dict(method="letkf", r=2),
        }
        kw = dict(scenarios=[scen], methods=methods,
                  n_runs_per_method=2, method_seed_base=1000,
                  parallel=False, verbose=False)
        r1 = Benchmark(**kw).run()
        r2 = Benchmark(**kw).run()

        # Per-cell bit-exact equality
        for row1, row2 in zip(r1.rows, r2.rows):
            assert row1["method"] == row2["method"]
            assert np.array_equal(row1["error_a"], row2["error_a"])
            assert np.array_equal(row1["error_b"], row2["error_b"])

    def test_method_seed_base_changes_results(self):
        scen = self._make_scenario()
        methods = {"EnKF": dict(method="enkf")}
        r1 = Benchmark(
            scenarios=[scen], methods=methods,
            n_runs_per_method=1, method_seed_base=1000,
            parallel=False, verbose=False,
        ).run()
        r2 = Benchmark(
            scenarios=[scen], methods=methods,
            n_runs_per_method=1, method_seed_base=2000,
            parallel=False, verbose=False,
        ).run()
        # Different method seed → different filter ensemble noise → different RMSE
        assert not np.array_equal(r1.rows[0]["error_a"],
                                   r2.rows[0]["error_a"])


# ----------------------------------------------------------------------
# Pre-loaded artifacts give the same scenario
# ----------------------------------------------------------------------
class TestPreloadedConsistency:
    """Computing a scenario from scratch vs from preloaded artifacts must
    give identical results, as long as the spinup parameters are passed
    correctly for the truth-sync step."""

    def test_preloaded_artifacts_match_full_compute(self):
        m = Lorenz96(n=20)
        n = m.get_number_of_variables()

        # Full compute
        s_full = Scenario.generate(
            model=m,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=6,
            spinup_truth=1.0, pert_xb=0.5, spinup_xb=0.2,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.4, seed=42,
        )

        # Preloaded (from s_full's artifacts)
        s_pre = Scenario.generate(
            model=m,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            x0_ref=s_full.x0_ref, xb=s_full.xb,
            initial_ensemble=s_full.initial_ensemble,
            spinup_xb=0.2, spinup_ensemble=0.1,  # for truth sync
            obs_freq=0.1, end_time=0.4, seed=42,
        )

        # Truth must be bit-identical (deterministic given x0_ref + spinups)
        for a, b in zip(s_full.truth_trajectory, s_pre.truth_trajectory):
            assert np.array_equal(a, b)
        # And the operators + noise are seeded by `seed`, so observations match too
        for a, b in zip(s_full.observations, s_pre.observations):
            assert np.array_equal(a, b)
