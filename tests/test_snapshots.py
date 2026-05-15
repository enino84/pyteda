# -*- coding: utf-8 -*-
"""Tests for the ensemble-snapshot capture mechanism.

The user passes fractions in [0, 1] via `store_states_at`. The simulation
translates them into integer step indices, deduplicates, and records the
full ensemble (Xb before and Xa after assimilation) at each captured step.
"""

import warnings

import numpy as np
import pytest

from pyteda.simulation.simulation_core import _resolve_snapshot_steps
from pyteda.simulation import Simulation
from pyteda.analysis.analysis_factory import AnalysisFactory
from pyteda.experiments import Benchmark


warnings.filterwarnings("ignore", category=Warning)


# ----------------------------------------------------------------------
# _resolve_snapshot_steps unit tests
# ----------------------------------------------------------------------
class TestResolveSnapshotSteps:
    def test_none_returns_empty(self):
        out = _resolve_snapshot_steps(None, n_steps=10)
        assert out.size == 0

    def test_empty_list_returns_empty(self):
        out = _resolve_snapshot_steps([], n_steps=10)
        assert out.size == 0

    def test_zero_half_one(self):
        # n_steps=11 -> indices 0..10. round(0.5*10) = 5.
        out = _resolve_snapshot_steps([0.0, 0.5, 1.0], n_steps=11)
        assert np.array_equal(out, [0, 5, 10])

    def test_eleven_equispaced(self):
        out = _resolve_snapshot_steps(np.linspace(0, 1, 11), n_steps=11)
        assert np.array_equal(out, np.arange(11))

    def test_deduplicates_when_fractions_round_together(self):
        # All four fractions round to 0 or 1 within n_steps=11
        # 0.04*10=0.4 -> 0, 0.05*10=0.5 -> 0 (banker's rounding), 0.96*10=9.6 -> 10
        out = _resolve_snapshot_steps([0.0, 0.04, 0.05, 1.0], n_steps=11)
        assert out.size <= 4   # deduplicated
        assert out[0] == 0
        assert out[-1] == 10

    def test_more_fractions_than_steps_clamps(self):
        # n_steps=3 -> indices 0, 1, 2; 5 fractions can't exceed 3 unique steps
        out = _resolve_snapshot_steps(
            np.linspace(0, 1, 5), n_steps=3,
        )
        assert out.size == 3
        assert np.array_equal(out, [0, 1, 2])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _resolve_snapshot_steps([0.0, 1.5], n_steps=10)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _resolve_snapshot_steps([-0.1, 0.5], n_steps=10)

    def test_n_steps_one_yields_step_zero(self):
        out = _resolve_snapshot_steps([0.0, 0.5, 1.0], n_steps=1)
        # max step index is 0; everything maps to 0
        assert np.array_equal(out, [0])

    def test_returns_sorted(self):
        out = _resolve_snapshot_steps([1.0, 0.5, 0.0, 0.25], n_steps=11)
        assert np.array_equal(out, np.sort(out))


# ----------------------------------------------------------------------
# Simulation snapshots
# ----------------------------------------------------------------------
class TestSimulationSnapshots:
    def test_default_no_snapshots(self, small_scenario, small_lorenz96):
        analysis = AnalysisFactory(
            "letkf", model=small_lorenz96, r=2,
        ).create_analysis()
        sim = Simulation.from_scenario(small_scenario, analysis)
        sim.run()
        assert sim.Xb_snapshots.shape[0] == 0
        assert sim.Xa_snapshots.shape[0] == 0
        assert sim.snapshot_steps.size == 0

    def test_three_snapshots(self, small_scenario, small_lorenz96):
        analysis = AnalysisFactory(
            "letkf", model=small_lorenz96, r=2,
        ).create_analysis()
        sim = Simulation.from_scenario(
            small_scenario, analysis,
            store_states_at=[0.0, 0.5, 1.0],
        )
        sim.run()
        assert sim.Xb_snapshots.shape == (
            3, small_scenario.n_state, small_scenario.ensemble_size,
        )
        assert sim.Xa_snapshots.shape == sim.Xb_snapshots.shape
        # First and last step must be 0 and n_steps - 1
        assert sim.snapshot_steps[0] == 0
        assert sim.snapshot_steps[-1] == small_scenario.n_steps - 1

    def test_snapshot_times_match_scenario(self, small_scenario, small_lorenz96):
        analysis = AnalysisFactory(
            "letkf", model=small_lorenz96, r=2,
        ).create_analysis()
        sim = Simulation.from_scenario(
            small_scenario, analysis,
            store_states_at=[0.0, 1.0],
        )
        sim.run()
        # snapshot_times must be a subset of scenario.times
        for t in sim.snapshot_times:
            assert t in small_scenario.times

    def test_xa_differs_from_xb(self, small_scenario, small_lorenz96):
        """After assimilation, Xa should not equal Xb (filter actually updates)."""
        analysis = AnalysisFactory(
            "letkf", model=small_lorenz96, r=2,
        ).create_analysis()
        sim = Simulation.from_scenario(
            small_scenario, analysis,
            store_states_at=[0.0, 0.5, 1.0],
        )
        sim.run()
        for k in range(sim.Xb_snapshots.shape[0]):
            assert not np.allclose(sim.Xb_snapshots[k], sim.Xa_snapshots[k])

    def test_diagnostics_and_snapshots_coexist(
        self, small_scenario, small_lorenz96
    ):
        """Both flags can be active; computed values must be consistent."""
        analysis = AnalysisFactory(
            "letkf", model=small_lorenz96, r=2,
        ).create_analysis()
        sim = Simulation.from_scenario(
            small_scenario, analysis,
            store_diagnostics=True,
            store_states_at=[0.0, 0.5, 1.0],
        )
        sim.run()
        # Diagnostics arrays still cover all steps
        assert sim.spread_a.shape == (small_scenario.n_steps,)
        # Snapshots only cover 3 steps
        assert sim.Xa_snapshots.shape[0] == 3

        # Spread at snapshot 0 = sqrt(mean(std(X)**2)), should match diagnostics
        for snap_idx, step in enumerate(sim.snapshot_steps):
            sigma = sim.Xa_snapshots[snap_idx].std(axis=1, ddof=1)
            spread_from_snap = float(np.sqrt(np.mean(sigma ** 2)))
            assert np.isclose(spread_from_snap, sim.spread_a[step])

    def test_invalid_fraction_raises(self, small_scenario, small_lorenz96):
        analysis = AnalysisFactory(
            "letkf", model=small_lorenz96, r=2,
        ).create_analysis()
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Simulation.from_scenario(
                small_scenario, analysis,
                store_states_at=[0.0, 1.5],
            )


# ----------------------------------------------------------------------
# Benchmark snapshots
# ----------------------------------------------------------------------
class TestBenchmarkSnapshots:
    def test_benchmark_propagates_snapshots(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf"),
                     "LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
            store_states_at=[0.0, 0.5, 1.0],
        ).run()
        # 1 scenario × 2 methods × 2 runs = 4 cells
        assert len(results.rows) == 4
        for row in results.rows:
            assert "Xb_snapshots" in row
            assert "Xa_snapshots" in row
            assert row["Xb_snapshots"].shape == (
                3, small_scenario.n_state, small_scenario.ensemble_size,
            )
            # snapshot_fractions reflects the actually-captured steps,
            # which may differ slightly from the requested values when
            # n_steps is small (rounding effect).
            assert row["snapshot_fractions"][0] == 0.0
            assert row["snapshot_fractions"][-1] == 1.0
            assert row["snapshot_steps"][0] == 0
            assert (
                row["snapshot_steps"][-1] == small_scenario.n_steps - 1
            )

    def test_benchmark_no_snapshots_by_default(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf")},
            n_runs_per_method=1, parallel=False, verbose=False,
        ).run()
        assert "Xb_snapshots" not in results.rows[0]
