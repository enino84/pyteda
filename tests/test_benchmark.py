# -*- coding: utf-8 -*-
"""Tests for pyteda.experiments.Benchmark and BenchmarkResults diagnostics."""

import warnings

import numpy as np
import pytest

from pyteda.experiments import Benchmark


warnings.filterwarnings("ignore", category=Warning)


# ----------------------------------------------------------------------
# Benchmark grid
# ----------------------------------------------------------------------
class TestBenchmarkGrid:
    def test_runs_one_scenario_one_method(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf")},
            n_runs_per_method=1, parallel=False, verbose=False,
        ).run()
        assert len(results.rows) == 1

    def test_runs_M_scenarios_K_methods_R_runs(self, small_scenario):
        # 2 scenarios × 2 methods × 3 runs = 12 cells
        results = Benchmark(
            scenarios=[small_scenario, small_scenario],
            methods={"EnKF": dict(method="enkf"),
                     "LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=3, parallel=False, verbose=False,
        ).run()
        assert len(results.rows) == 12

    def test_summary_table_columns(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf"),
                     "LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
        ).run()
        df = results.summary_table()
        for col in ["method", "mean_rmse_a", "std_rmse_a", "mean_rmse_b",
                    "n", "mean_elapsed_s"]:
            assert col in df.columns
        assert len(df) == 2  # one row per method


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
class TestDiagnostics:
    def test_diagnostics_only_when_enabled(self, small_scenario):
        # Without store_diagnostics
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf")},
            n_runs_per_method=1, parallel=False, verbose=False,
            store_diagnostics=False,
        ).run()
        with pytest.raises(RuntimeError, match="store_diagnostics"):
            results.diagnostics_summary()

    def test_diagnostics_summary_columns(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf"),
                     "LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
            store_diagnostics=True,
        ).run()
        df = results.diagnostics_summary()
        for col in ["method", "mean_rmse_a", "mean_spread_a",
                    "spread_error_ratio", "mean_crps_a"]:
            assert col in df.columns

    def test_spread_error_ratio_finite_and_positive(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
            store_diagnostics=True,
        ).run()
        df = results.diagnostics_summary()
        assert np.isfinite(df["spread_error_ratio"].iloc[0])
        assert df["spread_error_ratio"].iloc[0] > 0

    def test_rank_counts_sum_correct(self, small_scenario):
        # rank_counts should sum to n_steps × n_state across all bins
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf")},
            n_runs_per_method=1, parallel=False, verbose=False,
            store_diagnostics=True,
        ).run()
        row = results.rows[0]
        n_state = small_scenario.n_state
        n_steps = small_scenario.n_steps
        assert row["rank_counts_a"].sum() == n_steps * n_state

    def test_crps_finite(self, small_scenario):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf")},
            n_runs_per_method=1, parallel=False, verbose=False,
            store_diagnostics=True,
        ).run()
        row = results.rows[0]
        assert np.isfinite(row["crps_a"]).all()
        # CRPS is non-negative by definition
        assert (row["crps_a"] >= 0).all()


# ----------------------------------------------------------------------
# export_csv
# ----------------------------------------------------------------------
class TestExportCSV:
    def test_writes_three_files_without_diagnostics(
        self, small_scenario, tmp_path
    ):
        results = Benchmark(
            scenarios=[small_scenario, small_scenario],
            methods={"EnKF": dict(method="enkf"),
                     "LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
            store_diagnostics=False,
        ).run()
        written = results.export_csv(tmp_path)

        # Without diagnostics: summary, summary_aggregated, error_curves
        # (no diagnostics_summary)
        assert "summary" in written
        assert "summary_aggregated" in written
        assert "error_curves" in written
        assert "diagnostics_summary" not in written
        for path in written.values():
            assert path.exists()

    def test_writes_four_files_with_diagnostics(
        self, small_scenario, tmp_path
    ):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
            store_diagnostics=True,
        ).run()
        written = results.export_csv(tmp_path)
        assert set(written.keys()) == {
            "summary", "summary_aggregated",
            "diagnostics_summary", "error_curves",
        }

    def test_summary_csv_row_count(self, small_scenario, tmp_path):
        """summary.csv has one row per (scenario × method × run) cell."""
        import pandas as pd
        results = Benchmark(
            scenarios=[small_scenario, small_scenario, small_scenario],
            methods={"EnKF": dict(method="enkf"),
                     "LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
        ).run()
        results.export_csv(tmp_path)
        df = pd.read_csv(tmp_path / "summary.csv")
        # 3 scenarios × 2 methods × 2 runs = 12 cells
        assert len(df) == 12
        assert set(df["method"].unique()) == {"EnKF", "LETKF"}

    def test_error_curves_csv_long_format(self, small_scenario, tmp_path):
        """error_curves.csv is long-format with cells × steps rows."""
        import pandas as pd
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=2, parallel=False, verbose=False,
        ).run()
        results.export_csv(tmp_path)
        df = pd.read_csv(tmp_path / "error_curves.csv")
        # 1 scenario × 1 method × 2 runs × n_steps
        assert len(df) == 2 * small_scenario.n_steps
        for col in ["method", "scenario_id", "run_id", "step",
                    "time", "error_b", "error_a"]:
            assert col in df.columns

    def test_diagnostics_columns_present_when_enabled(
        self, small_scenario, tmp_path
    ):
        import pandas as pd
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"LETKF": dict(method="letkf", r=2)},
            n_runs_per_method=1, parallel=False, verbose=False,
            store_diagnostics=True,
        ).run()
        results.export_csv(tmp_path)
        df_summary = pd.read_csv(tmp_path / "summary.csv")
        df_curves = pd.read_csv(tmp_path / "error_curves.csv")
        for col in ["mean_spread_a", "mean_crps_a", "spread_error_ratio_a"]:
            assert col in df_summary.columns
        for col in ["spread_a", "crps_a", "spread_b", "crps_b"]:
            assert col in df_curves.columns

    def test_creates_directory_if_missing(self, small_scenario, tmp_path):
        results = Benchmark(
            scenarios=[small_scenario],
            methods={"EnKF": dict(method="enkf")},
            n_runs_per_method=1, parallel=False, verbose=False,
        ).run()
        target = tmp_path / "deep" / "nested" / "dir"
        results.export_csv(target)
        assert target.exists()
        assert (target / "summary.csv").exists()
