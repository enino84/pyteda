# -*- coding: utf-8 -*-
"""
Benchmark: run many filters across many scenarios with reproducible seeds.

A `Benchmark` evaluates a set of filter configurations across a list of
`Scenario`s, with `n_runs_per_method` repeated runs per (scenario, method)
cell. Each repeat uses a different deterministic seed for the filter's
internal randomness (e.g. EnKF perturbations) but sees exactly the same
truth, observations, and initial ensemble.

The result is a long-form pandas DataFrame:

    scenario_id | method | run_id | step | t | error_b | error_a

plus convenience methods for summary tables and plots.

Parallelism is provided via joblib when available; otherwise runs are
sequential.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..analysis.analysis_factory import AnalysisFactory
from ..simulation import Simulation
from .scenario import Scenario


# ----------------------------------------------------------------------
# Single-cell runner (pickle-safe for joblib)
# ----------------------------------------------------------------------
def _run_one_cell(
    scenario: Scenario,
    scenario_id: int,
    method_name: str,
    method_cfg: Dict[str, Any],
    run_id: int,
    method_seed: int,
    inflation_factor: float,
    store_diagnostics: bool = False,
    store_states_at=None,
):
    """Run one (scenario, method, run) cell and return a dict of results."""
    cfg = dict(method_cfg)
    method = cfg.pop("method")
    # Inject the model from the scenario if the analysis class needs it.
    cfg.setdefault("model", scenario.model)
    analysis = AnalysisFactory(method, **cfg).create_analysis()

    method_rng = np.random.default_rng(method_seed)
    t0 = time.perf_counter()
    sim = Simulation.from_scenario(
        scenario,
        analysis,
        inflation_factor=inflation_factor,
        method_rng=method_rng,
        store_diagnostics=store_diagnostics,
        store_states_at=store_states_at,
    )
    sim.run()
    elapsed = time.perf_counter() - t0
    errb, erra = sim.get_errors()
    out = {
        "scenario_id": scenario_id,
        "method": method_name,
        "run_id": run_id,
        "method_seed": method_seed,
        "elapsed": elapsed,
        "times": np.asarray(scenario.times),
        "error_b": np.asarray(errb),
        "error_a": np.asarray(erra),
    }
    if store_diagnostics:
        out["spread_b"] = np.asarray(sim.spread_b)
        out["spread_a"] = np.asarray(sim.spread_a)
        out["crps_b"] = np.asarray(sim.crps_b)
        out["crps_a"] = np.asarray(sim.crps_a)
        out["rank_counts_b"] = np.asarray(sim.rank_counts_b)
        out["rank_counts_a"] = np.asarray(sim.rank_counts_a)
    if sim.snapshot_steps.size > 0:
        out["Xb_snapshots"] = np.asarray(sim.Xb_snapshots)
        out["Xa_snapshots"] = np.asarray(sim.Xa_snapshots)
        out["snapshot_steps"] = np.asarray(sim.snapshot_steps)
        out["snapshot_times"] = np.asarray(sim.snapshot_times)
        out["snapshot_fractions"] = np.asarray(sim.snapshot_fractions)
    return out


# ----------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------
@dataclass
class BenchmarkResults:
    rows: List[Dict[str, Any]]
    method_names: List[str] = field(default_factory=list)
    n_scenarios: int = 0
    n_runs_per_method: int = 0

    # Lazy-import pandas so it stays optional at install time.
    @staticmethod
    def _pd():
        import pandas as pd
        return pd

    def to_dataframe(self):
        """Long-form DataFrame: one row per (scenario, method, run, step)."""
        pd = self._pd()
        records = []
        for r in self.rows:
            for k, (t, eb, ea) in enumerate(zip(r["times"], r["error_b"], r["error_a"])):
                records.append({
                    "scenario_id": r["scenario_id"],
                    "method": r["method"],
                    "run_id": r["run_id"],
                    "step": k,
                    "t": float(t),
                    "error_b": float(eb),
                    "error_a": float(ea),
                })
        return pd.DataFrame.from_records(records)

    def summary_table(self, kind: str = "mean"):
        """Per-method summary of the time-averaged analysis RMSE.

        Returns a DataFrame with columns: method, mean, std, median, n_runs,
        elapsed_mean — aggregated across all scenarios and runs.
        """
        pd = self._pd()
        recs = []
        for r in self.rows:
            recs.append({
                "method": r["method"],
                "scenario_id": r["scenario_id"],
                "run_id": r["run_id"],
                "elapsed": r["elapsed"],
                "rmse_a_time_avg": float(np.mean(r["error_a"])),
                "rmse_b_time_avg": float(np.mean(r["error_b"])),
            })
        df = pd.DataFrame.from_records(recs)
        agg = df.groupby("method").agg(
            mean_rmse_a=("rmse_a_time_avg", "mean"),
            std_rmse_a=("rmse_a_time_avg", "std"),
            median_rmse_a=("rmse_a_time_avg", "median"),
            mean_rmse_b=("rmse_b_time_avg", "mean"),
            n=("rmse_a_time_avg", "count"),
            mean_elapsed_s=("elapsed", "mean"),
        ).reset_index()
        return agg.sort_values("mean_rmse_a").reset_index(drop=True)

    def compare(self, method_a: str, method_b: str):
        """Wilcoxon signed-rank test of time-averaged RMSE_a between two
        methods, paired by (scenario, run). Returns a dict."""
        from scipy import stats

        df = self.to_dataframe()
        agg = (df.groupby(["method", "scenario_id", "run_id"])["error_a"]
                 .mean().reset_index())
        a = agg[agg.method == method_a].sort_values(["scenario_id", "run_id"])
        b = agg[agg.method == method_b].sort_values(["scenario_id", "run_id"])
        if len(a) != len(b):
            raise ValueError("Methods must have the same number of paired runs.")
        diff = a["error_a"].values - b["error_a"].values
        try:
            stat, p = stats.wilcoxon(a["error_a"].values, b["error_a"].values)
        except ValueError:
            stat, p = float("nan"), 1.0
        return {
            "method_a": method_a,
            "method_b": method_b,
            "mean_a": float(a["error_a"].mean()),
            "mean_b": float(b["error_a"].mean()),
            "mean_diff_a_minus_b": float(diff.mean()),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "n_pairs": int(len(diff)),
        }

    def plot_error_curves(self, ax=None, kind: str = "analysis", q_low=0.25, q_high=0.75):
        """Plot mean error curves with an inter-quartile band per method.

        Aggregates over (scenario, run) to produce one curve per method.
        """
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        col = "error_a" if kind == "analysis" else "error_b"
        agg = (df.groupby(["method", "step", "t"])[col]
                 .agg(median="median", q_low=lambda s: s.quantile(q_low),
                      q_high=lambda s: s.quantile(q_high))
                 .reset_index())

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        for method, sub in agg.groupby("method"):
            sub = sub.sort_values("t")
            ax.plot(sub["t"], sub["median"], label=method)
            ax.fill_between(sub["t"], sub["q_low"], sub["q_high"], alpha=0.2)

        ax.set_xlabel("time")
        ax.set_ylabel(f"RMSE ({kind})")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Diagnostics (require store_diagnostics=True at run time)
    # ------------------------------------------------------------------
    def _check_diagnostics(self):
        """Raise a helpful error if diagnostics were not recorded."""
        if not self.rows or "spread_a" not in self.rows[0]:
            raise RuntimeError(
                "Diagnostics not stored. Re-run the Benchmark with "
                "`store_diagnostics=True`."
            )

    def diagnostics_summary(self):
        """Per-method summary of calibration diagnostics.

        Returns a DataFrame with one row per method, aggregating over
        scenarios × runs × time:

            mean_rmse_a    : time-averaged analysis RMSE (same as summary_table)
            mean_spread_a  : time-averaged analysis spread
            spread_error_ratio : mean_spread_a / mean_rmse_a
                            (≈ 1 means well-calibrated; <1 sub-disperse;
                             >1 over-disperse)
            mean_crps_a    : time-averaged analysis CRPS
            mean_rmse_b / mean_spread_b / mean_crps_b — same for background.
        """
        self._check_diagnostics()
        pd = self._pd()
        recs = []
        for r in self.rows:
            recs.append({
                "method": r["method"],
                "scenario_id": r["scenario_id"],
                "run_id": r["run_id"],
                "rmse_a": float(np.mean(r["error_a"])),
                "rmse_b": float(np.mean(r["error_b"])),
                "spread_a": float(np.mean(r["spread_a"])),
                "spread_b": float(np.mean(r["spread_b"])),
                "crps_a": float(np.mean(r["crps_a"])),
                "crps_b": float(np.mean(r["crps_b"])),
            })
        df = pd.DataFrame.from_records(recs)
        agg = df.groupby("method").agg(
            mean_rmse_a=("rmse_a", "mean"),
            mean_spread_a=("spread_a", "mean"),
            mean_crps_a=("crps_a", "mean"),
            mean_rmse_b=("rmse_b", "mean"),
            mean_spread_b=("spread_b", "mean"),
            mean_crps_b=("crps_b", "mean"),
            n=("rmse_a", "count"),
        ).reset_index()
        agg["spread_error_ratio"] = agg["mean_spread_a"] / agg["mean_rmse_a"]
        # Reorder columns so the most-read metrics come first.
        cols = ["method", "mean_rmse_a", "mean_spread_a",
                "spread_error_ratio", "mean_crps_a",
                "mean_rmse_b", "mean_spread_b", "mean_crps_b", "n"]
        return agg[cols].sort_values("mean_rmse_a").reset_index(drop=True)

    def export_csv(self, directory, burn_in_frac=0.0):
        """Write four CSVs that fully describe the benchmark.

        Files written under ``directory`` (created if needed):

          summary.csv
              One row per (scenario, method, run) with mean/median/final
              RMSE for analysis and background, and — if diagnostics were
              recorded — spread, CRPS, and the spread/error ratio.
              This is the file users will most often re-aggregate
              externally for boxplots, t-tests, etc.

          summary_aggregated.csv
              The output of ``summary_table()`` — one row per method,
              aggregated over scenarios and runs.

          diagnostics_summary.csv
              The output of ``diagnostics_summary()``. Only written if
              the benchmark was run with ``store_diagnostics=True``.

          error_curves.csv
              Long-format table with one row per (scenario, method, run,
              step), suitable for re-plotting with seaborn / pandas /
              ggplot. Includes spread and CRPS columns when available.

        Parameters
        ----------
        directory : str or pathlib.Path
            Output directory. Created if it does not exist.
        burn_in_frac : float, optional (default 0.0)
            Fraction of the initial (spin-up) cycles to discard before
            computing the time-averaged metrics (``mean_rmse_a``,
            ``mean_spread_a``, ``mean_crps_a``, and the spread/error
            ratio). The transient during which all filters are still
            converging is not representative of steady-state skill, so
            for data-assimilation reporting this is typically set to
            0.2–0.3. ``final_rmse_a`` and ``median_rmse_a`` are unaffected
            (they are already steady-state-representative). The
            ``error_curves.csv`` keeps the full series regardless.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping from CSV name (without extension) to the path written.
            Useful for chaining or asserting in tests.
        """
        import pathlib

        pd = self._pd()
        out_dir = pathlib.Path(directory).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, pathlib.Path] = {}

        has_diag = bool(self.rows) and "spread_a" in self.rows[0]

        def _post(series):
            """Discard the first burn_in_frac of a per-cycle series."""
            arr = np.asarray(series)
            if burn_in_frac <= 0.0 or arr.size == 0:
                return arr
            cut = int(np.ceil(burn_in_frac * arr.size))
            cut = min(cut, arr.size - 1)   # always keep at least one point
            return arr[cut:]

        # 1. Per-cell summary (one row per scenario × method × run)
        cell_records = []
        for r in self.rows:
            ea_post = _post(r["error_a"])
            eb_post = _post(r["error_b"])
            rec = {
                "method":        r["method"],
                "scenario_id":   r["scenario_id"],
                "run_id":        r["run_id"],
                "method_seed":   r["method_seed"],
                "elapsed_s":     r["elapsed"],
                "mean_rmse_a":   float(np.mean(ea_post)),
                "median_rmse_a": float(np.median(ea_post)),
                "final_rmse_a":  float(r["error_a"][-1]),
                "mean_rmse_b":   float(np.mean(eb_post)),
                "median_rmse_b": float(np.median(eb_post)),
            }
            if has_diag:
                rec["mean_spread_a"] = float(np.mean(_post(r["spread_a"])))
                rec["mean_spread_b"] = float(np.mean(_post(r["spread_b"])))
                rec["mean_crps_a"]   = float(np.mean(_post(r["crps_a"])))
                rec["mean_crps_b"]   = float(np.mean(_post(r["crps_b"])))
                rec["spread_error_ratio_a"] = (
                    rec["mean_spread_a"] / rec["mean_rmse_a"]
                    if rec["mean_rmse_a"] > 0 else float("nan")
                )
            cell_records.append(rec)
        summary_path = out_dir / "summary.csv"
        pd.DataFrame.from_records(cell_records).to_csv(
            summary_path, index=False, float_format="%.6g",
        )
        written["summary"] = summary_path

        # 2. Per-method aggregated summary
        agg_path = out_dir / "summary_aggregated.csv"
        self.summary_table().to_csv(
            agg_path, index=False, float_format="%.6g",
        )
        written["summary_aggregated"] = agg_path

        # 3. Calibration diagnostics — only if recorded
        if has_diag:
            diag_path = out_dir / "diagnostics_summary.csv"
            self.diagnostics_summary().to_csv(
                diag_path, index=False, float_format="%.6g",
            )
            written["diagnostics_summary"] = diag_path

        # 4. Long-format per-step table
        long_records = []
        for r in self.rows:
            n_steps = len(r["error_a"])
            for k in range(n_steps):
                entry = {
                    "method":      r["method"],
                    "scenario_id": r["scenario_id"],
                    "run_id":      r["run_id"],
                    "step":        k,
                    "time":        float(r["times"][k]),
                    "error_b":     float(r["error_b"][k]),
                    "error_a":     float(r["error_a"][k]),
                }
                if has_diag:
                    entry["spread_b"] = float(r["spread_b"][k])
                    entry["spread_a"] = float(r["spread_a"][k])
                    entry["crps_b"]   = float(r["crps_b"][k])
                    entry["crps_a"]   = float(r["crps_a"][k])
                long_records.append(entry)
        curves_path = out_dir / "error_curves.csv"
        pd.DataFrame.from_records(long_records).to_csv(
            curves_path, index=False, float_format="%.6g",
        )
        written["error_curves"] = curves_path

        return written

    def plot_spread_vs_error(self, ax=None, kind: str = "analysis"):
        """Plot spread vs RMSE over time per method.

        A well-calibrated ensemble has spread ≈ RMSE: the two curves
        overlap. Spread below RMSE → underdispersive (overconfident);
        spread above → overdispersive.
        """
        self._check_diagnostics()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        suffix = "a" if kind == "analysis" else "b"

        # Aggregate over scenarios/runs: median per (method, step)
        from collections import defaultdict
        bucket = defaultdict(lambda: {"t": None, "rmse": [], "spread": []})
        for r in self.rows:
            key = r["method"]
            b = bucket[key]
            if b["t"] is None:
                b["t"] = r["times"]
            b["rmse"].append(r[f"error_{suffix}"])
            b["spread"].append(r[f"spread_{suffix}"])

        for method, b in bucket.items():
            t = b["t"]
            rmse = np.median(np.stack(b["rmse"], axis=0), axis=0)
            spread = np.median(np.stack(b["spread"], axis=0), axis=0)
            line, = ax.plot(t, rmse, label=f"{method} — RMSE")
            ax.plot(t, spread, "--", color=line.get_color(),
                    label=f"{method} — spread")

        ax.set_xlabel("time")
        ax.set_ylabel(f"{kind} (solid = RMSE, dashed = spread)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_rank_histogram(
        self,
        method: str,
        kind: str = "analysis",
        ax=None,
    ):
        """Plot the Talagrand rank histogram for a single method.

        Bin counts are summed across all scenarios, runs, and state
        components. A well-calibrated ensemble produces a flat histogram;
        a U-shape indicates underdispersion (truth often outside ensemble);
        an inverted-U indicates overdispersion. A monotonic slope
        indicates a bias in the ensemble mean.
        """
        self._check_diagnostics()
        import matplotlib.pyplot as plt

        suffix = "a" if kind == "analysis" else "b"
        total = None
        for r in self.rows:
            if r["method"] != method:
                continue
            counts = r[f"rank_counts_{suffix}"]
            total = counts if total is None else total + counts
        if total is None:
            raise ValueError(f"No rows for method '{method}'.")

        N = total.size  # = N_ens + 1 bins
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.bar(np.arange(N), total / total.sum(), width=0.9,
               edgecolor="white", linewidth=0.5)
        ax.axhline(1.0 / N, color="red", linestyle="--", linewidth=1,
                   label=f"flat = 1/{N} (calibrated)")
        ax.set_xlabel("rank of truth among ensemble members (sorted)")
        ax.set_ylabel("frequency")
        ax.set_title(f"Rank histogram — {method} ({kind})")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        return ax


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
class Benchmark:
    """Run a grid of (scenario, method, run) cells and aggregate results.

    Parameters
    ----------
    scenarios : list[Scenario]
        Scenarios to run on. All methods are evaluated on every scenario.
    methods : dict[str, dict]
        Mapping `display_name -> config dict`. Each config must contain a
        `'method'` key naming a registered analysis (e.g. `'enkf'`,
        `'letkf'`), plus any extra kwargs for that analysis (e.g. `r=2`).
    n_runs_per_method : int
        Number of repeated runs per (scenario, method).
    inflation_factor : float
        Inflation factor applied at every step (same for all methods).
    method_seed_base : int
        Base seed for method-internal randomness. The actual seed for each
        cell is `method_seed_base + scenario_id * 10_000 + run_id`.
    parallel : bool
        If True, attempts to run cells in parallel using joblib.
    n_jobs : int
        Number of parallel workers (passed to joblib). -1 uses all cores.
    """

    def __init__(
        self,
        scenarios: List[Scenario],
        methods: Dict[str, Dict[str, Any]],
        n_runs_per_method: int = 1,
        inflation_factor: float = 1.04,
        method_seed_base: int = 1000,
        parallel: bool = False,
        n_jobs: int = -1,
        verbose: bool = True,
        store_diagnostics: bool = False,
        store_states_at=None,
    ):
        if not scenarios:
            raise ValueError("Provide at least one scenario.")
        if not methods:
            raise ValueError("Provide at least one method.")
        self.scenarios = list(scenarios)
        self.methods = dict(methods)
        self.n_runs_per_method = int(n_runs_per_method)
        self.inflation_factor = float(inflation_factor)
        self.method_seed_base = int(method_seed_base)
        self.parallel = bool(parallel)
        self.n_jobs = int(n_jobs)
        self.verbose = bool(verbose)
        self.store_diagnostics = bool(store_diagnostics)
        self.store_states_at = store_states_at

    def _build_cells(self):
        cells = []
        for s_id, scenario in enumerate(self.scenarios):
            for method_name, cfg in self.methods.items():
                for run_id in range(self.n_runs_per_method):
                    method_seed = (
                        self.method_seed_base + s_id * 10_000 + run_id
                    )
                    cells.append(
                        (scenario, s_id, method_name, cfg, run_id, method_seed)
                    )
        return cells

    def run(self) -> BenchmarkResults:
        cells = self._build_cells()
        n = len(cells)
        if self.verbose:
            print(f"[Benchmark] {len(self.scenarios)} scenarios × "
                  f"{len(self.methods)} methods × "
                  f"{self.n_runs_per_method} runs = {n} cells")

        if self.parallel:
            try:
                from joblib import Parallel, delayed
                rows = Parallel(n_jobs=self.n_jobs)(
                    delayed(_run_one_cell)(s, sid, mn, cfg, rid, seed,
                                           self.inflation_factor,
                                           self.store_diagnostics,
                                           self.store_states_at)
                    for (s, sid, mn, cfg, rid, seed) in cells
                )
            except ImportError:
                if self.verbose:
                    print("[Benchmark] joblib not available — falling back to sequential.")
                rows = self._run_sequential(cells)
        else:
            rows = self._run_sequential(cells)

        return BenchmarkResults(
            rows=rows,
            method_names=list(self.methods.keys()),
            n_scenarios=len(self.scenarios),
            n_runs_per_method=self.n_runs_per_method,
        )

    def _run_sequential(self, cells):
        rows = []
        for i, (s, sid, mn, cfg, rid, seed) in enumerate(cells, 1):
            if self.verbose:
                print(f"  [{i}/{len(cells)}] scenario={sid} method={mn} run={rid}")
            rows.append(_run_one_cell(s, sid, mn, cfg, rid, seed,
                                      self.inflation_factor,
                                      self.store_diagnostics,
                                      self.store_states_at))
        return rows
