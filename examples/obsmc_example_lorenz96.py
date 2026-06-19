# -*- coding: utf-8 -*-
"""
Lorenz96 — parameter SWEEP for the observation-space EnKF-MC (Woodbury).

Sweeps the localization radius r in {1,2,3,4,5} and the observation spacing
s in {1,2,3}; each (r, s) cell is run over 10 random scenarios (seeds). For
every cell we persist:

  * the four benchmark CSVs (summary, summary_aggregated, diagnostics,
    error_curves) inside a per-cell folder  results/r{R}_s{S}/,
  * the analysis snapshots (Xa_snapshots) as a compressed .npz per cell,
    for later radar / map plots,
  * a row appended to a MASTER long-format CSV (master_summary.csv) and to a
    master error-curves CSV (master_error_curves.csv), both tagged with the
    columns ``r`` and ``s`` so the whole sweep can be re-plotted from a
    single file with no manual bookkeeping.

The sweep does NOT plot; it only computes and stores. Plotting (radar charts
etc.) is done afterwards from master_summary.csv / the per-cell .npz files.

Run:
    python sweep_lorenz96_obs_woodbury.py
"""

from __future__ import annotations

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal, strided_indices
from pyteda.experiments import Scenario, Benchmark

from datetime import datetime


# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------
RADII            = [1, 2, 3, 4, 5]      # localization radius r
SPACINGS         = [1, 2, 3]            # observation spacing s
ENSEMBLE_SIZES   = [20, 40, 80]         # ensemble size N
N_SCENARIOS      = 10                   # scenarios (seeds) per cell
SCENARIO_SEED0   = 42                   # seeds = 42 .. 42 + N_SCENARIOS - 1

# Fixed model / filter configuration (shared across all cells).
N_STATE          = 40
NOISE_STD        = 0.01
OBS_FREQ         = 0.30
END_TIME         = 50.0
INFLATION        = 1.14
ADAPTIVE_INFLATION = dict(lambda0=1.04, gain=0.15, lo=1.0, hi=1.6)
BURN_IN_FRAC     = 0.30

SPINUP_TRUTH     = 10.0
PERT_XB          = 0.5
SPINUP_XB        = 10.0
PERT_ENSEMBLE    = 0.05
SPINUP_ENSEMBLE  = 10.0

ALPHA            = 0.001                # Ridge penalty for B^{-1}
SNAP_FRACTIONS   = [0.0, 0.25, 0.5, 0.75, 1.0]

# Output root (timestamped so successive sweeps don't collide).
SWEEP_DIR = "sweep_lorenz96_obswoodbury_NrS_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def build_scenarios(spacing, ensemble_size):
    """Build N_SCENARIOS scenarios for a given spacing and ensemble size."""
    model = Lorenz96(n=N_STATE)
    n = model.get_number_of_variables()
    obs_indices, n_obs = strided_indices(model, spacing=spacing)

    x0_synth = model.get_initial_condition()
    x0_ref = model.propagate(x0_synth, np.array([0.0, SPINUP_TRUTH]))

    scenarios = []
    for k in range(N_SCENARIOS):
        s = SCENARIO_SEED0 + k
        scen = Scenario.generate(
            model=model,
            operator_factory=lambda rng, idx=obs_indices: LinearSelection(
                m=idx.size, n_state=n, indices=idx,
            ),
            noise=IsotropicDiagonal(std=NOISE_STD, dim=n_obs),
            ensemble_size=ensemble_size,
            x0_ref=x0_ref,
            pert_xb=PERT_XB, spinup_xb=SPINUP_XB,
            pert_ensemble=PERT_ENSEMBLE,
            spinup_ensemble=SPINUP_ENSEMBLE,
            obs_freq=OBS_FREQ, end_time=END_TIME,
            seed=s,
        )
        scenarios.append(scen)
    return model, scenarios, n_obs


def run_cell(radius, spacing, ensemble_size, cell_dir):
    """Run one (N, r, s) cell over N_SCENARIOS and persist everything."""
    os.makedirs(cell_dir, exist_ok=True)

    model, scenarios, n_obs = build_scenarios(spacing, ensemble_size)

    methods = {
        "EnKF-OBS-Woodbury": dict(method="enkf-obs-woodbury-cholesky",
                                  r=radius, alpha=ALPHA),
        "LEnKF":             dict(method="lenkf", r=radius),
        "LETKF":             dict(method="letkf", r=radius),
    }

    results = Benchmark(
        scenarios=scenarios,
        methods=methods,
        n_runs_per_method=1,
        inflation_factor=INFLATION,
        method_seed_base=1000,
        parallel=False,
        verbose=False,
        store_diagnostics=True,
        store_states_at=SNAP_FRACTIONS,
        adaptive_inflation_cfg=ADAPTIVE_INFLATION,
    ).run()

    # 1. Per-cell CSVs.
    results.export_csv(cell_dir, burn_in_frac=BURN_IN_FRAC)

    # 2. Per-cell snapshots (Xa_snapshots) for radar / map plots.
    #    Saved compressed, keyed by method + scenario + run.
    snap = {}
    for row in results.rows:
        if "Xa_snapshots" not in row:
            continue
        key = f"{row['method']}__scen{row['scenario_id']}__run{row['run_id']}"
        snap[key] = np.asarray(row["Xa_snapshots"])
    if snap:
        snap["__snapshot_fractions__"] = np.asarray(SNAP_FRACTIONS)
        np.savez_compressed(os.path.join(cell_dir, "snapshots.npz"), **snap)

    # 3. Rows for the master tables, tagged with r and s.
    summary = results.summary_table()
    summary_rows = summary.copy()
    summary_rows.insert(0, "N", ensemble_size)
    summary_rows.insert(1, "r", radius)
    summary_rows.insert(2, "s", spacing)
    summary_rows.insert(3, "n_obs", n_obs)

    # Long-format per-cell summary (one row per scenario × method × run).
    pd_local = pd.read_csv(os.path.join(cell_dir, "summary.csv"))
    pd_local.insert(0, "N", ensemble_size)
    pd_local.insert(1, "r", radius)
    pd_local.insert(2, "s", spacing)
    pd_local.insert(3, "n_obs", n_obs)

    # Error curves, tagged.
    curves = pd.read_csv(os.path.join(cell_dir, "error_curves.csv"))
    curves.insert(0, "N", ensemble_size)
    curves.insert(1, "r", radius)
    curves.insert(2, "s", spacing)

    return summary_rows, pd_local, curves


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    results_root = os.path.join(SWEEP_DIR, "results")
    os.makedirs(results_root, exist_ok=True)

    # Persist the sweep configuration for reproducibility.
    config = dict(
        RADII=RADII, SPACINGS=SPACINGS, N_SCENARIOS=N_SCENARIOS,
        SCENARIO_SEED0=SCENARIO_SEED0, N_STATE=N_STATE, NOISE_STD=NOISE_STD,
        ENSEMBLE_SIZES=ENSEMBLE_SIZES, OBS_FREQ=OBS_FREQ, END_TIME=END_TIME,
        INFLATION=INFLATION, ADAPTIVE_INFLATION=ADAPTIVE_INFLATION,
        BURN_IN_FRAC=BURN_IN_FRAC, ALPHA=ALPHA,
        SNAP_FRACTIONS=SNAP_FRACTIONS,
    )
    with open(os.path.join(SWEEP_DIR, "sweep_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    n_cells = len(ENSEMBLE_SIZES) * len(RADII) * len(SPACINGS)
    print("=" * 72)
    print(f" SWEEP: {len(ENSEMBLE_SIZES)} ensemble sizes × {len(RADII)} radii "
          f"× {len(SPACINGS)} spacings = {n_cells} cells, "
          f"{N_SCENARIOS} scenarios each")
    print(f" output: {SWEEP_DIR}")
    print("=" * 72)

    master_summary = []     # aggregated (one row per method per cell)
    master_percell = []     # per scenario × method × run
    master_curves  = []     # long-format error curves

    cell_i = 0
    for N in ENSEMBLE_SIZES:
        for s in SPACINGS:
            for r in RADII:
                cell_i += 1
                cell_name = f"N{N}_r{r}_s{s}"
                cell_dir = os.path.join(results_root, cell_name)
                print(f"  [{cell_i}/{n_cells}] cell {cell_name} "
                      f"(N={N}, radius={r}, spacing={s}, "
                      f"{N_SCENARIOS} scenarios) ...", flush=True)
                try:
                    summ, percell, curves = run_cell(r, s, N, cell_dir)
                    master_summary.append(summ)
                    master_percell.append(percell)
                    master_curves.append(curves)
                    best = summ.sort_values("mean_rmse_a").iloc[0]
                    print(f"       done — best: {best['method']} "
                          f"(mean_rmse_a={best['mean_rmse_a']:.4f})",
                          flush=True)
                except Exception as e:
                    print(f"       FAILED: {type(e).__name__}: {e}",
                          flush=True)

    # Write master tables.
    if master_summary:
        pd.concat(master_summary, ignore_index=True).to_csv(
            os.path.join(SWEEP_DIR, "master_summary.csv"), index=False)
    if master_percell:
        pd.concat(master_percell, ignore_index=True).to_csv(
            os.path.join(SWEEP_DIR, "master_percell.csv"), index=False)
    if master_curves:
        pd.concat(master_curves, ignore_index=True).to_csv(
            os.path.join(SWEEP_DIR, "master_error_curves.csv"), index=False)

    print()
    print("=" * 72)
    print(" SWEEP complete.")
    print(f"   master_summary.csv      (aggregated per method per cell)")
    print(f"   master_percell.csv      (per scenario × method × run)")
    print(f"   master_error_curves.csv (long-format curves)")
    print(f"   results/r*_s*/          (per-cell CSVs + snapshots.npz)")
    print(f"   all under: {os.path.abspath(SWEEP_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    main()