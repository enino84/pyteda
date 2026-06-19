# -*- coding: utf-8 -*-
"""
SWE on the sphere — parameter SWEEP for the observation-space EnKF-MC
(Woodbury), mirroring the Lorenz-96 sweep logic.

Sweeps the ensemble size N in {20, 40} and the localization radius r in
{1, 3, 5}; the observation coverage is FIXED at 70% (continental mask).
Each (N, r) cell is run over N_SCENARIOS random scenarios (seeds). For
every cell we persist:

  * the benchmark CSVs (summary, summary_aggregated, diagnostics,
    error_curves) inside a per-cell folder  results/N{N}_r{R}/,
  * the analysis snapshots (Xa_snapshots) as a compressed .npz per cell,
  * rows appended to MASTER CSVs (master_summary.csv, master_percell.csv,
    master_error_curves.csv), each tagged with columns N and r.

Spin-up artefacts (x0_ref, xb, initial ensembles for BOTH sizes, truth
trajectory) are cached in ./swe_cache/ and reused across cells. Because the
two ensemble sizes need different initial ensembles, the ensemble cache is
keyed by size: initial_ensemble_N{N}.nc.

The sweep does NOT plot; it only computes and stores. Plotting is done
afterwards from the master CSVs / per-cell .npz files.

Run:
    python sweep_swe_obs_woodbury.py
"""

from __future__ import annotations

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pyteda.models import SWEModel
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario, Benchmark
from pyteda.io import (
    save_state_vector, load_state_vector,
    save_initial_ensemble, load_initial_ensemble,
)

from datetime import datetime


# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------
ENSEMBLE_SIZES   = [20, 40]            # ensemble size N
RADII            = [1, 3, 5]           # localization radius r
OBS_FRACTION     = 0.70                # FIXED observation coverage (70%)
N_SCENARIOS      = 10                  # scenarios (seeds) per (N, r) cell
SCENARIO_SEED0   = 42

# ----------------------------------------------------------------------
# Fixed model configuration (shared across all cells).
# ----------------------------------------------------------------------
LMAX             = 21
DT               = 120.0
STATE_VARS       = ["u", "v", "h"]

DAY              = 86400.0
HOUR             = 3600.0
SPINUP_TRUTH     = 18 * DAY
SPINUP_XB        = 18 * DAY
SPINUP_ENSEMBLE  = 18 * DAY
OBS_FREQ         = 6 * HOUR
END_TIME         = 7 * DAY

PERT_XB_U, PERT_XB_V, PERT_XB_H    = 2.0, 2.0, 30.0
PERT_ENS_U, PERT_ENS_V, PERT_ENS_H = 0.5, 0.5, 8.0
PERT_TRUNC       = 12

INFLATION        = 1.04
ADAPTIVE_INFLATION = dict(lambda0=1.04, gain=0.15, lo=1.0, hi=1.4)
BURN_IN_FRAC     = 0.30

NOISE_STD        = 5.0
ALPHA            = 0.1                  # Ridge penalty for B^{-1}
SNAP_FRACTIONS   = [0.0, 0.25, 0.5, 0.75, 1.0]

# Largest ensemble we will need (build this many members; smaller N uses a
# prefix of the same pool, so the cells share members and are comparable).
MAX_ENS          = max(ENSEMBLE_SIZES)

CACHE_DIR = "swe_cache"
SWEEP_DIR = "sweep_swe_obswoodbury_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ======================================================================
# Continental observation mask (fixed 70% coverage)
# ======================================================================
def _continental_mask(grid, fraction=0.70, seed=0):
    Nlat, Nlon = grid.Nlat, grid.Nlon
    lats = np.linspace(np.pi / 2, -np.pi / 2, Nlat)
    lons = np.linspace(0, 2 * np.pi, Nlon, endpoint=False)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    blobs = [
        (np.deg2rad(40),  np.deg2rad(60),  np.deg2rad(35), np.deg2rad(55), 1.2),
        (np.deg2rad(20),  np.deg2rad(280), np.deg2rad(40), np.deg2rad(35), 1.1),
        (np.deg2rad(-30), np.deg2rad(135), np.deg2rad(15), np.deg2rad(20), 0.7),
        (np.deg2rad(-80), np.deg2rad(0),   np.deg2rad(10), np.deg2rad(180), 0.9),
    ]
    field = np.zeros((Nlat, Nlon))
    for lat0, lon0, slat, slon, w in blobs:
        dlon = np.minimum(np.abs(LON - lon0), 2 * np.pi - np.abs(LON - lon0))
        field += w * np.exp(-0.5 * ((LAT - lat0) / slat) ** 2
                            - 0.5 * (dlon / slon) ** 2)
    rng = np.random.default_rng(seed)
    field += 0.10 * rng.standard_normal(field.shape)
    thresh = np.quantile(field.ravel(), 1.0 - fraction)
    return field >= thresh


def _obs_indices_from_mask(mask, model):
    flat_mask = mask.ravel()
    keep_per_field = np.where(flat_mask)[0]
    indices = []
    for var in model.state_vars:
        offset = model.var_blocks[var].start
        indices.append(keep_per_field + offset)
    return np.concatenate(indices), keep_per_field.size


# ======================================================================
# Smooth (large-scale) perturbations
# ======================================================================
def _smooth_field(grid, magnitude, n_modes, rng):
    Nlat, Nlon = grid.Nlat, grid.Nlon
    lats = np.linspace(np.pi / 2, -np.pi / 2, Nlat)
    lons = np.linspace(0, 2 * np.pi, Nlon, endpoint=False)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    field = np.zeros((Nlat, Nlon))
    for _ in range(n_modes):
        kx = rng.integers(1, PERT_TRUNC + 1)
        ky = rng.integers(1, PERT_TRUNC + 1)
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        amp = rng.standard_normal()
        field += amp * np.cos(kx * LON + px) * np.cos(ky * LAT + py)
    field *= np.cos(LAT) ** 2
    field /= max(field.std(), 1e-12)
    return magnitude * field


def _build_perturbation(model, mu, mv, mh, rng, n_modes=20):
    pert = np.zeros(model.dim)
    mags = {"u": mu, "v": mv, "h": mh}
    for var in model.state_vars:
        sl = model.var_blocks[var]
        pert[sl] = _smooth_field(model.grid, mags[var], n_modes, rng).ravel()
    return pert


# ======================================================================
# Build / cache spin-up artefacts (shared across cells)
#   - x0_ref, xb, truth: identical for all N
#   - initial ensemble: built once at MAX_ENS, smaller N uses a prefix
# ======================================================================
def build_or_load_artifacts(model):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p_x0 = os.path.join(CACHE_DIR, "x0_ref.nc")
    p_xb = os.path.join(CACHE_DIR, "xb.nc")
    p_ens = os.path.join(CACHE_DIR, f"initial_ensemble_N{MAX_ENS}.nc")

    if os.path.exists(p_x0):
        print(f"  x0_ref   cached -> {p_x0}")
        x0_ref = load_state_vector(p_x0)
    else:
        print("  x0_ref   building ...")
        x_synth = model.get_initial_condition()
        x0_ref = model.propagate(x_synth, np.array([0.0, SPINUP_TRUTH]))
        save_state_vector(x0_ref, p_x0, name="x0_ref")

    if os.path.exists(p_xb):
        print(f"  xb       cached -> {p_xb}")
        xb = load_state_vector(p_xb)
    else:
        print("  xb       building ...")
        rng = np.random.default_rng(99)
        kick = _build_perturbation(model, PERT_XB_U, PERT_XB_V, PERT_XB_H, rng)
        xb = model.propagate(x0_ref + kick, np.array([0.0, SPINUP_XB]))
        save_state_vector(xb, p_xb, name="xb")

    if os.path.exists(p_ens):
        print(f"  ensemble cached -> {p_ens}  (pool of {MAX_ENS} members)")
        X0_pool = load_initial_ensemble(p_ens)
    else:
        print(f"  ensemble building pool of {MAX_ENS} members ...")
        t0 = time.perf_counter()
        rng = np.random.default_rng(101)
        X0_pool = np.empty((xb.size, MAX_ENS))
        for k in range(MAX_ENS):
            kick = _build_perturbation(model, PERT_ENS_U, PERT_ENS_V,
                                       PERT_ENS_H, rng)
            X0_pool[:, k] = model.propagate(
                xb + kick, np.array([0.0, SPINUP_ENSEMBLE]))
            print(f"    member {k+1}/{MAX_ENS} "
                  f"({time.perf_counter()-t0:.0f}s)")
        save_initial_ensemble(X0_pool, p_ens)
    return x0_ref, xb, X0_pool


def build_or_load_truth(model):
    p = os.path.join(CACHE_DIR, "truth.nc")
    sync_time = float(SPINUP_XB + SPINUP_ENSEMBLE)
    n_assim = int(round(END_TIME / OBS_FREQ))
    if os.path.exists(p):
        print(f"  truth    cached -> {p}")
        from pyteda.io import load_truth_trajectory
        return load_truth_trajectory(p)
    print(f"  truth    building (sync + {n_assim} steps) ...")
    x0_ref = load_state_vector(os.path.join(CACHE_DIR, "x0_ref.nc"))
    n_sync = int(round(sync_time / OBS_FREQ))
    x = x0_ref.copy()
    for _ in range(n_sync):
        x = model.propagate(x, np.array([0.0, OBS_FREQ]))
    truth = [x.copy()]
    for _ in range(n_assim):
        x = model.propagate(x, np.array([0.0, OBS_FREQ]))
        truth.append(x.copy())
    times = np.arange(n_assim + 1) * OBS_FREQ
    from pyteda.io import save_truth_trajectory
    save_truth_trajectory(truth, times, p)
    return truth, times


# ======================================================================
# One (N, r) cell
# ======================================================================
def run_cell(radius, ensemble_size, cell_dir,
             model, x0_ref, xb, X0_pool, truth, times, obs_indices):
    os.makedirs(cell_dir, exist_ok=True)
    n_obs = obs_indices.size

    # Use the first `ensemble_size` members of the shared pool.
    X0 = X0_pool[:, :ensemble_size].copy()

    scenarios = []
    for k in range(N_SCENARIOS):
        s = SCENARIO_SEED0 + k
        scen = Scenario.generate(
            model=model,
            operator_factory=lambda rng, idx=obs_indices: LinearSelection(
                m=n_obs, n_state=model.dim, indices=idx),
            noise=IsotropicDiagonal(std=NOISE_STD, dim=n_obs),
            x0_ref=x0_ref, xb=xb, initial_ensemble=X0,
            truth_trajectory=truth,
            spinup_xb=SPINUP_XB, spinup_ensemble=SPINUP_ENSEMBLE,
            obs_freq=OBS_FREQ, end_time=END_TIME, seed=s)
        scenarios.append(scen)

    methods = {
        "EnKF-OBS-Woodbury": dict(method="enkf-obs-woodbury-cholesky",
                                  r=radius, alpha=ALPHA),
        "LEnKF":             dict(method="lenkf", r=radius),
        "LETKF":             dict(method="letkf", r=radius),
    }

    results = Benchmark(
        scenarios=scenarios, methods=methods, n_runs_per_method=1,
        inflation_factor=INFLATION, method_seed_base=1000, parallel=False,
        verbose=False, store_diagnostics=True, store_states_at=SNAP_FRACTIONS,
        adaptive_inflation_cfg=ADAPTIVE_INFLATION,
    ).run()

    # 1. per-cell CSVs
    results.export_csv(cell_dir, burn_in_frac=BURN_IN_FRAC)

    # 2. per-cell snapshots
    snap = {}
    for row in results.rows:
        if "Xa_snapshots" not in row:
            continue
        key = f"{row['method']}__scen{row['scenario_id']}__run{row['run_id']}"
        snap[key] = np.asarray(row["Xa_snapshots"])
    if snap:
        snap["__snapshot_fractions__"] = np.asarray(SNAP_FRACTIONS)
        np.savez_compressed(os.path.join(cell_dir, "snapshots.npz"), **snap)

    # 3. master rows tagged with N and r
    summ = results.summary_table()
    summ.insert(0, "N", ensemble_size)
    summ.insert(1, "r", radius)
    summ.insert(2, "obs_frac", OBS_FRACTION)
    summ.insert(3, "n_obs", n_obs)

    percell = pd.read_csv(os.path.join(cell_dir, "summary.csv"))
    percell.insert(0, "N", ensemble_size)
    percell.insert(1, "r", radius)
    percell.insert(2, "obs_frac", OBS_FRACTION)

    curves = pd.read_csv(os.path.join(cell_dir, "error_curves.csv"))
    curves.insert(0, "N", ensemble_size)
    curves.insert(1, "r", radius)

    return summ, percell, curves


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    results_root = os.path.join(SWEEP_DIR, "results")
    os.makedirs(results_root, exist_ok=True)

    config = dict(
        ENSEMBLE_SIZES=ENSEMBLE_SIZES, RADII=RADII, OBS_FRACTION=OBS_FRACTION,
        N_SCENARIOS=N_SCENARIOS, SCENARIO_SEED0=SCENARIO_SEED0, LMAX=LMAX,
        DT=DT, STATE_VARS=STATE_VARS, OBS_FREQ=OBS_FREQ, END_TIME=END_TIME,
        NOISE_STD=NOISE_STD, INFLATION=INFLATION,
        ADAPTIVE_INFLATION=ADAPTIVE_INFLATION, BURN_IN_FRAC=BURN_IN_FRAC,
        ALPHA=ALPHA, SNAP_FRACTIONS=SNAP_FRACTIONS,
    )
    with open(os.path.join(SWEEP_DIR, "sweep_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 72)
    print(" SWE setup")
    print("=" * 72)
    model = SWEModel(LMAX=LMAX, dt=DT, state_vars=STATE_VARS)
    print(f"  grid {model.grid.Nlat}×{model.grid.Nlon}  state dim={model.dim}")

    mask = _continental_mask(model.grid, fraction=OBS_FRACTION, seed=0)
    obs_indices, n_keep = _obs_indices_from_mask(mask, model)
    print(f"  obs mask: {n_keep} pts/var  total obs={obs_indices.size} "
          f"({OBS_FRACTION*100:.0f}% coverage)")

    print()
    print("=" * 72)
    print(f" Spin-up (cached in ./{CACHE_DIR}/)")
    print("=" * 72)
    x0_ref, xb, X0_pool = build_or_load_artifacts(model)
    truth, times = build_or_load_truth(model)

    n_cells = len(ENSEMBLE_SIZES) * len(RADII)
    print()
    print("=" * 72)
    print(f" SWEEP: {len(ENSEMBLE_SIZES)} ensemble sizes × {len(RADII)} radii "
          f"= {n_cells} cells, {N_SCENARIOS} scenarios each  (obs fixed 70%)")
    print(f" output: {SWEEP_DIR}")
    print("=" * 72)

    master_summary, master_percell, master_curves = [], [], []
    cell_i = 0
    for N in ENSEMBLE_SIZES:
        for r in RADII:
            cell_i += 1
            cell_name = f"N{N}_r{r}"
            cell_dir = os.path.join(results_root, cell_name)
            t0 = time.perf_counter()
            print(f"  [{cell_i}/{n_cells}] cell {cell_name} "
                  f"(N={N}, radius={r}, {N_SCENARIOS} scenarios) ...",
                  flush=True)
            try:
                summ, percell, curves = run_cell(
                    r, N, cell_dir, model, x0_ref, xb, X0_pool,
                    truth, times, obs_indices)
                master_summary.append(summ)
                master_percell.append(percell)
                master_curves.append(curves)
                best = summ.sort_values("mean_rmse_a").iloc[0]
                print(f"       done in {(time.perf_counter()-t0)/60:.1f} min "
                      f"— best: {best['method']} "
                      f"(mean_rmse_a={best['mean_rmse_a']:.4f})", flush=True)
            except Exception as e:
                print(f"       FAILED: {type(e).__name__}: {e}", flush=True)

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
    print(f"   master_summary.csv / master_percell.csv / master_error_curves.csv")
    print(f"   results/N*_r*/ (per-cell CSVs + snapshots.npz)")
    print(f"   all under: {os.path.abspath(SWEEP_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    main()