# -*- coding: utf-8 -*-
"""
Shallow Water Equations on the sphere — full experiment.

Configuration mirrors example_lorenz96.py but adapted to SWE physics:

  Phase 1 (truth):      synthetic IC -> propagate 12 days -> x0_ref
                        x0_ref is computed ONCE, cached on disk, and shared
                        across the 5 scenarios.
  Phase 2 (xb):         kick + propagate 12 days -> xb           (cached)
  Phase 3 (ensemble):   per-member kick + propagate 12 days -> X_b (cached)
  Assimilation:         every 6 hours during 12 days  =>  48 cycles
  Observation noise:    isotropic Gaussian (per-variable scale)
  Observations:         a continental mask is generated for the sphere; only
                        components inside the mask are observed (70% of all
                        state points). All three fields (u, v, h) are
                        observed at the same locations.
  Methods:              5 with localisation (the only ones that survive at
                        LMAX=32 with 20 members: LETKF, LEnKF, EnKF-MC,
                        EnKF-Shrinkage, EnKF-BLoc).
  Snapshots:            ensemble stored at 0%, 25%, 50%, 75%, 100%.

The cached spin-up artefacts live in ./swe_cache/. Delete that folder if
you want to rebuild from scratch.

Plots are written under ./swe_demo_figs/ with the same layout as Lorenz96:
  - 01_error_curves_all.png
  - 02_analysis_vs_background.png
  - 03_spread_vs_error.png
  - 06_boxplot_log.png  /  06_boxplot_linear.png
  - per_method/curves_<m>.png and rank_histogram_<m>.png
  - maps/state_<method>_t<frac>.png  -> spatial maps (one per (method, snap))
  - All four CSVs from results.export_csv

Run:

    python example_swe.py
"""

from __future__ import annotations

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from pyteda.models import SWEModel
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario, Benchmark
from pyteda.io import (
    save_state_vector, load_state_vector,
    save_initial_ensemble, load_initial_ensemble,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
LMAX            = 21
DT              = 120.0      # seconds per RK4 substep
STATE_VARS      = ["u", "v", "h"]

# Time scales
DAY             = 86400.0
HOUR            = 3600.0
# These three spinups are only used the FIRST time you run the script —
# afterwards everything is cached in ./swe_cache/. So they're generous:
# 18 days >> Lyapunov time (~2 days), so the system is fully chaotically
# saturated by then. Cost is paid once, then forever cached.
SPINUP_TRUTH    = 18 * DAY
SPINUP_XB       = 18 * DAY    # truth -> kick -> propagate -> xb
SPINUP_ENSEMBLE = 18 * DAY    # xb -> kick -> propagate -> X_b (per member)
OBS_FREQ        = 6 * HOUR    # assimilate every 6 hours
END_TIME        = 7 * DAY     # 28 assimilation cycles

# Ensemble
ENSEMBLE_SIZE   = 20

# Per-variable perturbation magnitudes (in physical units).
# Gaussian-white noise on a high-resolution grid has energy at all
# wavelengths, including grid-scale modes that destabilise the SWE
# integrator. We instead build perturbations from low-degree spherical
# harmonics scaled to a realistic background-error magnitude.
PERT_XB_U       = 2.0          # m/s   ~5% of U0=38
PERT_XB_V       = 2.0          # m/s
PERT_XB_H       = 30.0         # m     ~1% of H0=2800
PERT_ENS_U      = 0.5          # m/s
PERT_ENS_V      = 0.5          # m/s
PERT_ENS_H      = 8.0          # m

# Maximum spherical-harmonic degree used to construct perturbations.
# Truncating at this degree filters out grid-scale noise, which keeps
# the integrator stable. degree=12 gives features of size ~(2*pi*R)/12
# ~3300 km — synoptic-scale, like real background errors.
PERT_TRUNC      = 12

INFLATION       = 1.04

# Observation noise per variable.  IsotropicDiagonal uses one std for the
# whole vector, so we scale each variable to a comparable relative noise
# by setting std relative to the typical magnitude.  For mixed (u,v,h) we
# pick a single value that's reasonable for h (the dominant signal).
NOISE_STD       = 5.0          # ~5 m of geopotential-equivalent noise

# Observation coverage
OBS_FRACTION    = 0.70         # 70% of components fall on "continents"

# Experiment grid
N_SCENARIOS     = 5
SCENARIO_SEEDS  = list(range(42, 42 + N_SCENARIOS))

SNAP_FRACTIONS  = [0.0, 0.25, 0.5, 0.75, 1.0]

# Methods — full-rank shrinkage estimators don't need localisation;
# localised filters are also included for comparison.
METHODS = {
    # Localised filters (R-localisation and B-localisation)
    "LETKF(r=4)":          dict(method="letkf", r={"u": 4, "v": 4, "h": 4}),
    "LEnKF(r=4)":          dict(method="lenkf", r={"u": 4, "v": 4, "h": 4}),

    # Modified-Cholesky family (sparse precision via local Ridge regressions)
    "EnKF-MC(r=2)":        dict(method="enkf-modified-cholesky",
                                  r={"u": 2, "v": 2, "h": 2}),

    # Shrinkage family — three different formulations
    "EnKF-LW":             dict(method="enkf-lw"),                  # Ledoit-Wolf
    "EnKF-RBLW":           dict(method="enkf-rblw"),                # Rao-Blackwell LW
}

CACHE_DIR = "swe_cache"
FIG_DIR   = "swe_demo_figs"

N_TOP_METHODS = len(METHODS)   # all of them get the per-method visuals


# ======================================================================
# Continental observation mask
# ======================================================================
def _continental_mask(grid, fraction=0.70, seed=0):
    """Return a boolean array (Nlat, Nlon) approximating continental coverage.

    Real continents cluster into a few large landmasses with strong
    latitudinal asymmetry (more land in the Northern hemisphere). We
    approximate this with three Gaussian "blobs" centred where the major
    landmasses sit, then threshold at a level that gives the requested
    coverage fraction.
    """
    Nlat = grid.Nlat
    Nlon = grid.Nlon
    # latitudes north -> south, from +pi/2 to -pi/2
    lats = np.linspace(np.pi / 2, -np.pi / 2, Nlat)
    lons = np.linspace(0, 2 * np.pi, Nlon, endpoint=False)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    # Three blobs roughly mimicking Eurasia+Africa, the Americas, Australia/Antarctica
    blobs = [
        # (lat0, lon0, sigma_lat, sigma_lon, weight)
        (np.deg2rad(40),  np.deg2rad(60),  np.deg2rad(35), np.deg2rad(55), 1.2),  # Eurasia/Africa
        (np.deg2rad(20),  np.deg2rad(280), np.deg2rad(40), np.deg2rad(35), 1.1),  # Americas
        (np.deg2rad(-30), np.deg2rad(135), np.deg2rad(15), np.deg2rad(20), 0.7),  # Australia
        (np.deg2rad(-80), np.deg2rad(0),   np.deg2rad(10), np.deg2rad(180), 0.9), # Antarctica
    ]
    field = np.zeros((Nlat, Nlon))
    for lat0, lon0, slat, slon, w in blobs:
        # use periodic distance in longitude
        dlon = np.minimum(np.abs(LON - lon0), 2 * np.pi - np.abs(LON - lon0))
        field += w * np.exp(
            -0.5 * ((LAT - lat0) / slat) ** 2
            - 0.5 * (dlon / slon) ** 2
        )
    # add a tiny amount of noise so the mask edge isn't perfectly smooth
    rng = np.random.default_rng(seed)
    field += 0.10 * rng.standard_normal(field.shape)

    # Threshold at the percentile that gives the desired fraction
    thresh = np.quantile(field.ravel(), 1.0 - fraction)
    return field >= thresh


def _obs_indices_from_mask(mask, model):
    """Return an int ndarray of state-vector indices that lie under the
    mask, for each variable in model.state_vars (concatenated)."""
    field_size = model.field_size
    flat_mask = mask.ravel()                      # (Nlat * Nlon,)
    keep_per_field = np.where(flat_mask)[0]
    n_keep = keep_per_field.size

    indices = []
    for var in model.state_vars:
        offset = model.var_blocks[var].start      # block start in state vec
        indices.append(keep_per_field + offset)
    return np.concatenate(indices), n_keep


# ======================================================================
# Smooth (large-scale) perturbations for SWE
# ======================================================================
def _smooth_field(grid, magnitude, n_modes, rng):
    """Generate a smooth random field on the (Nlat, Nlon) grid.

    The field is built as a sum of ``n_modes`` random sinusoids in
    longitude and latitude with wavenumbers up to ``PERT_TRUNC``. This
    yields synoptic-scale spatial patterns (~3000 km features) that are
    physically realistic as background errors and, crucially, contain no
    grid-scale energy — so the SWE integrator stays stable.
    """
    Nlat, Nlon = grid.Nlat, grid.Nlon
    lats = np.linspace(np.pi / 2, -np.pi / 2, Nlat)
    lons = np.linspace(0, 2 * np.pi, Nlon, endpoint=False)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    field = np.zeros((Nlat, Nlon))
    for _ in range(n_modes):
        kx = rng.integers(1, PERT_TRUNC + 1)             # zonal wavenumber
        ky = rng.integers(1, PERT_TRUNC + 1)             # meridional wavenumber
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        amp = rng.standard_normal()
        field += amp * np.cos(kx * LON + phase_x) * np.cos(ky * LAT + phase_y)

    # taper to zero at the poles to avoid pole-singularity issues
    field *= np.cos(LAT) ** 2

    # normalise to unit standard deviation, then scale to requested magnitude
    field /= max(field.std(), 1e-12)
    return magnitude * field


def _build_perturbation(model, mag_u, mag_v, mag_h, rng, n_modes=20):
    """Build a state-vector perturbation with per-variable magnitudes."""
    pert = np.zeros(model.dim)
    mags = {"u": mag_u, "v": mag_v, "h": mag_h}
    for var in model.state_vars:
        sl = model.var_blocks[var]
        f = _smooth_field(model.grid, mags[var], n_modes, rng)
        pert[sl] = f.ravel()
    return pert


# ======================================================================
# Build / cache the spinup artefacts
# ======================================================================
def build_or_load_artifacts(model):
    """Build x0_ref, xb and initial_ensemble (cached on disk)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {
        "x0_ref":  os.path.join(CACHE_DIR, "x0_ref.nc"),
        "xb":      os.path.join(CACHE_DIR, "xb.nc"),
        "ens":     os.path.join(CACHE_DIR, "initial_ensemble.nc"),
    }

    # ---- x0_ref ----
    if os.path.exists(cache["x0_ref"]):
        print(f"  x0_ref     cached  -> {cache['x0_ref']}")
        x0_ref = load_state_vector(cache["x0_ref"])
    else:
        print("  x0_ref     building (12 days of spinup) ...")
        t0 = time.perf_counter()
        x_synth = model.get_initial_condition()
        x0_ref = model.propagate(x_synth, np.array([0.0, SPINUP_TRUTH]))
        save_state_vector(x0_ref, cache["x0_ref"], name="x0_ref")
        print(f"             done in {time.perf_counter() - t0:.1f}s")

    # ---- xb ----
    if os.path.exists(cache["xb"]):
        print(f"  xb         cached  -> {cache['xb']}")
        xb = load_state_vector(cache["xb"])
    else:
        print("  xb         building (smooth kick + 12 days of spinup) ...")
        t0 = time.perf_counter()
        rng = np.random.default_rng(99)
        kick = _build_perturbation(
            model, PERT_XB_U, PERT_XB_V, PERT_XB_H, rng,
        )
        xb = model.propagate(x0_ref + kick, np.array([0.0, SPINUP_XB]))
        save_state_vector(xb, cache["xb"], name="xb")
        print(f"             done in {time.perf_counter() - t0:.1f}s")

    # ---- initial_ensemble ----
    if os.path.exists(cache["ens"]):
        print(f"  ensemble   cached  -> {cache['ens']}")
        X0 = load_initial_ensemble(cache["ens"])
    else:
        print(f"  ensemble   building ({ENSEMBLE_SIZE} members "
              f"× {SPINUP_ENSEMBLE/DAY:.0f} day(s) of spinup) ...")
        t0 = time.perf_counter()
        rng = np.random.default_rng(101)
        X0 = np.empty((xb.size, ENSEMBLE_SIZE))
        for k in range(ENSEMBLE_SIZE):
            kick = _build_perturbation(
                model, PERT_ENS_U, PERT_ENS_V, PERT_ENS_H, rng,
            )
            X0[:, k] = model.propagate(
                xb + kick, np.array([0.0, SPINUP_ENSEMBLE]),
            )
            print(f"      member {k+1}/{ENSEMBLE_SIZE}  "
                  f"({time.perf_counter() - t0:.0f}s elapsed)")
        save_initial_ensemble(X0, cache["ens"])
        print(f"             done in {time.perf_counter() - t0:.1f}s")

    return x0_ref, xb, X0


def build_or_load_truth_trajectory(model):
    """Build the truth trajectory step-by-step, cached on disk.

    The integrator is fragile when run for many simulated days without
    restoration — to avoid blow-up we propagate in 6-hour chunks and let
    each chunk start from the previous output, with the model's spectral
    filter cleaning it up between chunks. This is exactly how Scenario
    would do it internally, but doing it here makes the trajectory a
    cacheable artefact and avoids re-doing the work for each scenario.
    """
    cache_path = os.path.join(CACHE_DIR, "truth.nc")
    sync_time = float(SPINUP_XB + SPINUP_ENSEMBLE)
    n_assim = int(round(END_TIME / OBS_FREQ))
    n_total = n_assim + 1   # truth[0] is at sync_time, truth[n_assim] is at sync_time + END_TIME

    if os.path.exists(cache_path):
        print(f"  truth      cached  -> {cache_path}")
        from pyteda.io import load_truth_trajectory
        truth_list, times_list = load_truth_trajectory(cache_path)
        return truth_list, times_list

    print(f"  truth      building (sync + {n_assim} steps × "
          f"{OBS_FREQ/HOUR:.0f}h) ...")
    t0 = time.perf_counter()

    # Synchronise: x0_ref -> propagate by sync_time -> truth[0]
    # but in chunks of OBS_FREQ to keep the integrator happy
    x0_ref = load_state_vector(os.path.join(CACHE_DIR, "x0_ref.nc"))
    n_sync_chunks = int(round(sync_time / OBS_FREQ))
    print(f"             sync phase: {n_sync_chunks} chunks of "
          f"{OBS_FREQ/HOUR:.0f}h each")
    x = x0_ref.copy()
    for k in range(n_sync_chunks):
        x = model.propagate(x, np.array([0.0, OBS_FREQ]))
        if (k + 1) % 10 == 0:
            print(f"               sync {k+1}/{n_sync_chunks} "
                  f"({time.perf_counter() - t0:.0f}s)")

    # Now x corresponds to truth[0] — store the trajectory from here
    truth_list = [x.copy()]
    for k in range(n_assim):
        x = model.propagate(x, np.array([0.0, OBS_FREQ]))
        truth_list.append(x.copy())
        if (k + 1) % 5 == 0:
            print(f"             assim phase: {k+1}/{n_assim} "
                  f"({time.perf_counter() - t0:.0f}s)")

    times_list = np.arange(n_total) * OBS_FREQ
    from pyteda.io import save_truth_trajectory
    save_truth_trajectory(truth_list, times_list, cache_path)
    print(f"             done in {time.perf_counter() - t0:.1f}s")
    return truth_list, times_list


# ======================================================================
# Build the 5 scenarios sharing the cached artefacts
# ======================================================================
def build_scenarios(model, x0_ref, xb, X0, truth_list, times_list,
                    obs_indices):
    print()
    print("=" * 72)
    print(" Building SWE scenarios (sharing x0_ref / xb / initial_ensemble"
          " / truth)")
    print("=" * 72)

    n_obs = obs_indices.size
    scenarios = []
    for s in SCENARIO_SEEDS:
        scen = Scenario.generate(
            model=model,
            operator_factory=lambda rng, idx=obs_indices: LinearSelection(
                m=n_obs, n_state=model.dim, indices=idx,
            ),
            noise=IsotropicDiagonal(std=NOISE_STD, dim=n_obs),
            x0_ref=x0_ref, xb=xb, initial_ensemble=X0,
            truth_trajectory=truth_list,        # share the pre-built truth
            spinup_xb=SPINUP_XB, spinup_ensemble=SPINUP_ENSEMBLE,
            obs_freq=OBS_FREQ, end_time=END_TIME,
            seed=s,
        )
        scenarios.append(scen)
        print(f"  scenario seed={s}  n_steps={scen.n_steps}  "
              f"obs/step={scen.dim_obs}")

    print(f"\n  built {len(scenarios)} scenarios. "
          f"Each runs for {END_TIME / DAY:.0f} days "
          f"with {OBS_FREQ / HOUR:.0f}h between assimilations.")
    return scenarios


# ======================================================================
# Run the benchmark
# ======================================================================
def run_benchmark(scenarios):
    n_cells = len(METHODS) * len(scenarios)
    print()
    print("=" * 72)
    print(f" Running benchmark — {len(METHODS)} methods × "
          f"{len(scenarios)} scenarios = {n_cells} cells")
    print(f" LMAX={LMAX}, {END_TIME/DAY:.0f}-day assimilation window. "
          "This will take a while.")
    print("=" * 72)

    t0 = time.perf_counter()
    results = Benchmark(
        scenarios=scenarios,
        methods=METHODS,
        n_runs_per_method=1,
        inflation_factor=INFLATION,
        method_seed_base=1000,
        parallel=False,
        verbose=True,
        store_diagnostics=True,
        store_states_at=SNAP_FRACTIONS,
    ).run()
    elapsed = time.perf_counter() - t0
    print(f"\n  benchmark done in {elapsed/60:.1f} minutes "
          f"({elapsed:.0f}s)")

    print()
    print("--- summary_table() ---")
    print(results.summary_table().to_string(index=False))
    print()
    print("--- diagnostics_summary() ---")
    print(results.diagnostics_summary().to_string(index=False))
    return results


# ======================================================================
# Aggregation helper (median + 16-84% band over scenarios)
# ======================================================================
def _aggregate(results, kind="error_a"):
    by_method = {}
    times = None
    for row in results.rows:
        if times is None:
            times = row["times"]
        by_method.setdefault(row["method"], []).append(row[kind])
    out = {}
    for m, curves in by_method.items():
        arr = np.stack(curves, axis=0)
        median = np.median(arr, axis=0)
        low = np.percentile(arr, 16, axis=0)
        high = np.percentile(arr, 84, axis=0)
        out[m] = (times, median, low, high)
    return out


def _safe_name(s):
    return (s.replace("(", "_").replace(")", "")
             .replace("=", "").replace(" ", "_")
             .replace("/", "_"))


# ======================================================================
# Plots
# ======================================================================
def plot_rmse_evolution(results, savedir):
    agg = _aggregate(results, kind="error_a")
    methods_sorted = sorted(agg.keys(),
                            key=lambda m: float(agg[m][1].mean()))

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate(methods_sorted):
        t, median, low, high = agg[m]
        color = cmap(i % 20)
        # convert to days for readable x-axis
        t_days = t / DAY
        ax.plot(t_days, median, lw=1.6, color=color, label=m)
        ax.fill_between(t_days, low, high,
                        color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("time (days)")
    ax.set_ylabel(r"RMSE$_a$  (median, 16–84% band over scenarios)")
    ax.set_yscale("log")
    ax.set_title("SWE — analysis-error evolution per method")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(savedir, "01_error_curves_all.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_analysis_vs_background(results, savedir):
    agg_a = _aggregate(results, kind="error_a")
    agg_b = _aggregate(results, kind="error_b")
    methods_sorted = sorted(agg_a.keys(),
                            key=lambda m: float(agg_a[m][1].mean()))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = plt.get_cmap("tab20")
    method_handles = []
    for i, m in enumerate(methods_sorted):
        t, med_a, lo_a, hi_a = agg_a[m]
        _, med_b, lo_b, hi_b = agg_b[m]
        color = cmap(i % 20)
        t_days = t / DAY

        line_a, = ax.plot(t_days, med_a, lw=1.6, color=color, label=m)
        ax.fill_between(t_days, lo_a, hi_a,
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(t_days, med_b, lw=1.1, color=color, ls="--", alpha=0.85)
        ax.fill_between(t_days, lo_b, hi_b,
                        color=color, alpha=0.08, linewidth=0)
        method_handles.append(line_a)

    style_a = plt.Line2D([], [], color="grey", lw=1.6, label="analysis (solid)")
    style_b = plt.Line2D([], [], color="grey", lw=1.1, ls="--",
                         label="background (dashed)")
    leg1 = ax.legend(handles=method_handles,
                     loc="center left", bbox_to_anchor=(1.02, 0.65),
                     fontsize=9, frameon=False, title="method")
    ax.add_artist(leg1)
    ax.legend(handles=[style_a, style_b],
              loc="center left", bbox_to_anchor=(1.02, 0.15),
              fontsize=9, frameon=False, title="line style")

    ax.set_xlabel("time (days)")
    ax.set_ylabel(r"RMSE  (median, 16–84% band over scenarios)")
    ax.set_yscale("log")
    ax.set_title("SWE — analysis (solid) vs background (dashed) per method")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(savedir, "02_analysis_vs_background.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_per_method_curves(results, savedir):
    agg_a = _aggregate(results, kind="error_a")
    agg_b = _aggregate(results, kind="error_b")
    methods_sorted = sorted(agg_a.keys(),
                            key=lambda m: float(agg_a[m][1].mean()))
    cmap = plt.get_cmap("tab20")

    subdir = os.path.join(savedir, "per_method")
    os.makedirs(subdir, exist_ok=True)

    for i, m in enumerate(methods_sorted):
        t, med_a, lo_a, hi_a = agg_a[m]
        _, med_b, lo_b, hi_b = agg_b[m]
        color = cmap(i % 20)
        t_days = t / DAY

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t_days, med_b, lw=1.4, color=color, ls="--",
                label="background (median)")
        ax.fill_between(t_days, lo_b, hi_b, color=color, alpha=0.10,
                        linewidth=0, label="background 16–84% band")
        ax.plot(t_days, med_a, lw=2.0, color=color,
                label="analysis (median)")
        ax.fill_between(t_days, lo_a, hi_a, color=color, alpha=0.25,
                        linewidth=0, label="analysis 16–84% band")

        ax.set_xlabel("time (days)")
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.set_title(f"SWE — {m}\nanalysis vs background  ·  "
                     f"{N_SCENARIOS} scenarios")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="0.85")
        fig.tight_layout()

        safe = _safe_name(m)
        out = os.path.join(subdir, f"curves_{safe}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_method_boxplot(results, savedir, burn_in_frac=0.30):
    method_color = {}
    cmap = plt.get_cmap("tab20")
    by_method_a, by_method_b = {}, {}
    times = None
    for r in results.rows:
        if times is None:
            times = r["times"]
        by_method_a.setdefault(r["method"], []).append(r["error_a"])
        by_method_b.setdefault(r["method"], []).append(r["error_b"])

    methods_sorted = sorted(
        by_method_a.keys(),
        key=lambda m: float(np.mean(np.stack(by_method_a[m]))),
    )
    for i, m in enumerate(methods_sorted):
        method_color[m] = cmap(i % 20)

    n_steps = len(times)
    cutoff = int(np.ceil(burn_in_frac * n_steps))
    burn_t_days = times[cutoff] / DAY if cutoff < n_steps else times[-1] / DAY

    rmse_a_by_method = {
        m: np.array([curve[cutoff:].mean() for curve in by_method_a[m]])
        for m in methods_sorted
    }
    rmse_b_by_method = {
        m: np.array([curve[cutoff:].mean() for curve in by_method_b[m]])
        for m in methods_sorted
    }

    for yscale in ("log", "linear"):
        fig, ax = plt.subplots(figsize=(15, 6))
        positions_a = np.arange(len(methods_sorted)) * 3.0 + 0.0
        positions_b = np.arange(len(methods_sorted)) * 3.0 + 1.0

        for i, m in enumerate(methods_sorted):
            color = method_color[m]
            color_light = (*color[:3], 0.45)

            ax.boxplot(
                rmse_a_by_method[m], positions=[positions_a[i]],
                widths=0.7, patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor="#222",
                              linewidth=1.0, alpha=0.85),
                medianprops=dict(color="white", linewidth=2.0),
                whiskerprops=dict(color="#222"),
                capprops=dict(color="#222"),
                showfliers=False,
            )
            ax.scatter(
                np.full_like(rmse_a_by_method[m], positions_a[i])
                + np.random.uniform(-0.18, 0.18, size=len(rmse_a_by_method[m])),
                rmse_a_by_method[m],
                color="#111", s=14, zorder=3, alpha=0.7, linewidth=0,
            )

            ax.boxplot(
                rmse_b_by_method[m], positions=[positions_b[i]],
                widths=0.7, patch_artist=True,
                boxprops=dict(facecolor=color_light, edgecolor="#222",
                              linewidth=1.0, alpha=0.85, hatch="///"),
                medianprops=dict(color="#111", linewidth=2.0),
                whiskerprops=dict(color="#222"),
                capprops=dict(color="#222"),
                showfliers=False,
            )
            ax.scatter(
                np.full_like(rmse_b_by_method[m], positions_b[i])
                + np.random.uniform(-0.18, 0.18, size=len(rmse_b_by_method[m])),
                rmse_b_by_method[m],
                color="#555", s=14, zorder=3, alpha=0.6, linewidth=0,
            )

        tick_positions = np.arange(len(methods_sorted)) * 3.0 + 0.5
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(methods_sorted, rotation=30, ha="right",
                           fontsize=10)
        ax.set_ylabel(f"RMSE  (mean over t > {burn_t_days:.1f} days)")
        ax.set_yscale(yscale)
        ax.set_title(
            f"SWE — RMSE distribution across {N_SCENARIOS} scenarios  ·  "
            f"burn-in = first {int(burn_in_frac*100)}% discarded  ·  "
            f"y-{yscale}"
        )
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.set_axisbelow(True)

        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor="grey", edgecolor="#222", alpha=0.85,
                  label="analysis"),
            Patch(facecolor="lightgrey", edgecolor="#222", alpha=0.85,
                  hatch="///", label="background"),
        ]
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=10,
                  facecolor="white", edgecolor="0.85")

        fig.tight_layout()
        out = os.path.join(savedir, f"06_boxplot_{yscale}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_rank_histogram_per_method(results, savedir):
    subdir = os.path.join(savedir, "per_method")
    os.makedirs(subdir, exist_ok=True)
    summary = results.summary_table().sort_values("mean_rmse_a")
    methods_sorted = summary["method"].tolist()
    cmap = plt.get_cmap("tab20")

    for i, m in enumerate(methods_sorted):
        fig, ax = plt.subplots(figsize=(8, 4))
        results.plot_rank_histogram(m, kind="analysis", ax=ax)
        color = cmap(i % 20)
        for patch in ax.patches:
            patch.set_facecolor(color)
            patch.set_edgecolor("#222")
            patch.set_alpha(0.85)
        ax.set_title(f"SWE — rank histogram (analysis)  ·  {m}")
        fig.tight_layout()
        safe = _safe_name(m)
        out = os.path.join(subdir, f"rank_histogram_{safe}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_spread_vs_error(results, savedir):
    fig, ax = plt.subplots(figsize=(11, 6))
    results.plot_spread_vs_error(ax=ax, kind="analysis")
    ax.set_title("SWE — analysis spread (dashed) vs RMSE (solid)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(savedir, "03_spread_vs_error.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------
# Spatial maps — for each (method, snapshot) plot truth, analysis mean,
# and analysis - truth as 2D maps on the (lat, lon) grid.
# ----------------------------------------------------------------------
def plot_state_maps(model, scenarios, results, savedir,
                     scenario_id=0, var="h"):
    """Render lat-lon maps of truth, analysis mean, and residual at each
    snapshot for every method. One PNG per (method, snapshot)."""
    grid = model.grid
    Nlat, Nlon = grid.Nlat, grid.Nlon
    lats = np.rad2deg(np.linspace(np.pi / 2, -np.pi / 2, Nlat))
    lons = np.rad2deg(np.linspace(0, 2 * np.pi, Nlon, endpoint=False))

    var_block = model.var_blocks[var]

    subdir = os.path.join(savedir, "maps")
    os.makedirs(subdir, exist_ok=True)

    summary = results.summary_table().sort_values("mean_rmse_a")
    methods_sorted = summary["method"].tolist()

    rows_by_method = {}
    for row in results.rows:
        if (row["scenario_id"] == scenario_id
                and row["method"] in methods_sorted
                and "Xa_snapshots" in row):
            rows_by_method[row["method"]] = row
    if not rows_by_method:
        print("  [maps] no snapshots stored — skipping")
        return

    scen = scenarios[scenario_id]

    # global colour scale for truth/analysis (per variable)
    truth_field_max = max(
        np.abs(np.asarray(scen.truth_trajectory[k])[var_block]).max()
        for k in range(scen.n_steps)
    )
    vmax_state = float(truth_field_max)
    vmin_state = -vmax_state

    for method in methods_sorted:
        row = rows_by_method[method]
        Xa = row["Xa_snapshots"]
        snap_steps = row["snapshot_steps"]
        snap_times = row["snapshot_times"]
        snap_fracs = row["snapshot_fractions"]

        for ci in range(Xa.shape[0]):
            step = int(snap_steps[ci])
            t_days = float(snap_times[ci]) / DAY
            frac_pct = int(round(float(snap_fracs[ci]) * 100))

            mean_a = Xa[ci].mean(axis=1)
            truth = np.asarray(scen.truth_trajectory[step])

            f_truth = truth[var_block].reshape(Nlat, Nlon)
            f_a     = mean_a[var_block].reshape(Nlat, Nlon)
            f_diff  = f_a - f_truth

            # use a tighter colour scale for the residual
            vmax_diff = float(np.abs(f_diff).max())
            vmax_diff = max(vmax_diff, 1e-6)

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.2),
                                      facecolor="white")
            extent = [lons[0], lons[-1], lats[-1], lats[0]]

            im0 = axes[0].imshow(f_truth, extent=extent, aspect="auto",
                                  cmap="RdBu_r",
                                  vmin=vmin_state, vmax=vmax_state)
            axes[0].set_title(f"Truth — {var}")
            fig.colorbar(im0, ax=axes[0], shrink=0.8)

            im1 = axes[1].imshow(f_a, extent=extent, aspect="auto",
                                  cmap="RdBu_r",
                                  vmin=vmin_state, vmax=vmax_state)
            axes[1].set_title(f"Analysis mean — {var}")
            fig.colorbar(im1, ax=axes[1], shrink=0.8)

            im2 = axes[2].imshow(f_diff, extent=extent, aspect="auto",
                                  cmap="RdBu_r",
                                  vmin=-vmax_diff, vmax=vmax_diff)
            axes[2].set_title("Analysis − Truth (residual)")
            fig.colorbar(im2, ax=axes[2], shrink=0.8)

            for ax in axes:
                ax.set_xlabel("longitude (°)")
                ax.set_ylabel("latitude (°)")

            fig.suptitle(
                f"SWE  ·  {method}  ·  t = {t_days:.2f} days  "
                f"({frac_pct}%)  ·  scenario {scenario_id}  ·  variable {var}",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            safe = _safe_name(method)
            out = os.path.join(subdir,
                               f"map_{safe}_t{frac_pct:03d}_{var}.png")
            fig.savefig(out, dpi=130, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
        print(f"  saved {Xa.shape[0]} maps for method '{method}' "
              f"in {subdir}/  (variable: {var})")


def plot_observation_mask(model, mask, savedir):
    """Save a quick figure of the continental mask used for observations."""
    grid = model.grid
    Nlat, Nlon = grid.Nlat, grid.Nlon
    lats = np.rad2deg(np.linspace(np.pi / 2, -np.pi / 2, Nlat))
    lons = np.rad2deg(np.linspace(0, 2 * np.pi, Nlon, endpoint=False))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(mask, extent=[lons[0], lons[-1], lats[-1], lats[0]],
              aspect="auto", cmap="Greys", vmin=0, vmax=1)
    coverage = mask.mean()
    ax.set_title(f"Observation mask  ·  approx. continental coverage  "
                 f"·  {coverage*100:.1f}% of grid")
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    fig.tight_layout()
    out = os.path.join(savedir, "00_observation_mask.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 72)
    print(" SWE setup")
    print("=" * 72)
    print(f"  LMAX = {LMAX}  ·  state_vars = {STATE_VARS}  ·  dt = {DT:.0f}s")
    model = SWEModel(LMAX=LMAX, dt=DT, state_vars=STATE_VARS)
    print(f"  grid: {model.grid.Nlat} × {model.grid.Nlon}  "
          f"·  field_size = {model.field_size}  "
          f"·  state dim = {model.dim}")

    # 1. Build observation mask
    mask = _continental_mask(model.grid, fraction=OBS_FRACTION, seed=0)
    obs_indices, n_keep_per_field = _obs_indices_from_mask(mask, model)
    print(f"  observation mask: {n_keep_per_field} of "
          f"{model.field_size} grid points per variable")
    print(f"  total observations: {obs_indices.size} "
          f"(across {len(STATE_VARS)} variables)")
    plot_observation_mask(model, mask, FIG_DIR)

    # 2. Build / load cached spinups
    print()
    print("=" * 72)
    print(" Spinup phase (cached on disk in ./{}/)".format(CACHE_DIR))
    print("=" * 72)
    x0_ref, xb, X0 = build_or_load_artifacts(model)

    # 2b. Build / load the truth trajectory (also cached)
    truth_list, times_list = build_or_load_truth_trajectory(model)

    # 3. Build scenarios
    scenarios = build_scenarios(model, x0_ref, xb, X0,
                                truth_list, times_list, obs_indices)

    # 4. Run benchmark
    results = run_benchmark(scenarios)

    # 5. Export CSVs
    print()
    print("=" * 72)
    print(" Exporting CSVs")
    print("=" * 72)
    written = results.export_csv(FIG_DIR)
    for name, path in written.items():
        print(f"  saved {path}")

    # 6. Plot
    print()
    print("=" * 72)
    print(" Plotting")
    print("=" * 72)
    plot_rmse_evolution(results, FIG_DIR)
    plot_analysis_vs_background(results, FIG_DIR)
    plot_per_method_curves(results, FIG_DIR)
    plot_method_boxplot(results, FIG_DIR, burn_in_frac=0.30)
    plot_spread_vs_error(results, FIG_DIR)
    plot_rank_histogram_per_method(results, FIG_DIR)

    # spatial maps for each variable
    for var in STATE_VARS:
        plot_state_maps(model, scenarios, results, FIG_DIR,
                        scenario_id=0, var=var)

    print()
    print("Done. Figures in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()