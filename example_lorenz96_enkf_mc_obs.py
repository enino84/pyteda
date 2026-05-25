# -*- coding: utf-8 -*-
"""
Lorenz96 — full experiment: 14 methods × 10 scenarios.

Companion to: "An ensemble Kalman filter implementation based on
shrinkage estimators of the precision matrix via modified Cholesky
decomposition" (Niño-Ruiz, J. Comput. Appl. Math., in preparation).
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal, strided_indices
from pyteda.experiments import Scenario, Benchmark

from datetime import datetime




# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N_STATE         = 40
NOISE_STD       = 0.01
ENSEMBLE_SIZE   = 20
OBS_FREQ        = 0.30
END_TIME        = 50.0
INFLATION       = 1.14

# Adaptive multiplicative inflation (framework-level, innovation-based).
# When this is a dict, it OVERRIDES the fixed INFLATION above and is applied
# uniformly to EVERY method each cycle (fair comparison). Set to None to fall
# back to the fixed INFLATION factor.
#   lambda0 : initial factor          gain : smoothing of the recursion
#   lo, hi  : clip bounds for the factor
# Note: tuned for NOISE_STD~0.05. With small NOISE_STD (e.g. 0.01) consider
# lowering `hi` (e.g. 1.4) if you see over-inflation in spread_error_ratio.
ADAPTIVE_INFLATION = dict(lambda0=1.04, gain=0.15, lo=1.0, hi=1.6)

# Fraction of initial (spin-up) cycles discarded when computing the
# time-averaged CSV metrics (mean_rmse_a, mean_spread_a, CRPS, and the
# spread/error ratio). The transient while the filters are still
# converging is not representative of steady-state skill, so for DA
# reporting this is set to 0.2-0.3. final_rmse_a / median_rmse_a are
# unaffected (already steady-state-representative).
BURN_IN_FRAC    = 0.30

SPINUP_TRUTH    = 10.0
PERT_XB         = 0.5
SPINUP_XB       = 10.0
PERT_ENSEMBLE   = 0.05
SPINUP_ENSEMBLE = 10.0

# Observation network: strided sampling, observe 1 of every SPACING
# variables (spacing=2 -> 20 of 40 = 50%; spacing=3 -> ~35%).
SPACING         = 2

# Modified-Cholesky regression method for the Cholesky-based filters
# (EnKF-MC and the three precision-space shrinkage criteria): solve each
# local conditional regression either by Ridge regression ("ridge",
# default) or by truncated-SVD pseudo-inverse ("svd") with relative
# tolerance CHOL_TOL.
CHOL_METHOD     = "ridge"
CHOL_TOL        = 0.35

# One run per method.
N_SCENARIOS     = 1
SCENARIO_SEEDS  = list(range(42, 42 + N_SCENARIOS))

SNAP_FRACTIONS  = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TOP_METHODS   = 6

METHODS = {

    "EnKF-OBS-MC(r=2)": dict(method="enkf-obs-modified-cholesky",
                            r=3, r_state=3, b_via_cholesky=True, alpha=0.1),

    "EnKF-OBS-MC-LOCAL(r=2)": dict(method="enkf-obs-modified-cholesky-local",
                            r=2, alpha=0.1),

    # Plain stochastic EnKF and its variants without shrinkage
    "EnKF-BLoc":            dict(method="enkf-b-loc"),

    # Localised filters  (labels match the actual radius below)
    "LEnKF(r=3)":           dict(method="lenkf", r=3),
    "LETKF(r=3)":           dict(method="letkf", r=3),
}

FIG_DIR = f"lorenz96_mc_obs_sp{SPACING}_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def build_scenarios():
    print("=" * 72)
    print(" Building Lorenz96 scenarios (3-phase recipe, shared x0_ref)")
    print("=" * 72)

    model = Lorenz96(n=N_STATE)
    n = model.get_number_of_variables()

    # Strided observation network: 1 of every SPACING variables.
    obs_indices, n_obs = strided_indices(model, spacing=SPACING)
    print(f"  strided spacing={SPACING}: {n_obs} of {n} variables observed "
          f"({100*n_obs/n:.0f}%)")

    x0_synth = model.get_initial_condition()
    x0_ref = model.propagate(x0_synth, np.array([0.0, SPINUP_TRUTH]))
    print(f"  x0_ref built once  (||x0_ref|| = {np.linalg.norm(x0_ref):.3f})")

    scenarios = []
    for s in SCENARIO_SEEDS:
        scen = Scenario.generate(
            model=model,
            operator_factory=lambda rng, idx=obs_indices: LinearSelection(
                m=idx.size, n_state=n, indices=idx,
            ),
            noise=IsotropicDiagonal(std=NOISE_STD, dim=n_obs),
            ensemble_size=ENSEMBLE_SIZE,
            x0_ref=x0_ref,
            pert_xb=PERT_XB, spinup_xb=SPINUP_XB,
            pert_ensemble=PERT_ENSEMBLE,
            spinup_ensemble=SPINUP_ENSEMBLE,
            obs_freq=OBS_FREQ, end_time=END_TIME,
            seed=s,
        )
        scenarios.append(scen)

    print(f"  built {len(scenarios)} scenarios with seeds {SCENARIO_SEEDS}")
    print(f"  n_steps per scenario = {scenarios[0].n_steps}")
    print(f"  obs per step         = {scenarios[0].dim_obs}")
    return model, scenarios


def run_benchmark(scenarios):
    n_cells = len(METHODS) * len(scenarios)
    print()
    print("=" * 72)
    print(f" Running benchmark — {len(METHODS)} methods × "
          f"{len(scenarios)} scenarios = {n_cells} cells")
    if ADAPTIVE_INFLATION is not None:
        print(f" Adaptive inflation: {ADAPTIVE_INFLATION}")
    else:
        print(f" Fixed inflation factor: {INFLATION}")
    print("=" * 72)

    results = Benchmark(
        scenarios=scenarios,
        methods=METHODS,
        n_runs_per_method=1,
        inflation_factor=INFLATION,
        method_seed_base=1000,
        parallel=False,
        verbose=False,
        store_diagnostics=True,
        store_states_at=SNAP_FRACTIONS,
        adaptive_inflation_cfg=ADAPTIVE_INFLATION,
    ).run()

    print()
    print("--- summary_table() ---")
    print(results.summary_table().to_string(index=False))
    print()
    print("--- diagnostics_summary() ---")
    print(results.diagnostics_summary().to_string(index=False))
    return results


def _aggregate(results, kind="error_a"):
    by_method: dict = {}
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


def plot_rmse_evolution(results, savedir):
    agg = _aggregate(results, kind="error_a")
    methods_sorted = sorted(agg.keys(),
                            key=lambda m: float(agg[m][1].mean()))
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate(methods_sorted):
        t, median, low, high = agg[m]
        color = cmap(i % 20)
        ax.plot(t, median, lw=1.6, color=color, label=m)
        ax.fill_between(t, low, high,
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("time")
    ax.set_ylabel(r"RMSE$_a$  (median over scenarios)")
    ax.set_yscale("log")
    ax.set_title("Lorenz96 — analysis-error evolution per method")
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
        line_a, = ax.plot(t, med_a, lw=1.6, color=color, label=m)
        ax.fill_between(t, lo_a, hi_a,
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(t, med_b, lw=1.1, color=color, ls="--", alpha=0.85)
        ax.fill_between(t, lo_b, hi_b,
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
    ax.set_xlabel("time")
    ax.set_ylabel(r"RMSE  (median over scenarios)")
    ax.set_yscale("log")
    ax.set_title("Lorenz96 — analysis (solid) vs background (dashed) per method")
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
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, med_b, lw=1.4, color=color, ls="--",
                label="background (median)")
        ax.fill_between(t, lo_b, hi_b, color=color, alpha=0.10,
                        linewidth=0, label="background 16-84% band")
        ax.plot(t, med_a, lw=2.0, color=color, label="analysis (median)")
        ax.fill_between(t, lo_a, hi_a, color=color, alpha=0.25,
                        linewidth=0, label="analysis 16-84% band")
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.set_title(f"Lorenz96 - {m}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="0.85")
        fig.tight_layout()
        safe = _safe_name(m)
        out = os.path.join(subdir, f"curves_{safe}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_spread_vs_error(results, savedir):
    fig, ax = plt.subplots(figsize=(11, 6))
    results.plot_spread_vs_error(ax=ax, kind="analysis")
    ax.set_title("Lorenz96 — analysis spread (dashed) vs RMSE (solid)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(savedir, "03_spread_vs_error.png")
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
    burn_t = times[cutoff] if cutoff < n_steps else times[-1]
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
        ax.set_ylabel(f"RMSE  (mean over t > {burn_t:.2f}, "
                      f"{n_steps - cutoff} steps)")
        ax.set_yscale(yscale)
        ax.set_title(
            f"Lorenz96 — RMSE distribution  ·  "
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
        ax.set_title(f"Lorenz96 — rank histogram (analysis)  ·  {m}")
        fig.tight_layout()
        safe = _safe_name(m)
        out = os.path.join(subdir, f"rank_histogram_{safe}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def _safe_name(s):
    return (s.replace("(", "_").replace(")", "")
             .replace("=", "").replace(" ", "_"))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"  output folder: {FIG_DIR}")
    print(f"  Cholesky regression: method={CHOL_METHOD}, tol={CHOL_TOL}")
    print(f"  CSV metrics: burn-in = first {int(BURN_IN_FRAC*100)}% of "
          f"cycles discarded (post-spin-up)")
    model, scenarios = build_scenarios()
    results = run_benchmark(scenarios)

    print()
    print("=" * 72)
    print(" Exporting CSVs")
    print("=" * 72)
    written = results.export_csv(FIG_DIR, burn_in_frac=BURN_IN_FRAC)
    for name, path in written.items():
        print(f"  saved {path}")

    print()
    print("=" * 72)
    print(" Plotting")
    print("=" * 72)
    plot_rmse_evolution(results, FIG_DIR)
    plot_analysis_vs_background(results, FIG_DIR)
    plot_per_method_curves(results, FIG_DIR)
    plot_method_boxplot(results, FIG_DIR, burn_in_frac=BURN_IN_FRAC)
    plot_spread_vs_error(results, FIG_DIR)

    summary = results.summary_table().sort_values("mean_rmse_a")
    best = summary.iloc[0]["method"]
    print(f"  best method by mean_rmse_a: {best}")
    plot_rank_histogram_per_method(results, FIG_DIR)

    print()
    print("Done. Figures in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()