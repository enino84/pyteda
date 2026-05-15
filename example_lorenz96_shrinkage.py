# -*- coding: utf-8 -*-
"""
Lorenz96 — full experiment: 14 methods × 10 scenarios.

Companion to: "An ensemble Kalman filter implementation based on
shrinkage estimators of the precision matrix via modified Cholesky
decomposition" (Niño-Ruiz, J. Comput. Appl. Math., in preparation).

This is the EXTENDED comparison, including the three new
precision-space shrinkage criteria (Frobenius MSE, Stein, DA-aware)
introduced in the paper. The earlier file ``example_lorenz96.py``
covered the baseline comparison without these three.

Configuration:
  Phase 1 (truth):      synthetic IC -> propagate 10 time units -> x0_ref
                        x0_ref is computed ONCE and shared across the 10 scenarios.
  Phase 2 (xb):         kick + propagate 10 time units -> xb     (varies by seed)
  Phase 3 (ensemble):   per-member kick + propagate 10 time units -> X_b (varies)
  Assimilation:         obs_freq = 0.5, end_time = 10.0  (~21 obs cycles)
  Observation noise:    isotropic Gaussian, std = 0.01
  Methods:              14 from the registry (incl. LW, RBLW, and the 3
                        new precision-space criteria)
  Scenarios per method: 10 (different observation network and initial ensemble)

Plots:
  01_error_curves_all.png       Mean RMSE_a(t) per method, ±1σ band, log-Y.
  02_analysis_vs_background.png Same plus background (dashed). All methods, log-Y.
  03_spread_vs_error.png        Built-in diagnostic.
  06_boxplot_log/linear.png     Per-method RMSE distributions (analysis + background).
  per_method/                   One PNG per method: curves + rank histogram.
  radars/                       One PNG per (top method, snapshot).

Run:

    python example_lorenz96_shrinkage.py

Figures land in ./lorenz96_shrinkage_figs/ (separate from the baseline run).
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario, Benchmark


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N_STATE         = 40
N_OBS           = 32
NOISE_STD       = 0.01
ENSEMBLE_SIZE   = 20
OBS_FREQ        = 0.50
END_TIME        = 10.0
INFLATION       = 1.04

SPINUP_TRUTH    = 10.0
PERT_XB         = 0.5
SPINUP_XB       = 10.0
PERT_ENSEMBLE   = 0.05
SPINUP_ENSEMBLE = 10.0

N_SCENARIOS     = 10
SCENARIO_SEEDS  = list(range(42, 42 + N_SCENARIOS))

SNAP_FRACTIONS  = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TOP_METHODS   = 6

METHODS = {
    # Plain stochastic EnKF and its variants without shrinkage
    "EnKF":                 dict(method="enkf"),
    "EnKF-Cholesky":        dict(method="enkf-cholesky"),
    "EnKF-Naive":           dict(method="enkf-naive"),
    "EnKF-BLoc":            dict(method="enkf-b-loc"),

    # Modified-Cholesky family
    "EnKF-MC(r=2)":         dict(method="enkf-modified-cholesky", r=2),

    # Covariance-space shrinkage — Niño-Ruiz, Guzman, Jabba 2021 baselines
    "EnKF-LW":              dict(method="enkf-lw"),
    "EnKF-RBLW":            dict(method="enkf-rblw"),

    # Precision-space shrinkage — three principled criteria from the
    # new paper. All use modified-Cholesky target with r=2 and
    # truncated-SVD pseudo-inverse of P^b with rtol=0.1.
    "EnKF-Sh-Binv-MSE":     dict(method="enkf-shrinkage-binv-mse",
                                   r=2, rtol_pseudo_inverse=0.05),
    "EnKF-Sh-Binv-Stein":   dict(method="enkf-shrinkage-binv-stein",
                                   r=2, rtol_pseudo_inverse=0.05),
    "EnKF-Sh-Binv-DA":      dict(method="enkf-shrinkage-binv-da",
                                   r=2, rtol_pseudo_inverse=0.05),

    # Square-root and transform filters
    "ETKF":                 dict(method="etkf"),
    "EnSRF":                dict(method="ensrf"),

    # Localised filters
    "LEnKF(r=2)":           dict(method="lenkf", r=2),
    "LETKF(r=2)":           dict(method="letkf", r=2),
}

FIG_DIR = "lorenz96_shrinkage_figs"


def build_scenarios():
    print("=" * 72)
    print(" Building Lorenz96 scenarios (3-phase recipe, shared x0_ref)")
    print("=" * 72)

    model = Lorenz96(n=N_STATE)
    n = model.get_number_of_variables()

    x0_synth = model.get_initial_condition()
    x0_ref = model.propagate(x0_synth, np.array([0.0, SPINUP_TRUTH]))
    print(f"  x0_ref built once  (||x0_ref|| = {np.linalg.norm(x0_ref):.3f})")

    scenarios = []
    for s in SCENARIO_SEEDS:
        scen = Scenario.generate(
            model=model,
            operator_factory=lambda rng: LinearSelection(
                m=N_OBS, n_state=n, rng=rng,
            ),
            noise=IsotropicDiagonal(std=NOISE_STD, dim=N_OBS),
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
    ax.set_ylabel(r"RMSE$_a$  (median, 16–84% band over 10 scenarios)")
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
    ax.set_ylabel(r"RMSE  (median, 16–84% band over 10 scenarios)")
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
                        linewidth=0, label="background 16–84% band")
        ax.plot(t, med_a, lw=2.0, color=color, label="analysis (median)")
        ax.fill_between(t, lo_a, hi_a, color=color, alpha=0.25,
                        linewidth=0, label="analysis 16–84% band")
        ax.set_xlabel("time")
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.set_title(f"Lorenz96 — {m}\nanalysis vs background  ·  10 scenarios")
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
            f"Lorenz96 — RMSE distribution across 10 scenarios  ·  "
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


def _radar_panel(ax, angles, truth, mean_b, mean_a,
                 obs_indices, obs_values, F_shift,
                 rmin, rmax, title):
    ang_c = np.concatenate([angles, angles[:1]])
    truth_c = np.concatenate([truth, truth[:1]]) + F_shift
    mb_c    = np.concatenate([mean_b, mean_b[:1]]) + F_shift
    ma_c    = np.concatenate([mean_a, mean_a[:1]]) + F_shift
    ax.plot(ang_c, mb_c, color="#7a7a7a", lw=1.3, ls="--",
            label="background mean", zorder=2)
    ax.plot(ang_c, truth_c, color="#111111", lw=2.2,
            label="truth", zorder=4)
    ax.plot(ang_c, ma_c, color="#1f4ed8", lw=1.8,
            label="analysis mean", zorder=5)
    if obs_indices is not None and len(obs_indices) > 0:
        obs_ang = angles[obs_indices]
        obs_r = obs_values + F_shift
        ax.scatter(obs_ang, obs_r,
                   s=28, color="#dc2626",
                   edgecolor="white", linewidth=0.8,
                   zorder=6, label="observations")
    ax.set_ylim(rmin, rmax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[::5])
    ax.set_xticklabels([str(i) for i in range(0, len(angles), 5)],
                       fontsize=14, color="#222", fontweight="bold")
    ax.set_yticklabels([])
    ax.tick_params(axis="x", pad=10)
    ax.grid(True, alpha=0.32, color="#bbbbbb", linestyle=":")
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title(title, fontsize=12, pad=14, fontweight="bold",
                 color="#222222")


def plot_radar_per_method(scenarios, results, savedir,
                           n_top=N_TOP_METHODS, scenario_id=0):
    summary = results.summary_table().sort_values("mean_rmse_a")
    top_methods = summary.head(n_top)["method"].tolist()
    rows_by_method = {}
    for row in results.rows:
        if (row["method"] in top_methods
                and row["scenario_id"] == scenario_id
                and "Xa_snapshots" in row):
            rows_by_method[row["method"]] = row
    if not rows_by_method:
        print("  [radar] no snapshots stored — skipping")
        return
    scen = scenarios[scenario_id]
    n_state = scen.n_state
    F_shift = 8.0
    all_vals = []
    for row in rows_by_method.values():
        all_vals.append(row["Xa_snapshots"].ravel())
    for k in range(scen.n_steps):
        all_vals.append(np.asarray(scen.truth_trajectory[k]))
    vmin = min(arr.min() for arr in all_vals) + F_shift
    vmax = max(arr.max() for arr in all_vals) + F_shift
    rmin = max(0.0, vmin - 0.5)
    rmax = vmax + 0.5
    angles = np.linspace(0, 2 * np.pi, n_state, endpoint=False)
    radar_dir = os.path.join(savedir, "radars")
    os.makedirs(radar_dir, exist_ok=True)
    op_indices = None
    if hasattr(scen.operators[0], "indices"):
        op_indices = np.asarray(scen.operators[0].indices)
    for method, row in rows_by_method.items():
        Xb = row["Xb_snapshots"]
        Xa = row["Xa_snapshots"]
        snap_steps = row["snapshot_steps"]
        snap_times = row["snapshot_times"]
        snap_fracs = row["snapshot_fractions"]
        for ci in range(Xa.shape[0]):
            step = int(snap_steps[ci])
            t_at = float(snap_times[ci])
            frac_pct = int(round(float(snap_fracs[ci]) * 100))
            mean_b = Xb[ci].mean(axis=1)
            mean_a = Xa[ci].mean(axis=1)
            truth = np.asarray(scen.truth_trajectory[step])
            obs_vals = None
            obs_idx = op_indices
            if obs_idx is not None:
                y_k = np.asarray(scen.observations[step])
                obs_vals = y_k
            fig = plt.figure(figsize=(6.4, 6.4), facecolor="white")
            ax = fig.add_subplot(111, projection="polar")
            title = (f"{method}   ·   t = {t_at:.2f}   ({frac_pct:>3d}%)")
            _radar_panel(
                ax, angles,
                truth=truth,
                mean_b=mean_b, mean_a=mean_a,
                obs_indices=obs_idx, obs_values=obs_vals,
                F_shift=F_shift, rmin=rmin, rmax=rmax,
                title=title,
            )
            handles = [
                plt.Line2D([], [], color="#111111", lw=2.2, label="truth"),
                plt.Line2D([], [], color="#1f4ed8", lw=1.8, label="analysis mean"),
                plt.Line2D([], [], color="#7a7a7a", lw=1.3, ls="--",
                           label="background mean"),
                plt.Line2D([], [], marker="o", linestyle="",
                           markerfacecolor="#dc2626",
                           markeredgecolor="white", markersize=7,
                           label="observations"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=4,
                       frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))
            fig.tight_layout(rect=[0, 0.06, 1, 0.99])
            fname = f"radar_{_safe_name(method)}_t{frac_pct:03d}.png"
            out = os.path.join(radar_dir, fname)
            fig.savefig(out, dpi=140, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
        print(f"  saved {Xa.shape[0]} radars for method '{method}' "
              f"in {radar_dir}/")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model, scenarios = build_scenarios()
    results = run_benchmark(scenarios)

    print()
    print("=" * 72)
    print(" Exporting CSVs")
    print("=" * 72)
    written = results.export_csv(FIG_DIR)
    for name, path in written.items():
        print(f"  saved {path}")

    print()
    print("=" * 72)
    print(" Plotting")
    print("=" * 72)
    plot_rmse_evolution(results, FIG_DIR)
    plot_analysis_vs_background(results, FIG_DIR)
    plot_per_method_curves(results, FIG_DIR)
    plot_method_boxplot(results, FIG_DIR, burn_in_frac=0.30)
    plot_spread_vs_error(results, FIG_DIR)

    summary = results.summary_table().sort_values("mean_rmse_a")
    best = summary.iloc[0]["method"]
    print(f"  best method by mean_rmse_a: {best}")
    plot_rank_histogram_per_method(results, FIG_DIR)
    plot_radar_per_method(scenarios, results, FIG_DIR, n_top=N_TOP_METHODS)

    print()
    print("Done. Figures in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()