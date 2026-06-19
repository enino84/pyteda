# -*- coding: utf-8 -*-
"""
Simulation v2.

Two modes are supported:

1) Legacy mode (preserved verbatim from the original implementation):

       sim = Simulation(model, background, analysis, observation, params={...})
       sim.run()
       sim.get_errors()

   Truth trajectory, observations, and the initial ensemble are generated
   inside `run()`. Each call to `run()` may differ unless numpy's global
   seed is set externally.

2) Scenario mode (recommended for benchmarks):

       sim = Simulation.from_scenario(scenario, analysis, inflation_factor=1.04)
       sim.run()

   The scenario provides the truth, observations, and initial ensemble;
   the filter only assimilates. This decouples the random-experiment
   setup from the filter and makes runs fully reproducible.

`Simulation.from_scenario` accepts either an `Analysis` instance or an
`AnalysisFactory` config dict; in the latter case the analysis is built
internally so different runs can be re-seeded independently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from ..background import Background
from ..observation import Observation


def _rmse_relative(xr: np.ndarray, xs: np.ndarray) -> float:
    return float(np.linalg.norm(xs - xr) / np.linalg.norm(xr))


def _rmse_relative_by_var(xr: np.ndarray, xs: np.ndarray, var_blocks) -> dict:
    """Relative RMSE computed independently per variable block.

    Each variable is normalized by the norm of its OWN truth block, so a
    large-magnitude field (e.g. geopotential height) no longer dominates a
    small-magnitude one (e.g. velocity). For a single-block model this
    returns exactly one entry equal to the global ``_rmse_relative``.
    """
    out = {}
    for name, sl in var_blocks.items():
        denom = np.linalg.norm(xr[sl])
        out[name] = float(np.linalg.norm(xs[sl] - xr[sl]) / denom) if denom > 0 else float("nan")
    return out


def _ensemble_spread(X: np.ndarray) -> float:
    """RMS of the per-component ensemble standard deviation.

    For a well-calibrated ensemble this should be close to RMSE.
    X has shape (n_state, N_ens).
    """
    # ddof=1 uses the unbiased estimator (consistent with what filters
    # implicitly assume when using N-1 in covariance estimation).
    sigma = np.std(X, axis=1, ddof=1)
    return float(np.sqrt(np.mean(sigma ** 2)))


def _ensemble_spread_by_var(X: np.ndarray, x_true: np.ndarray, var_blocks) -> dict:
    """Relative ensemble spread per variable block.

    Returns ``||sigma[block]|| / ||x_true[block]||`` so it shares the same
    normalization as ``_rmse_relative_by_var``. With that, the per-variable
    spread/error ratio = spread/rmse is unit-invariant and ≈ 1 when the
    block is well calibrated, regardless of the field's physical units.
    """
    sigma = np.std(X, axis=1, ddof=1)
    out = {}
    for name, sl in var_blocks.items():
        denom = np.linalg.norm(x_true[sl])
        out[name] = float(np.linalg.norm(sigma[sl]) / denom) if denom > 0 else float("nan")
    return out


def _ensemble_crps(X: np.ndarray, x_true: np.ndarray) -> float:
    """Continuous Ranked Probability Score, averaged over state components.

    CRPS_i = mean_j |X[i,j] - y[i]| - 0.5 * mean_{j,k} |X[i,j] - X[i,k]|

    Uses the sort-based O(N log N) formula (Hersbach 2000) per component:
        CRPS_i = (2/N^2) * sum_j (X_{(j)} - y) * (j - 0.5*(N-1))    [if y < min]
    More robustly, we evaluate the closed form directly.

    X has shape (n_state, N_ens), x_true has shape (n_state,).
    Returns scalar (mean over components).
    """
    n_state, N = X.shape
    # Term 1: mean_j |X[i,j] - y[i]|, vectorized over i
    t1 = np.mean(np.abs(X - x_true[:, None]), axis=1)
    # Term 2: 0.5 * mean_{j,k} |X[i,j] - X[i,k]|
    # Closed form via sort: 2/N^2 * sum_j (2j - N - 1) * X_{(j)}, j in 1..N
    Xs = np.sort(X, axis=1)
    j = np.arange(1, N + 1)
    weights = (2 * j - N - 1) / (N ** 2)
    t2_full = np.sum(weights * Xs, axis=1)  # this equals mean_{j,k} |X_j - X_k|
    crps_per_component = t1 - 0.5 * t2_full
    return float(np.mean(crps_per_component))


def _ensemble_crps_by_var(X: np.ndarray, x_true: np.ndarray, var_blocks) -> dict:
    """CRPS per variable block, normalized by the block's RMS magnitude.

    Dividing the mean CRPS of a block by ``RMS(x_true[block])`` puts every
    field on a common, unit-free scale so they can be compared and averaged.
    """
    out = {}
    for name, sl in var_blocks.items():
        c = _ensemble_crps(X[sl], x_true[sl])
        scale = float(np.sqrt(np.mean(x_true[sl] ** 2)))
        out[name] = float(c / scale) if scale > 0 else float("nan")
    return out


def _rank_counts(X: np.ndarray, x_true: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-component rank of truth among ensemble members, accumulated as histogram.

    For each component i, the rank is the number of members strictly less
    than x_true[i] (so rank ∈ {0, 1, ..., N}).
    Ties are broken by random assignment within the tied bracket to avoid
    spurious peaks at duplicates (rare with continuous ensembles, common
    with deterministic filters that produce identical members in degenerate
    cases).
    """
    # X: (n_state, N), x_true: (n_state,)
    less = np.sum(X < x_true[:, None], axis=1)
    equal = np.sum(X == x_true[:, None], axis=1)
    # Distribute ties uniformly: add a uniform offset in [0, equal[i]] to
    # `less[i]`. This is the standard tie-breaking rule for rank histograms.
    if np.any(equal > 0):
        rng = np.random.default_rng(0)  # deterministic tie-breaking per call
        less = less + rng.integers(0, equal + 1)
    counts = np.bincount(less.astype(int), minlength=n_bins)
    # Truncate (in case ties pushed past last bin) — np.bincount may produce
    # length > n_bins if integers exceeded N; clamp.
    return counts[:n_bins]


def _rank_counts_by_var(X: np.ndarray, x_true: np.ndarray, n_bins: int, var_blocks) -> dict:
    """One rank histogram per variable block.

    Pooling ranks across fields of different magnitude (as the global
    version does) corrupts the histogram. Restricting to a block keeps each
    field's calibration interpretable on its own.
    """
    return {name: _rank_counts(X[sl], x_true[sl], n_bins)
            for name, sl in var_blocks.items()}


def _resolve_snapshot_steps(
    store_states_at, n_steps: int
) -> np.ndarray:
    """Convert user-provided fractions in [0, 1] into integer step indices.

    Parameters
    ----------
    store_states_at : None, sequence of floats, or numpy array
        Fractions of the simulation at which to store ensemble snapshots.
        Each fraction ``f`` maps to ``round(f * (n_steps - 1))``. ``None``
        disables snapshotting.
    n_steps : int
        Total number of assimilation steps in the run.

    Returns
    -------
    target_steps : ndarray of int, shape (k,)
        Sorted, deduplicated step indices in ``[0, n_steps - 1]``. Returns
        an empty array if ``store_states_at`` is None.

    Raises
    ------
    ValueError
        If any fraction is outside ``[0, 1]``.
    """
    if store_states_at is None:
        return np.empty(0, dtype=int)
    fractions = np.asarray(store_states_at, dtype=float).ravel()
    if fractions.size == 0:
        return np.empty(0, dtype=int)
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError(
            "store_states_at values must lie in [0, 1] "
            f"(received: {fractions.tolist()})."
        )
    if n_steps <= 0:
        return np.empty(0, dtype=int)
    target = np.round(fractions * (n_steps - 1)).astype(int)
    return np.unique(target)


class Simulation:
    """Run a data-assimilation experiment."""

    # ------------------------------------------------------------------
    # Legacy constructor (preserved)
    # ------------------------------------------------------------------
    def __init__(
        self,
        model=None,
        background: Optional[Background] = None,
        analysis=None,
        observation: Optional[Observation] = None,
        params: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
        scenario=None,
        inflation_factor: float = 1.04,
        method_rng: Optional[np.random.Generator] = None,
        store_diagnostics: bool = False,
        store_states_at=None,
        adaptive_inflation=None,
    ):
        # ----- Mode selection -----
        if scenario is not None:
            # Scenario mode
            if analysis is None:
                raise ValueError("scenario mode requires an `analysis`.")
            self._mode = "scenario"
            self.scenario = scenario
            self.model = scenario.model
            self.analysis = analysis
            self.inflation_factor = float(inflation_factor)
            self.method_rng = method_rng if method_rng is not None else np.random.default_rng()
            self.obs_freq = scenario.obs_freq
            self.end_time = scenario.end_time
            self.store_diagnostics = bool(store_diagnostics)
            # Resolve snapshot step indices once we know n_steps from the scenario.
            self._store_states_at = store_states_at
            self.snapshot_steps = _resolve_snapshot_steps(
                store_states_at, scenario.n_steps
            )
        else:
            # Legacy mode
            if any(x is None for x in (model, background, analysis, observation)):
                raise ValueError(
                    "Legacy mode requires `model`, `background`, `analysis`, `observation`."
                )
            self._mode = "legacy"
            self.model = model
            self.background = background
            self.analysis = analysis
            self.observation = observation
            params = params or {"obs_freq": 0.1, "end_time": 15, "inf_fact": 1.04}
            self.obs_freq = params["obs_freq"]
            self.end_time = params["end_time"]
            self.inf_fact = params["inf_fact"]
            self.store_back_state = params.get("store_back_state", False)
            self.store_post_state = params.get("store_post_state", False)
            self.store_ref_state = params.get("store_ref_state", False)
            self.store_state_at = params.get("store_state_at", [])
            self.background_states = [] if self.store_back_state else None
            self.analysis_states = [] if self.store_post_state else None
            self.truth_states = [] if self.store_ref_state else None
            self._stored_indices = []

        # ----- Adaptive inflation (framework-level, filter-agnostic) -----
        # When provided, overrides the fixed inflation_factor each cycle by
        # an innovation-based estimate applied uniformly to every method.
        self.adaptive_inflation = adaptive_inflation

        # ----- Logging -----
        self.logger = logging.getLogger("Simulation")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
        )
        self.logger.handlers = []
        self.logger.addHandler(handler)
        if log_level is not None:
            self.logger.setLevel(log_level)
        else:
            self.logger.disabled = True

    # ------------------------------------------------------------------
    # Scenario constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_scenario(
        cls,
        scenario,
        analysis,
        inflation_factor: float = 1.04,
        method_rng: Optional[np.random.Generator] = None,
        log_level: int = logging.WARNING,
        store_diagnostics: bool = False,
        store_states_at=None,
        adaptive_inflation=None,
    ) -> "Simulation":
        """Create a scenario-based Simulation."""
        return cls(
            scenario=scenario,
            analysis=analysis,
            inflation_factor=inflation_factor,
            method_rng=method_rng,
            log_level=log_level,
            store_diagnostics=store_diagnostics,
            store_states_at=store_states_at,
            adaptive_inflation=adaptive_inflation,
        )

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    def relative_error(self, xr: np.ndarray, xs: np.ndarray) -> float:
        return _rmse_relative(xr, xs)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self):
        if self._mode == "scenario":
            return self._run_scenario()
        return self._run_legacy()

    # ------------------------ Legacy bucle (unchanged behavior) -------
    def _run_legacy(self):
        self.error_a = []
        self.error_b = []

        xtk = self.model.get_initial_condition()
        Xbk = self.background.get_initial_ensemble()
        T = np.linspace(0, self.obs_freq, num=2)
        t = 0

        while t <= self.end_time:
            self.logger.info(f"Time step {t} - {self.end_time}")
            self.observation.generate_observation(xtk)
            Xak = self.analysis.perform_assimilation(self.background, self.observation)

            if self.inf_fact > 0:
                self.analysis.inflate_ensemble(self.inf_fact)
                self.logger.debug(f"Inflated ensemble with factor {self.inf_fact}")

            xak = self.analysis.get_analysis_state()
            xbk = self.background.get_background_state()
            self.error_a.append(self.relative_error(xtk, xak))
            self.error_b.append(self.relative_error(xtk, xbk))

            self.logger.debug(
                f"Background error: {self.error_b[-1]:.4f}, Analysis error: {self.error_a[-1]:.4f}"
            )

            if t in self.store_state_at:
                if self.store_back_state:
                    self.background_states.append(xbk.copy())
                if self.store_post_state:
                    self.analysis_states.append(xak.copy())
                if self.store_ref_state:
                    self.truth_states.append(xtk.copy())
                self._stored_indices.append(t)

            Xbk = self.background.forecast_step(Xak, T)
            xtk = self.model.propagate(xtk, T)
            t += self.obs_freq

        self.error_a = np.array(self.error_a)
        self.error_b = np.array(self.error_b)
        self.logger.info("Simulation completed.")

    # ------------------------ Scenario bucle --------------------------
    def _run_scenario(self):
        """Run a filter on a frozen scenario."""
        scen = self.scenario
        # Seed numpy's global RNG used by some filters (np.random.multivariate_normal)
        # using a value drawn from method_rng. This makes the run reproducible
        # given method_rng.
        seed_for_global = int(self.method_rng.integers(0, 2**31 - 1))
        np.random.seed(seed_for_global)

        # Build a Background object that wraps the frozen initial ensemble.
        bg = Background(scen.model, ensemble_size=scen.ensemble_size)
        bg.Xb = scen.initial_ensemble.copy()
        bg.Xb0 = bg.Xb

        T_step = np.linspace(0.0, scen.obs_freq, num=2)
        K = scen.n_steps
        N = scen.ensemble_size
        n_state = scen.n_state

        self.error_b = np.empty(K)
        self.error_a = np.empty(K)

        # Per-variable metric containers. var_blocks is {name: slice}; for a
        # single-variable model this is one block and mirrors the global value.
        vb = self.model.var_blocks
        self.var_names = list(vb.keys())
        self.error_b_by_var = {name: np.empty(K) for name in vb}
        self.error_a_by_var = {name: np.empty(K) for name in vb}

        if self.store_diagnostics:
            # Per-step, per-component spread (root mean of variance over members).
            self.spread_b = np.empty(K)
            self.spread_a = np.empty(K)
            # Per-step CRPS averaged over state components.
            self.crps_b = np.empty(K)
            self.crps_a = np.empty(K)
            # Rank histogram counts: bin index = number of members below truth.
            # Range is 0..N (inclusive on both sides).
            self.rank_counts_b = np.zeros(N + 1, dtype=np.int64)
            self.rank_counts_a = np.zeros(N + 1, dtype=np.int64)
            # Per-variable diagnostics.
            self.spread_b_by_var = {name: np.empty(K) for name in vb}
            self.spread_a_by_var = {name: np.empty(K) for name in vb}
            self.crps_b_by_var = {name: np.empty(K) for name in vb}
            self.crps_a_by_var = {name: np.empty(K) for name in vb}
            self.rank_counts_b_by_var = {name: np.zeros(N + 1, dtype=np.int64) for name in vb}
            self.rank_counts_a_by_var = {name: np.zeros(N + 1, dtype=np.int64) for name in vb}

        # Snapshot machinery: pre-allocate stacked arrays once we know which
        # steps will be captured.
        snap_steps = self.snapshot_steps
        n_snap = int(snap_steps.size)
        if n_snap > 0:
            self.Xb_snapshots = np.empty((n_snap, n_state, N), dtype=float)
            self.Xa_snapshots = np.empty((n_snap, n_state, N), dtype=float)
            self.snapshot_times = np.asarray(scen.times)[snap_steps].copy()
            self.snapshot_fractions = (
                snap_steps.astype(float) / max(K - 1, 1)
            )
            # Build a fast lookup: step -> index into the snapshot arrays.
            self._snap_index = {int(s): i for i, s in enumerate(snap_steps)}
        else:
            self.Xb_snapshots = np.empty((0, n_state, N), dtype=float)
            self.Xa_snapshots = np.empty((0, n_state, N), dtype=float)
            self.snapshot_times = np.empty(0, dtype=float)
            self.snapshot_fractions = np.empty(0, dtype=float)
            self._snap_index = {}

        for k in range(K):
            x_true = scen.truth_trajectory[k]
            y_k = scen.observations[k]
            op_k = scen.operators[k]

            obs_k = Observation.from_arrays(y=y_k, operator=op_k, noise=scen.noise)

            # Background mean and ensemble before assimilation
            xbk = bg.get_background_state()
            self.error_b[k] = _rmse_relative(x_true, xbk)
            _eb = _rmse_relative_by_var(x_true, xbk, vb)
            for name in vb:
                self.error_b_by_var[name][k] = _eb[name]

            # We may need the full background ensemble for diagnostics, for
            # snapshots, or for both — fetch it at most once per step.
            need_Xb = (self.store_diagnostics or (k in self._snap_index)
                       or self.adaptive_inflation is not None)
            if need_Xb:
                Xb = bg.get_ensemble()
                if self.store_diagnostics:
                    self.spread_b[k] = _ensemble_spread(Xb)
                    self.crps_b[k] = _ensemble_crps(Xb, x_true)
                    self.rank_counts_b += _rank_counts(Xb, x_true, n_bins=N + 1)
                    _sb = _ensemble_spread_by_var(Xb, x_true, vb)
                    _cb = _ensemble_crps_by_var(Xb, x_true, vb)
                    _rb = _rank_counts_by_var(Xb, x_true, N + 1, vb)
                    for name in vb:
                        self.spread_b_by_var[name][k] = _sb[name]
                        self.crps_b_by_var[name][k] = _cb[name]
                        self.rank_counts_b_by_var[name] += _rb[name]
                if k in self._snap_index:
                    self.Xb_snapshots[self._snap_index[k]] = Xb

            # Adaptive inflation: derive this cycle's factor from the
            # innovation BEFORE assimilation, applied identically to every
            # method. Falls back to the fixed factor on any failure.
            if self.adaptive_inflation is not None:
                try:
                    H = obs_k.linearize(xbk) if hasattr(obs_k, "linearize") \
                        else obs_k.get_observation_operator()
                    if hasattr(obs_k, "noise") and hasattr(obs_k.noise, "R_diag"):
                        trR = float(np.sum(obs_k.noise.R_diag))
                    else:
                        trR = float(np.trace(obs_k.get_data_error_covariance()))
                    inf_k = self.adaptive_inflation.update(Xb, H, y_k, trR)
                except Exception:
                    inf_k = self.inflation_factor
            else:
                inf_k = self.inflation_factor

            self.analysis.perform_assimilation(bg, obs_k)
            if inf_k > 0:
                self.analysis.inflate_ensemble(inf_k)

            xak = self.analysis.get_analysis_state()
            self.error_a[k] = _rmse_relative(x_true, xak)
            _ea = _rmse_relative_by_var(x_true, xak, vb)
            for name in vb:
                self.error_a_by_var[name][k] = _ea[name]

            need_Xa = self.store_diagnostics or (k in self._snap_index)
            if need_Xa:
                Xa = self.analysis.get_ensemble()
                if self.store_diagnostics:
                    self.spread_a[k] = _ensemble_spread(Xa)
                    self.crps_a[k] = _ensemble_crps(Xa, x_true)
                    self.rank_counts_a += _rank_counts(Xa, x_true, n_bins=N + 1)
                    _sa = _ensemble_spread_by_var(Xa, x_true, vb)
                    _ca = _ensemble_crps_by_var(Xa, x_true, vb)
                    _ra = _rank_counts_by_var(Xa, x_true, N + 1, vb)
                    for name in vb:
                        self.spread_a_by_var[name][k] = _sa[name]
                        self.crps_a_by_var[name][k] = _ca[name]
                        self.rank_counts_a_by_var[name] += _ra[name]
                if k in self._snap_index:
                    self.Xa_snapshots[self._snap_index[k]] = Xa

            # Forecast next ensemble (skip on last step)
            if k < K - 1:
                Xa = self.analysis.get_ensemble()
                bg.forecast_step(Xa, T_step)

    def get_errors(self):
        return self.error_b, self.error_a

    def get_saved_states(self):
        if self._mode != "legacy":
            raise RuntimeError("get_saved_states is only available in legacy mode.")
        return {
            "background": self.background_states,
            "analysis": self.analysis_states,
            "truth": self.truth_states,
            "steps": self._stored_indices,
        }