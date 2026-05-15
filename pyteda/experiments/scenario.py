# -*- coding: utf-8 -*-
"""
Scenario: a frozen twin-experiment setup.

A `Scenario` bundles everything that, once chosen, defines a reproducible
twin experiment. It is generated once per random seed and then reused
across every method in a benchmark — guaranteeing that two filters
compared on the same scenario see exactly the same truth, the same
observation network, and start from the same initial ensemble.

Contents (all frozen at construction time):

* `truth_trajectory` : list of K+1 ndarrays, the true state at each
                       observation step.
* `observations`     : list of K+1 ndarrays, the sampled observations.
* `operators`        : list of K+1 ObservationOperator objects (often the
                       same object repeated when the network is fixed).
* `noise`            : ObservationNoise object (shared across steps).
* `initial_ensemble` : ndarray of shape (n, N_ens), the common initial
                       background ensemble.
* `times`            : ndarray of observation times, length K+1.
* `model`            : the dynamical model used.
* `meta`             : dict with seed, parameters, and a config hash.

Use `Scenario.generate(...)` to build a new scenario, or `save`/`load`
to (de)serialize one to disk via numpy's npz format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any

import numpy as np

from ..observation import (
    Observation,
    ObservationOperator,
    LinearSelection,
    ObservationNoise,
    IsotropicDiagonal,
    HeterogeneousDiagonal,
    DenseCovariance,
)


# ----------------------------------------------------------------------
# (De)serialization helpers for operators and noise
# ----------------------------------------------------------------------
def _operator_to_dict(op: ObservationOperator) -> Dict[str, Any]:
    """Serialize a (linear) operator to a dict. Nonlinear operators are
    not serializable by design — they would require pickling user code."""
    if isinstance(op, LinearSelection):
        return {
            "kind": "LinearSelection",
            "m": op.dim_obs,
            "n_state": op.dim_state,
            "indices": op.indices.tolist(),
        }
    # Generic linear: store H
    if op.is_linear:
        return {
            "kind": "LinearMatrix",
            "H": op.linearize(None).tolist(),
        }
    raise ValueError(
        "Nonlinear operators cannot be saved to disk (would require pickling code)."
    )


def _operator_from_dict(d: Dict[str, Any]) -> ObservationOperator:
    if d["kind"] == "LinearSelection":
        return LinearSelection(
            m=d["m"], n_state=d["n_state"], indices=np.array(d["indices"])
        )
    if d["kind"] == "LinearMatrix":
        from ..observation import LinearMatrix
        return LinearMatrix(np.array(d["H"]))
    raise ValueError(f"Unknown operator kind: {d['kind']}")


def _noise_to_dict(noise: ObservationNoise) -> Dict[str, Any]:
    if isinstance(noise, IsotropicDiagonal):
        return {"kind": "IsotropicDiagonal", "std": noise._std, "dim": noise.dim}
    if isinstance(noise, HeterogeneousDiagonal):
        return {"kind": "HeterogeneousDiagonal", "stds": noise._stds.tolist()}
    if isinstance(noise, DenseCovariance):
        return {"kind": "DenseCovariance", "R": noise.R.tolist()}
    raise ValueError(f"Cannot serialize noise of type {type(noise).__name__}")


def _noise_from_dict(d: Dict[str, Any]) -> ObservationNoise:
    if d["kind"] == "IsotropicDiagonal":
        return IsotropicDiagonal(std=d["std"], dim=d["dim"])
    if d["kind"] == "HeterogeneousDiagonal":
        return HeterogeneousDiagonal(stds=np.array(d["stds"]))
    if d["kind"] == "DenseCovariance":
        return DenseCovariance(R=np.array(d["R"]))
    raise ValueError(f"Unknown noise kind: {d['kind']}")


# ----------------------------------------------------------------------
# Scenario
# ----------------------------------------------------------------------
@dataclass
class Scenario:
    """A frozen twin-experiment setup."""

    truth_trajectory: List[np.ndarray]
    observations: List[np.ndarray]
    operators: List[ObservationOperator]
    noise: ObservationNoise
    initial_ensemble: np.ndarray
    times: np.ndarray
    model: Any  # Model object — NOT serialized, must be re-attached on load
    obs_freq: float
    end_time: float
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def generate(
        cls,
        model,
        operator_factory=None,
        operator=None,
        noise=None,
        ensemble_size: int = 20,
        obs_freq: float = 0.1,
        end_time: float = 10.0,
        seed: int = 0,
        operator_schedule: str = "fixed",
        # ---- Phase 1: x0_ref construction (one-time, very expensive) ----
        spinup_truth: float = 0.0,
        x0_ref=None,
        # ---- Phase 2: xb construction (separates ensemble centre from truth) ----
        pert_xb: float = 0.01,
        spinup_xb: float = 0.0,
        xb=None,
        # ---- Phase 3: ensemble dispersion around xb ----
        pert_ensemble: float = 0.05,
        spinup_ensemble: float = 0.0,
        initial_ensemble=None,
        # ---- Pre-computed truth / observations (skip generation) ----
        truth_trajectory=None,
        observations=None,
        # ---- Legacy aliases (deprecated, kept for backwards compat) ----
        long_spinup=None,
        short_spinup=None,
        perturbation_amp=None,
        initial_perturbation=None,
        spinup_time=None,
        x_true_init=None,
        x_ref=None,
    ):
        """Generate a Scenario with a physically-grounded ensemble.

        The ensemble construction follows a three-phase recipe that
        produces a realistic ``X_b`` whose centre is offset from the truth
        (as in real-world DA, where the background carries accumulated
        forecast error).

        Phase 1 — Reference state ``x0_ref``
            Start from a synthetic IC (``model.get_initial_condition()``)
            and propagate by ``spinup_truth`` to land on the model
            attractor. Skip with ``x0_ref=...``.

        Phase 2 — Ensemble centre ``xb``
            Perturb ``x0_ref`` with std ``pert_xb`` and propagate by
            ``spinup_xb``. Chaos amplifies the small initial perturbation
            into a meaningful separation between ``xb`` and the truth
            trajectory. Skip with ``xb=...``.

        Phase 3 — Initial ensemble ``X_b``
            Perturb ``xb`` with std ``pert_ensemble`` (one perturbation
            per ensemble member) and propagate each member by
            ``spinup_ensemble``. This disperses the ensemble around its
            centre without moving the centre much. Skip with
            ``initial_ensemble=...``.

        Truth synchronization
            The truth trajectory starts at ``x0_ref`` and is propagated
            by ``spinup_xb + spinup_ensemble`` so it lands at the same
            model time as the ensemble. From there, the truth is
            propagated step-by-step on the observation grid.

        Parameters
        ----------
        model : Model
            Dynamical model.
        operator, operator_factory : ObservationOperator or callable
            Either a pre-built operator, or a factory that builds one
            from an RNG.
        noise : ObservationNoise, optional
            Noise model. Defaults to ``IsotropicDiagonal(std=0.01)``.
        ensemble_size : int
            Initial ensemble size. Ignored if ``initial_ensemble`` is given.
        obs_freq : float
            Time between consecutive observations.
        end_time : float
            Duration of the truth trajectory after the spinup phases.
        seed : int
            Master seed (split into independent streams).
        operator_schedule : str
            ``'fixed'`` (default) or ``'random'``.
        spinup_truth : float
            Phase 1 propagation time (synthetic IC -> x0_ref).
        x0_ref : ndarray, optional
            Pre-computed reference state. Skips Phase 1.
        pert_xb : float
            Std of the gaussian perturbation that seeds Phase 2.
        spinup_xb : float
            Phase 2 propagation time (x0_ref + pert -> xb).
        xb : ndarray, optional
            Pre-computed ensemble centre. Skips Phase 2.
        pert_ensemble : float
            Std of the gaussian perturbation around xb (Phase 3).
        spinup_ensemble : float
            Phase 3 propagation time (xb + per-member perts -> X_b).
        initial_ensemble : ndarray, optional
            Pre-computed initial ensemble. Skips Phase 3.
        truth_trajectory, observations : optional
            Pre-computed truth/observations.

        Returns
        -------
        Scenario
            Fully-populated frozen experiment. The reference state and
            the ensemble centre are stored in ``meta`` for diagnostics
            (``scenario.x0_ref``, ``scenario.xb``).
        """
        # ---- Backwards-compat aliasing -------------------------------
        if perturbation_amp is not None and pert_ensemble == 0.05:
            pert_ensemble = float(perturbation_amp)
        if initial_perturbation is not None:
            pert_ensemble = float(initial_perturbation)
        if x_true_init is not None and x0_ref is None:
            x0_ref = x_true_init
        if x_ref is not None and x0_ref is None:
            x0_ref = x_ref
        if spinup_time is not None and spinup_truth == 0.0:
            spinup_truth = float(np.asarray(spinup_time)[-1])
        if long_spinup is not None and spinup_truth == 0.0:
            spinup_truth = float(long_spinup)
        if short_spinup is not None and spinup_ensemble == 0.0:
            spinup_ensemble = float(short_spinup)

        # ---- Validation ---------------------------------------------
        if (operator is None) == (operator_factory is None):
            raise ValueError(
                "Provide exactly one of `operator` or `operator_factory`."
            )

        provided_truth = truth_trajectory
        provided_observations = observations
        provided_x0_ref = x0_ref is not None
        provided_xb = xb is not None
        provided_initial_ensemble = initial_ensemble is not None

        # Independent RNG streams from a single master seed -----------
        ss = np.random.SeedSequence(seed)
        rng_op, rng_xb, rng_noise, rng_ens = [
            np.random.default_rng(s) for s in ss.spawn(4)
        ]

        # Operator(s) -------------------------------------------------
        if operator is not None:
            base_op = operator
            n_state = base_op.dim_state
        else:
            base_op = operator_factory(rng_op)
            n_state = base_op.dim_state

        # Noise -------------------------------------------------------
        if noise is None:
            noise = IsotropicDiagonal(std=0.01, dim=base_op.dim_obs)
        elif isinstance(noise, IsotropicDiagonal) and noise._dim is None:
            noise.bind_dim(base_op.dim_obs)
        if noise.dim != base_op.dim_obs:
            raise ValueError(
                f"Noise dim ({noise.dim}) must match operator dim_obs "
                f"({base_op.dim_obs})."
            )

        # Time grid ---------------------------------------------------
        n_steps = int(np.floor(end_time / obs_freq)) + 1
        times = np.arange(n_steps) * obs_freq
        T_step = np.linspace(0.0, obs_freq, num=2)

        # ============================================================
        # PHASE 1 — Reference state x0_ref
        # ============================================================
        if x0_ref is None:
            x_init = model.get_initial_condition()
            if spinup_truth > 0:
                x0_ref = model.propagate(
                    x_init, np.array([0.0, float(spinup_truth)])
                )
            else:
                x0_ref = np.array(x_init, dtype=float, copy=True)
        else:
            x0_ref = np.asarray(x0_ref, dtype=float).copy()
        if x0_ref.size != n_state:
            raise ValueError(
                f"x0_ref has size {x0_ref.size}, but operator expects "
                f"n_state={n_state}."
            )

        # ============================================================
        # PHASE 2 — Ensemble centre xb (offset from x0_ref via chaos)
        # ============================================================
        if xb is None:
            xb_init = x0_ref + pert_xb * rng_xb.standard_normal(n_state)
            if spinup_xb > 0:
                xb = model.propagate(
                    xb_init, np.array([0.0, float(spinup_xb)])
                )
            else:
                xb = xb_init
        else:
            xb = np.asarray(xb, dtype=float).copy()
        if xb.size != n_state:
            raise ValueError(
                f"xb has size {xb.size}, but operator expects n_state={n_state}."
            )

        # ============================================================
        # PHASE 3 — Initial ensemble X_b around xb
        # ============================================================
        if initial_ensemble is not None:
            Xb0 = np.asarray(initial_ensemble, dtype=float).copy()
            if Xb0.shape[0] != n_state:
                raise ValueError(
                    f"initial_ensemble has {Xb0.shape[0]} rows, "
                    f"expected n_state={n_state}."
                )
            ensemble_size = Xb0.shape[1]
        else:
            Xb0 = np.empty((n_state, ensemble_size), dtype=float)
            for e in range(ensemble_size):
                pert = pert_ensemble * rng_ens.standard_normal(n_state)
                x_member_init = xb + pert
                if spinup_ensemble > 0:
                    Xb0[:, e] = model.propagate(
                        x_member_init,
                        np.array([0.0, float(spinup_ensemble)]),
                    )
                else:
                    Xb0[:, e] = x_member_init

        # ============================================================
        # Truth trajectory (synced with the ensemble)
        # The ensemble lives at model time t = spinup_xb + spinup_ensemble
        # relative to x0_ref. We propagate x0_ref by that same amount.
        # ============================================================
        truth_list = []
        if provided_truth is not None:
            arr = np.asarray(provided_truth)
            if arr.ndim == 2:
                if arr.shape[0] != n_steps:
                    raise ValueError(
                        f"truth_trajectory has {arr.shape[0]} steps, "
                        f"expected {n_steps}."
                    )
                truth_list = [arr[k].copy() for k in range(n_steps)]
            else:
                if len(provided_truth) != n_steps:
                    raise ValueError(
                        f"truth_trajectory has {len(provided_truth)} entries, "
                        f"expected {n_steps}."
                    )
                truth_list = [np.asarray(x).copy() for x in provided_truth]
            if truth_list[0].size != n_state:
                raise ValueError(
                    f"truth_trajectory entries have size {truth_list[0].size}, "
                    f"operator expects n_state={n_state}."
                )
        else:
            sync_time = float(spinup_xb + spinup_ensemble)
            if sync_time > 0:
                x = model.propagate(
                    x0_ref, np.array([0.0, sync_time])
                )
            else:
                x = np.array(x0_ref, dtype=float, copy=True)
            for k in range(n_steps):
                truth_list.append(x.copy())
                x = model.propagate(x, T_step)

        # ============================================================
        # Operators per step
        # ============================================================
        operators_list = []
        for k in range(n_steps):
            if operator_schedule == "fixed" or operator is not None:
                operators_list.append(base_op)
            else:
                operators_list.append(operator_factory(rng_op))

        # ============================================================
        # Observations
        # ============================================================
        obs_list = []
        if provided_observations is not None:
            arr = np.asarray(provided_observations)
            if arr.ndim == 2:
                if arr.shape[0] != n_steps:
                    raise ValueError(
                        f"observations has {arr.shape[0]} steps, "
                        f"expected {n_steps}."
                    )
                obs_list = [arr[k].copy() for k in range(n_steps)]
            else:
                if len(provided_observations) != n_steps:
                    raise ValueError(
                        f"observations has {len(provided_observations)} entries, "
                        f"expected {n_steps}."
                    )
                obs_list = [np.asarray(y).copy() for y in provided_observations]
            if obs_list[0].size != base_op.dim_obs:
                raise ValueError(
                    f"observations entries have size {obs_list[0].size}, "
                    f"operator dim_obs={base_op.dim_obs}."
                )
        else:
            for k in range(n_steps):
                op_k = operators_list[k]
                y_k = op_k.apply(truth_list[k]) + noise.sample(rng_noise)
                obs_list.append(y_k)

        # ============================================================
        # Meta + hash
        # ============================================================
        meta = {
            "seed": int(seed),
            "ensemble_size": int(ensemble_size),
            "obs_freq": float(obs_freq),
            "end_time": float(end_time),
            "spinup_truth": float(spinup_truth),
            "spinup_xb": float(spinup_xb),
            "spinup_ensemble": float(spinup_ensemble),
            "pert_xb": float(pert_xb),
            "pert_ensemble": float(pert_ensemble),
            "n_state": int(n_state),
            "dim_obs": int(base_op.dim_obs),
            "operator_schedule": operator_schedule,
            "operator_kind": type(base_op).__name__,
            "noise_kind": type(noise).__name__,
            "model_class": type(model).__name__,
            "preloaded_truth": provided_truth is not None,
            "preloaded_observations": provided_observations is not None,
            "preloaded_x0_ref": provided_x0_ref,
            "preloaded_xb": provided_xb,
            "preloaded_initial_ensemble": provided_initial_ensemble,
        }
        meta["config_hash"] = hashlib.sha1(
            json.dumps(
                {k: v for k, v in meta.items() if k != "config_hash"},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]

        # Reassign final names that __init__ expects
        truth_trajectory = truth_list
        observations = obs_list
        operators = operators_list

        scen = cls(
            truth_trajectory=truth_trajectory,
            observations=observations,
            operators=operators,
            noise=noise,
            initial_ensemble=Xb0,
            times=times,
            model=model,
            obs_freq=obs_freq,
            end_time=end_time,
            meta=meta,
        )
        # Stash x0_ref and xb on the Scenario for diagnostics and IO.
        scen.x0_ref = x0_ref
        scen.xb = xb
        return scen

    # ------------------------------------------------------------------
    def iter_steps(self):
        """Yield (k, t_k, x_true_k, y_k, op_k) for k = 0..K."""
        for k, (t, x, y, op) in enumerate(
            zip(self.times, self.truth_trajectory, self.observations, self.operators)
        ):
            yield k, t, x, y, op

    @property
    def n_steps(self) -> int:
        return len(self.truth_trajectory)

    @property
    def n_state(self) -> int:
        return self.initial_ensemble.shape[0]

    @property
    def ensemble_size(self) -> int:
        return self.initial_ensemble.shape[1]

    @property
    def dim_obs(self) -> int:
        return self.operators[0].dim_obs

    # ------------------------------------------------------------------
    # Persistence — dispatches to pyteda.io by file extension
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save the scenario to disk.

        File format is selected by extension:
          * ``.nc``  → netCDF (default; portable, self-describing).
          * ``.npz`` → numpy compressed archive (legacy fallback).

        The dynamical model is NOT serialized; provide it on load.
        """
        from ..io import save_scenario
        save_scenario(self, path)

    @classmethod
    def load(cls, path: str, model) -> "Scenario":
        """Load a scenario from disk.

        File format is auto-detected from the extension. The dynamical
        ``model`` must be provided — it is not part of the payload.
        """
        from ..io import load_scenario
        return load_scenario(path, model=model)

    def __repr__(self) -> str:
        return (
            f"Scenario(model={type(self.model).__name__}, "
            f"n_state={self.n_state}, ensemble_size={self.ensemble_size}, "
            f"n_steps={self.n_steps}, dim_obs={self.dim_obs}, "
            f"hash={self.meta.get('config_hash', '?')})"
        )
