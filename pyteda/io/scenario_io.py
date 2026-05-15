# -*- coding: utf-8 -*-
"""Save and load a complete `Scenario` to disk.

Format is selected by extension:
  * .nc  → netCDF (default; portable, self-describing).
  * .npz → numpy compressed (legacy fallback).

The dynamical model is NOT serialized — provide it on load. Nonlinear
operators cannot be serialized either, since they would require pickling
arbitrary user code. For nonlinear operators, only their Jacobian at
each step would be storable; we deliberately do not do this and raise.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np

from ._common import detect_format
from ..experiments.scenario import (
    Scenario,
    _operator_to_dict,
    _operator_from_dict,
    _noise_to_dict,
    _noise_from_dict,
)
from ..observation import (
    LinearSelection,
    LinearMatrix,
    IsotropicDiagonal,
    HeterogeneousDiagonal,
    DenseCovariance,
)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def save_scenario(scenario: Scenario, path: str) -> None:
    """Save a Scenario to disk. See module docstring for details."""
    fmt = detect_format(path)
    if fmt == "netcdf":
        _save_scenario_nc(scenario, path)
    else:
        _save_scenario_npz(scenario, path)


def load_scenario(path: str, model) -> Scenario:
    """Load a Scenario from disk.

    Parameters
    ----------
    path : str
        Path to a .nc or .npz file produced by ``save_scenario``.
    model : Model
        The dynamical model — not part of the payload.
    """
    fmt = detect_format(path)
    if fmt == "netcdf":
        return _load_scenario_nc(path, model)
    return _load_scenario_npz(path, model)


# ----------------------------------------------------------------------
# netCDF backend
# ----------------------------------------------------------------------
def _save_scenario_nc(scenario: Scenario, path: str) -> None:
    import xarray as xr

    truth_arr = np.stack(scenario.truth_trajectory, axis=0)  # (T, n)
    obs_arr = np.stack(scenario.observations, axis=0)         # (T, m)

    # Operators: detect fixed schedule (same object every step).
    same_op = all(op is scenario.operators[0] for op in scenario.operators)
    base_op = scenario.operators[0]

    data_vars = {
        "truth": (("time", "state"), truth_arr),
        "observations": (("time", "obs"), obs_arr),
        "initial_ensemble": (("state", "member"), scenario.initial_ensemble),
    }
    coords = {
        "time": scenario.times,
        "state": np.arange(truth_arr.shape[1]),
        "obs": np.arange(obs_arr.shape[1]),
        "member": np.arange(scenario.initial_ensemble.shape[1]),
    }

    if same_op and isinstance(base_op, LinearSelection):
        data_vars["obs_indices"] = (("obs",), base_op.indices.astype(np.int64))
        operator_repr = "LinearSelection"
    elif same_op and isinstance(base_op, LinearMatrix):
        data_vars["H"] = (("obs", "state"), base_op.linearize(None))
        operator_repr = "LinearMatrix"
    elif same_op:
        raise NotImplementedError(
            f"Cannot save scenario with operator type {type(base_op).__name__} "
            f"(nonlinear or unsupported)."
        )
    else:
        # Variable schedule: store one H per step.
        Hs = np.stack([op.linearize(None) for op in scenario.operators], axis=0)
        data_vars["H_per_step"] = (("time", "obs", "state"), Hs)
        operator_repr = "VariableLinear"

    # Noise
    noise = scenario.noise
    if isinstance(noise, IsotropicDiagonal):
        noise_attrs = {"noise_kind": "IsotropicDiagonal", "std": float(noise._std)}
    elif isinstance(noise, HeterogeneousDiagonal):
        data_vars["R_diag"] = (("obs",), np.diag(noise.R))
        noise_attrs = {"noise_kind": "HeterogeneousDiagonal"}
    elif isinstance(noise, DenseCovariance):
        data_vars["R"] = (("obs", "obs2"), noise.R)
        coords["obs2"] = np.arange(noise.R.shape[1])
        noise_attrs = {"noise_kind": "DenseCovariance"}
    else:
        raise NotImplementedError(f"Cannot save noise type {type(noise).__name__}")

    attrs = {
        "obs_freq": float(scenario.obs_freq),
        "end_time": float(scenario.end_time),
        "operator_repr": operator_repr,
        "scenario_meta_json": json.dumps(scenario.meta),
        **noise_attrs,
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path)


def _load_scenario_nc(path: str, model) -> Scenario:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        truth_arr = ds["truth"].values
        obs_arr = ds["observations"].values
        Xb0 = ds["initial_ensemble"].values
        times = ds["time"].values
        attrs = dict(ds.attrs)

        n_steps = truth_arr.shape[0]
        n_state = truth_arr.shape[1]
        m = obs_arr.shape[1]

        # Operator(s) ---------------------------------------------------
        op_repr = attrs.get("operator_repr", "LinearSelection")
        if op_repr == "LinearSelection":
            indices = ds["obs_indices"].values
            op = LinearSelection(m=m, n_state=n_state, indices=indices)
            operators = [op] * n_steps
        elif op_repr == "LinearMatrix":
            H = ds["H"].values
            op = LinearMatrix(H)
            operators = [op] * n_steps
        elif op_repr == "VariableLinear":
            Hs = ds["H_per_step"].values
            operators = [LinearMatrix(Hs[k]) for k in range(n_steps)]
        else:
            raise NotImplementedError(f"Unknown operator_repr: {op_repr}")

        # Noise ---------------------------------------------------------
        nk = attrs["noise_kind"]
        if nk == "IsotropicDiagonal":
            noise = IsotropicDiagonal(std=float(attrs["std"]), dim=m)
        elif nk == "HeterogeneousDiagonal":
            stds = np.sqrt(ds["R_diag"].values)
            noise = HeterogeneousDiagonal(stds=stds)
        elif nk == "DenseCovariance":
            R = ds["R"].values
            noise = DenseCovariance(R=R)
        else:
            raise NotImplementedError(f"Unknown noise_kind: {nk}")

        meta = json.loads(attrs.get("scenario_meta_json", "{}"))
        obs_freq = float(attrs["obs_freq"])
        end_time = float(attrs["end_time"])

    truth = [truth_arr[k].copy() for k in range(n_steps)]
    obs_list = [obs_arr[k].copy() for k in range(n_steps)]

    return Scenario(
        truth_trajectory=truth,
        observations=obs_list,
        operators=operators,
        noise=noise,
        initial_ensemble=Xb0.copy(),
        times=times.copy(),
        model=model,
        obs_freq=obs_freq,
        end_time=end_time,
        meta=meta,
    )


# ----------------------------------------------------------------------
# npz backend (legacy)
# ----------------------------------------------------------------------
def _save_scenario_npz(scenario: Scenario, path: str) -> None:
    same_op = all(op is scenario.operators[0] for op in scenario.operators)
    if same_op:
        ops_meta = json.dumps({
            "same_op": True,
            "op": _operator_to_dict(scenario.operators[0]),
        })
    else:
        ops_meta = json.dumps({
            "same_op": False,
            "ops": [_operator_to_dict(op) for op in scenario.operators],
        })

    np.savez_compressed(
        path,
        truth=np.stack(scenario.truth_trajectory, axis=0),
        obs=np.stack(scenario.observations, axis=0),
        initial_ensemble=scenario.initial_ensemble,
        times=scenario.times,
        ops_meta=np.array(ops_meta),
        noise_meta=np.array(json.dumps(_noise_to_dict(scenario.noise))),
        scenario_meta=np.array(json.dumps(scenario.meta)),
        obs_freq=np.array(scenario.obs_freq),
        end_time=np.array(scenario.end_time),
    )


def _load_scenario_npz(path: str, model) -> Scenario:
    data = np.load(path, allow_pickle=False)
    truth = list(data["truth"])
    obs = list(data["obs"])
    ops_meta = json.loads(str(data["ops_meta"]))
    noise = _noise_from_dict(json.loads(str(data["noise_meta"])))
    meta = json.loads(str(data["scenario_meta"]))
    if ops_meta["same_op"]:
        op = _operator_from_dict(ops_meta["op"])
        operators = [op] * len(truth)
    else:
        operators = [_operator_from_dict(d) for d in ops_meta["ops"]]
    return Scenario(
        truth_trajectory=truth,
        observations=obs,
        operators=operators,
        noise=noise,
        initial_ensemble=np.array(data["initial_ensemble"]),
        times=np.array(data["times"]),
        model=model,
        obs_freq=float(data["obs_freq"]),
        end_time=float(data["end_time"]),
        meta=meta,
    )
