# -*- coding: utf-8 -*-
"""
Granular IO for reusable artifacts.

These helpers let you save expensive components once and reuse them
across many scenarios:

* ``save_initial_ensemble`` / ``load_initial_ensemble``
    The initial background ensemble X0 depends only on the model, the
    ensemble size, and the spinup. Compute it once, save it, share it.

* ``save_truth_trajectory`` / ``load_truth_trajectory``
    The truth trajectory depends only on the model and the truth seed.
    Save it if you plan to reuse the same truth across many scenarios
    that vary only in the observation network or noise.

* ``save_observations`` / ``load_observations``
    A sequence of observations and their associated time stamps. Useful
    for "feed identical y[k] sequences to many filters" experiments.

The default format is netCDF (``.nc``); ``.npz`` is supported as a
fallback that does not require netCDF tooling on the reader's side.
Format is chosen automatically from the file extension.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ._common import detect_format


# ----------------------------------------------------------------------
# Initial ensemble
# ----------------------------------------------------------------------
def save_initial_ensemble(
    X0: np.ndarray,
    path: str,
    meta: Optional[Dict] = None,
) -> None:
    """Save an initial ensemble matrix of shape (n_state, ensemble_size).

    Parameters
    ----------
    X0 : ndarray
        Ensemble matrix (n_state, ensemble_size).
    path : str
        Destination. ``.nc`` writes netCDF; ``.npz`` writes numpy.
    meta : dict, optional
        Free-form metadata stored as global attributes (netCDF) or as a
        JSON-encoded entry (npz). Only str/int/float values are reliable
        across formats.
    """
    X0 = np.asarray(X0)
    if X0.ndim != 2:
        raise ValueError(f"X0 must be 2D, got shape {X0.shape}.")
    fmt = detect_format(path)
    meta = dict(meta) if meta else {}

    if fmt == "netcdf":
        import xarray as xr
        ds = xr.Dataset(
            data_vars={"initial_ensemble": (("state", "member"), X0)},
            coords={
                "state": np.arange(X0.shape[0]),
                "member": np.arange(X0.shape[1]),
            },
            attrs={"kind": "initial_ensemble", **{k: str(v) for k, v in meta.items()}},
        )
        ds.to_netcdf(path)
    else:
        import json
        np.savez_compressed(
            path,
            initial_ensemble=X0,
            meta=np.array(json.dumps({"kind": "initial_ensemble", **meta})),
        )


def load_initial_ensemble(path: str) -> np.ndarray:
    """Load an initial ensemble previously saved with `save_initial_ensemble`."""
    fmt = detect_format(path)
    if fmt == "netcdf":
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return ds["initial_ensemble"].values
    else:
        data = np.load(path, allow_pickle=False)
        return np.array(data["initial_ensemble"])


# ----------------------------------------------------------------------
# State vector (used for x0_ref and xb — any single state of shape (n,))
# ----------------------------------------------------------------------
def save_state_vector(
    x: np.ndarray,
    path: str,
    name: str = "state",
    meta: Optional[Dict] = None,
) -> None:
    """Save a single state vector of shape ``(n_state,)``.

    Used to persist ``x0_ref`` (the reference state on the attractor)
    and ``xb`` (the ensemble centre) — both are length-``n_state`` arrays
    that are expensive to compute once and cheap to reload.

    Parameters
    ----------
    x : ndarray
        State vector, shape ``(n_state,)``.
    path : str
        Destination file (.nc or .npz).
    name : str
        Logical name written to the file (e.g. ``'x0_ref'``, ``'xb'``).
        Useful for documentation; not used to restrict load.
    meta : dict, optional
        Free-form metadata.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"State vector must be 1D, got shape {x.shape}.")
    fmt = detect_format(path)
    meta = dict(meta) if meta else {}

    if fmt == "netcdf":
        import xarray as xr
        ds = xr.Dataset(
            data_vars={"state": (("component",), x)},
            coords={"component": np.arange(x.size)},
            attrs={"kind": name,
                   **{k: str(v) for k, v in meta.items()}},
        )
        ds.to_netcdf(path)
    else:
        import json
        np.savez_compressed(
            path,
            state=x,
            meta=np.array(json.dumps({"kind": name, **meta})),
        )


def load_state_vector(path: str) -> np.ndarray:
    """Load a state vector previously saved with ``save_state_vector``."""
    fmt = detect_format(path)
    if fmt == "netcdf":
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return ds["state"].values
    else:
        data = np.load(path, allow_pickle=False)
        return np.array(data["state"])


# ----------------------------------------------------------------------
# Truth trajectory
# ----------------------------------------------------------------------
def save_truth_trajectory(
    truth: Union[List[np.ndarray], np.ndarray],
    times: np.ndarray,
    path: str,
    meta: Optional[Dict] = None,
) -> None:
    """Save a truth trajectory of length n_steps with associated times.

    Parameters
    ----------
    truth : list[ndarray] or ndarray
        Either a list of (n_state,) arrays, or a stacked (n_steps, n_state) array.
    times : ndarray
        Time stamps (length n_steps).
    path : str
        Destination (.nc or .npz).
    meta : dict, optional
        Free-form metadata.
    """
    arr = np.asarray(truth) if not isinstance(truth, list) else np.stack(truth, axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Truth must be 2D (n_steps, n_state), got {arr.shape}.")
    times = np.asarray(times)
    if times.shape[0] != arr.shape[0]:
        raise ValueError(
            f"len(times)={times.shape[0]} does not match truth steps={arr.shape[0]}."
        )
    fmt = detect_format(path)
    meta = dict(meta) if meta else {}

    if fmt == "netcdf":
        import xarray as xr
        ds = xr.Dataset(
            data_vars={"truth": (("time", "state"), arr)},
            coords={"time": times, "state": np.arange(arr.shape[1])},
            attrs={"kind": "truth_trajectory", **{k: str(v) for k, v in meta.items()}},
        )
        ds.to_netcdf(path)
    else:
        import json
        np.savez_compressed(
            path,
            truth=arr,
            times=times,
            meta=np.array(json.dumps({"kind": "truth_trajectory", **meta})),
        )


def load_truth_trajectory(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(truth, times)`` previously saved with `save_truth_trajectory`.

    `truth` has shape (n_steps, n_state); `times` has shape (n_steps,).
    """
    fmt = detect_format(path)
    if fmt == "netcdf":
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return ds["truth"].values, ds["time"].values
    else:
        data = np.load(path, allow_pickle=False)
        return np.array(data["truth"]), np.array(data["times"])


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------
def save_observations(
    observations: Union[List[np.ndarray], np.ndarray],
    times: np.ndarray,
    path: str,
    meta: Optional[Dict] = None,
) -> None:
    """Save a sequence of observations of shape (n_steps, dim_obs)."""
    arr = (np.asarray(observations) if not isinstance(observations, list)
           else np.stack(observations, axis=0))
    if arr.ndim != 2:
        raise ValueError(f"Observations must be 2D (n_steps, dim_obs), got {arr.shape}.")
    times = np.asarray(times)
    if times.shape[0] != arr.shape[0]:
        raise ValueError(
            f"len(times)={times.shape[0]} does not match obs steps={arr.shape[0]}."
        )
    fmt = detect_format(path)
    meta = dict(meta) if meta else {}

    if fmt == "netcdf":
        import xarray as xr
        ds = xr.Dataset(
            data_vars={"observations": (("time", "obs"), arr)},
            coords={"time": times, "obs": np.arange(arr.shape[1])},
            attrs={"kind": "observations", **{k: str(v) for k, v in meta.items()}},
        )
        ds.to_netcdf(path)
    else:
        import json
        np.savez_compressed(
            path,
            observations=arr,
            times=times,
            meta=np.array(json.dumps({"kind": "observations", **meta})),
        )


def load_observations(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(observations, times)``; obs shape is (n_steps, dim_obs)."""
    fmt = detect_format(path)
    if fmt == "netcdf":
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return ds["observations"].values, ds["time"].values
    else:
        data = np.load(path, allow_pickle=False)
        return np.array(data["observations"]), np.array(data["times"])
