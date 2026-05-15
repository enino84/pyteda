# -*- coding: utf-8 -*-
"""
IO submodule for TEDA.

Default format is netCDF (``.nc``); ``.npz`` is supported as a fallback.
The format is auto-detected from the file extension.

Granular helpers — save reusable artifacts independently:
    save_initial_ensemble / load_initial_ensemble
    save_truth_trajectory / load_truth_trajectory
    save_observations     / load_observations

Full Scenario IO:
    save_scenario / load_scenario

Tip: the initial ensemble depends only on (model, ensemble_size, spinup),
not on the observation network or the noise. Compute it once with
`Scenario.generate(...).initial_ensemble`, save it with
`save_initial_ensemble(...)`, and reuse it across many scenarios via
the `initial_ensemble=` argument of `Scenario.generate`.
"""

from .arrays import (
    save_initial_ensemble,
    load_initial_ensemble,
    save_state_vector,
    load_state_vector,
    save_truth_trajectory,
    load_truth_trajectory,
    save_observations,
    load_observations,
)
from .scenario_io import save_scenario, load_scenario
from ._paths import get_data_dir

__all__ = [
    "save_initial_ensemble",
    "load_initial_ensemble",
    "save_state_vector",
    "load_state_vector",
    "save_truth_trajectory",
    "load_truth_trajectory",
    "save_observations",
    "load_observations",
    "save_scenario",
    "load_scenario",
    "get_data_dir",
]
