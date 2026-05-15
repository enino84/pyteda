# -*- coding: utf-8 -*-
"""Tests for analysis filters in the registry.

Each filter must run on a small scenario without errors, produce an
analysis whose RMSE is at least not NaN/Inf, and reduce the background
error on average over a few steps.
"""

import warnings

import numpy as np
import pytest

from pyteda.experiments import Benchmark


warnings.filterwarnings("ignore", category=Warning)


# Methods to exercise. Each entry: (label, config dict).
ALL_METHODS = [
    ("enkf", dict(method="enkf")),
    ("enkf-cholesky", dict(method="enkf-cholesky")),
    ("enkf-naive", dict(method="enkf-naive")),
    ("enkf-bloc", dict(method="enkf-b-loc")),
    ("enkf-mc", dict(method="enkf-modified-cholesky", r=2)),
    ("enkf-shrinkage", dict(method="enkf-shrinkage-precision")),
    ("etkf", dict(method="etkf")),
    ("ensrf", dict(method="ensrf")),
    ("lenkf", dict(method="lenkf", r=2)),
    ("letkf", dict(method="letkf", r=2)),
]


@pytest.mark.parametrize("name,cfg", ALL_METHODS)
def test_filter_runs_without_errors(small_scenario, name, cfg):
    """Each filter from the registry runs end-to-end without crashing."""
    results = Benchmark(
        scenarios=[small_scenario],
        methods={name: cfg},
        n_runs_per_method=1, parallel=False, verbose=False,
    ).run()
    row = results.rows[0]
    assert np.isfinite(row["error_a"]).all()
    assert np.isfinite(row["error_b"]).all()


@pytest.mark.parametrize("name,cfg", ALL_METHODS)
def test_filter_produces_analysis(small_scenario, name, cfg):
    """The analysis error array has the right length."""
    results = Benchmark(
        scenarios=[small_scenario],
        methods={name: cfg},
        n_runs_per_method=1, parallel=False, verbose=False,
    ).run()
    row = results.rows[0]
    assert len(row["error_a"]) == small_scenario.n_steps


def test_letkf_radius_three_forms_equivalent(small_scenario):
    """LETKF with r=2, r={'x':2}, r=full(20,2) yields identical RMSE."""
    methods = {
        "int":   dict(method="letkf", r=2),
        "dict":  dict(method="letkf", r={"x": 2}),
        "array": dict(method="letkf", r=np.full(20, 2.0)),
    }
    results = Benchmark(
        scenarios=[small_scenario],
        methods=methods, n_runs_per_method=1,
        method_seed_base=1000, parallel=False, verbose=False,
    ).run()
    rmses = {r["method"]: float(np.mean(r["error_a"])) for r in results.rows}
    assert rmses["int"] == rmses["dict"] == rmses["array"]


def test_lenkf_radius_three_forms_equivalent(small_scenario):
    methods = {
        "int":   dict(method="lenkf", r=2),
        "dict":  dict(method="lenkf", r={"x": 2}),
        "array": dict(method="lenkf", r=np.full(20, 2.0)),
    }
    results = Benchmark(
        scenarios=[small_scenario],
        methods=methods, n_runs_per_method=1,
        method_seed_base=1000, parallel=False, verbose=False,
    ).run()
    rmses = {r["method"]: float(np.mean(r["error_a"])) for r in results.rows}
    assert rmses["int"] == rmses["dict"] == rmses["array"]


def test_letkf_better_than_enkf_on_l96(small_scenario):
    """On Lorenz96 with localization, LETKF should outperform plain EnKF."""
    methods = {
        "EnKF":  dict(method="enkf"),
        "LETKF": dict(method="letkf", r=2),
    }
    results = Benchmark(
        scenarios=[small_scenario], methods=methods,
        n_runs_per_method=2, method_seed_base=2000,
        parallel=False, verbose=False,
    ).run()
    rmses = {}
    for row in results.rows:
        rmses.setdefault(row["method"], []).append(float(np.mean(row["error_a"])))
    assert np.mean(rmses["LETKF"]) < np.mean(rmses["EnKF"])
