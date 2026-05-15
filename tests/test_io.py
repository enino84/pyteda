# -*- coding: utf-8 -*-
"""Tests for pyteda.io — netCDF and npz roundtrips."""

import os

import numpy as np
import pytest

from pyteda.io import (
    save_initial_ensemble, load_initial_ensemble,
    save_state_vector, load_state_vector,
    save_truth_trajectory, load_truth_trajectory,
    save_observations, load_observations,
    get_data_dir,
)


# ----------------------------------------------------------------------
# get_data_dir
# ----------------------------------------------------------------------
class TestGetDataDir:
    def test_default_creates_scenarios(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = get_data_dir()
        assert d.exists()
        assert d.name == "scenarios"

    def test_explicit_path(self, tmp_path):
        target = tmp_path / "custom_dir"
        d = get_data_dir(target)
        assert d.exists()
        assert d == target

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        # Use tmp_path as fake home so we don't pollute the real home dir
        monkeypatch.setenv("HOME", str(tmp_path))
        d = get_data_dir("~/teda_test_dir")
        assert d.exists()
        # Cleanup
        d.rmdir()


# ----------------------------------------------------------------------
# save/load_state_vector (used for x0_ref and xb)
# ----------------------------------------------------------------------
class TestStateVectorIO:
    def test_netcdf_roundtrip(self, tmp_path):
        x = np.random.default_rng(0).standard_normal(40)
        path = tmp_path / "x.nc"
        save_state_vector(x, str(path), name="x0_ref")
        x_back = load_state_vector(str(path))
        assert np.allclose(x, x_back)

    def test_npz_roundtrip(self, tmp_path):
        x = np.random.default_rng(0).standard_normal(40)
        path = tmp_path / "x.npz"
        save_state_vector(x, str(path), name="xb")
        x_back = load_state_vector(str(path))
        assert np.allclose(x, x_back)

    def test_2d_array_raises(self, tmp_path):
        with pytest.raises(ValueError, match="1D"):
            save_state_vector(np.zeros((4, 5)), str(tmp_path / "x.nc"))


# ----------------------------------------------------------------------
# save/load_initial_ensemble
# ----------------------------------------------------------------------
class TestInitialEnsembleIO:
    def test_netcdf_roundtrip(self, tmp_path):
        X = np.random.default_rng(0).standard_normal((40, 10))
        path = tmp_path / "X0.nc"
        save_initial_ensemble(X, str(path))
        X_back = load_initial_ensemble(str(path))
        assert X_back.shape == (40, 10)
        assert np.allclose(X, X_back)

    def test_npz_roundtrip(self, tmp_path):
        X = np.random.default_rng(0).standard_normal((40, 10))
        path = tmp_path / "X0.npz"
        save_initial_ensemble(X, str(path))
        X_back = load_initial_ensemble(str(path))
        assert np.allclose(X, X_back)

    def test_format_detected_by_extension(self, tmp_path):
        X = np.random.default_rng(0).standard_normal((20, 5))
        # netcdf
        save_initial_ensemble(X, str(tmp_path / "a.nc"))
        # npz
        save_initial_ensemble(X, str(tmp_path / "b.npz"))
        # Both files exist
        assert (tmp_path / "a.nc").exists()
        assert (tmp_path / "b.npz").exists()


# ----------------------------------------------------------------------
# save/load_truth_trajectory
# ----------------------------------------------------------------------
class TestTruthIO:
    def test_netcdf_roundtrip(self, tmp_path):
        truth = [np.random.default_rng(k).standard_normal(20) for k in range(5)]
        times = np.linspace(0, 1, 5)
        path = tmp_path / "truth.nc"
        save_truth_trajectory(truth, times, str(path))
        truth_back, times_back = load_truth_trajectory(str(path))
        assert times_back.shape == times.shape
        assert np.allclose(times, times_back)
        for a, b in zip(truth, truth_back):
            assert np.allclose(a, b)


# ----------------------------------------------------------------------
# save/load_observations
# ----------------------------------------------------------------------
class TestObservationsIO:
    def test_netcdf_roundtrip(self, tmp_path):
        obs = [np.random.default_rng(k).standard_normal(8) for k in range(5)]
        times = np.linspace(0, 1, 5)
        path = tmp_path / "obs.nc"
        save_observations(obs, times, str(path))
        obs_back, times_back = load_observations(str(path))
        for a, b in zip(obs, obs_back):
            assert np.allclose(a, b)
