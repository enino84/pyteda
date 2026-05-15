# -*- coding: utf-8 -*-
"""Tests for Scenario.generate (3-phase recipe) and Scenario.save/load."""

import warnings

import numpy as np
import pytest

from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario


warnings.filterwarnings("ignore", category=Warning)


# ----------------------------------------------------------------------
# 3-phase ensemble construction
# ----------------------------------------------------------------------
class TestThreePhaseRecipe:
    """Verify each phase produces the expected artifact."""

    @pytest.fixture
    def scen(self, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        return Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=16, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=16),
            ensemble_size=8,
            spinup_truth=1.0, pert_xb=0.5, spinup_xb=0.2,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.5, seed=42,
        )

    def test_x0_ref_attached(self, scen, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        assert hasattr(scen, "x0_ref")
        assert scen.x0_ref.shape == (n,)
        assert np.isfinite(scen.x0_ref).all()

    def test_xb_attached(self, scen, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        assert hasattr(scen, "xb")
        assert scen.xb.shape == (n,)
        assert np.isfinite(scen.xb).all()

    def test_xb_separated_from_x0_ref(self, scen):
        # With pert_xb=0.5 and Lorenz96 chaos for 0.2 time units, xb should be
        # measurably separated from x0_ref.
        sep = np.linalg.norm(scen.xb - scen.x0_ref) / np.linalg.norm(scen.x0_ref)
        assert sep > 0.05  # at least 5% separation

    def test_initial_ensemble_dispersed_around_xb(self, scen):
        # mean(X_b) should be close to xb (smaller separation than xb vs x0_ref)
        mean_X = scen.initial_ensemble.mean(axis=1)
        sep_mean_xb = np.linalg.norm(mean_X - scen.xb) / np.linalg.norm(scen.xb)
        sep_xb_x0 = np.linalg.norm(scen.xb - scen.x0_ref) / np.linalg.norm(scen.x0_ref)
        # Members should be more centered on xb than xb is on x0_ref.
        assert sep_mean_xb < sep_xb_x0

    def test_ensemble_has_spread(self, scen):
        # std across members should be > 0
        sigma = scen.initial_ensemble.std(axis=1)
        assert np.all(sigma > 0)


class TestZeroSpinups:
    """When all spinups are 0, the construction is purely instantaneous."""

    def test_zero_spinups_works(self, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        scen = Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=5,
            spinup_truth=0.0, pert_xb=0.5, spinup_xb=0.0,
            pert_ensemble=0.05, spinup_ensemble=0.0,
            obs_freq=0.1, end_time=0.3, seed=42,
        )
        # x0_ref = synth_IC, xb = x0_ref + pert (no chaos), truth[0] = x0_ref
        assert np.allclose(scen.truth_trajectory[0], scen.x0_ref)
        assert np.linalg.norm(scen.xb - scen.x0_ref) > 0  # at least the pert kicked


# ----------------------------------------------------------------------
# Pre-computed artifact reuse
# ----------------------------------------------------------------------
class TestPreComputedArtifacts:
    """Pre-computed x0_ref, xb, initial_ensemble must be respected as-is."""

    @pytest.fixture
    def base_scen(self, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        return Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=5,
            spinup_truth=1.0, pert_xb=0.5, spinup_xb=0.2,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.3, seed=42,
        )

    def test_preloaded_x0_ref_reused(self, small_lorenz96, base_scen):
        n = small_lorenz96.get_number_of_variables()
        scen2 = Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=5,
            x0_ref=base_scen.x0_ref,  # preloaded
            pert_xb=0.5, spinup_xb=0.2,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.3, seed=42,
        )
        assert np.allclose(scen2.x0_ref, base_scen.x0_ref)

    def test_preloaded_xb_reused(self, small_lorenz96, base_scen):
        n = small_lorenz96.get_number_of_variables()
        scen2 = Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=5,
            x0_ref=base_scen.x0_ref, xb=base_scen.xb,
            pert_ensemble=0.05, spinup_ensemble=0.1,
            obs_freq=0.1, end_time=0.3, seed=42,
        )
        assert np.allclose(scen2.xb, base_scen.xb)

    def test_preloaded_initial_ensemble_reused(self, small_lorenz96, base_scen):
        n = small_lorenz96.get_number_of_variables()
        scen2 = Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            x0_ref=base_scen.x0_ref, xb=base_scen.xb,
            initial_ensemble=base_scen.initial_ensemble,
            spinup_xb=0.2, spinup_ensemble=0.1,  # for truth sync
            obs_freq=0.1, end_time=0.3, seed=42,
        )
        assert np.allclose(scen2.initial_ensemble, base_scen.initial_ensemble)


# ----------------------------------------------------------------------
# Backwards-compatibility aliases
# ----------------------------------------------------------------------
class TestLegacyAliases:
    """Old parameter names from before the 3-phase refactor must still work."""

    def test_initial_perturbation_alias(self, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        scen = Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=5,
            initial_perturbation=0.07,  # legacy alias
            obs_freq=0.1, end_time=0.3, seed=42,
        )
        assert scen.meta["pert_ensemble"] == 0.07

    def test_spinup_time_alias(self, small_lorenz96):
        n = small_lorenz96.get_number_of_variables()
        scen = Scenario.generate(
            model=small_lorenz96,
            operator_factory=lambda rng: LinearSelection(m=10, n_state=n, rng=rng),
            noise=IsotropicDiagonal(std=0.01, dim=10),
            ensemble_size=5,
            spinup_time=np.arange(0, 0.5, 0.01),  # legacy alias
            obs_freq=0.1, end_time=0.3, seed=42,
        )
        # spinup_time -> spinup_truth = endpoint of array
        assert scen.meta["spinup_truth"] == pytest.approx(0.49)


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------
class TestScenarioIO:
    def test_netcdf_roundtrip(self, small_scenario, small_lorenz96, tmp_data_dir):
        path = tmp_data_dir / "scen.nc"
        small_scenario.save(str(path))
        scen2 = Scenario.load(str(path), model=small_lorenz96)
        # All key arrays preserved
        assert np.allclose(scen2.initial_ensemble, small_scenario.initial_ensemble)
        for a, b in zip(scen2.truth_trajectory, small_scenario.truth_trajectory):
            assert np.allclose(a, b)
        for a, b in zip(scen2.observations, small_scenario.observations):
            assert np.allclose(a, b)
        assert scen2.meta["config_hash"] == small_scenario.meta["config_hash"]

    def test_npz_roundtrip(self, small_scenario, small_lorenz96, tmp_data_dir):
        path = tmp_data_dir / "scen.npz"
        small_scenario.save(str(path))
        scen2 = Scenario.load(str(path), model=small_lorenz96)
        assert np.allclose(scen2.initial_ensemble, small_scenario.initial_ensemble)
