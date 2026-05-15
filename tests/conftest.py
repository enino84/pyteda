# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the TEDA test suite.

Fixtures here are designed to be cheap so the suite stays fast. Slower
configurations (QG, SWE, large benchmarks) build their own dedicated
inputs inside the relevant test files.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyteda.models import Lorenz96
from pyteda.observation import LinearSelection, IsotropicDiagonal
from pyteda.experiments import Scenario


# Suppress harmless warnings (sparse efficiency from QG, etc.) globally.
warnings.filterwarnings("ignore", category=Warning)


@pytest.fixture
def small_lorenz96():
    """A small Lorenz96 instance (n=20) for fast tests."""
    return Lorenz96(n=20)


@pytest.fixture
def lorenz96_40():
    """The standard Lorenz96 instance (n=40) for tests that need it."""
    return Lorenz96(n=40)


@pytest.fixture
def small_scenario(small_lorenz96):
    """A minimal but realistic 3-phase scenario for filter/benchmark tests.

    Lorenz96 with n=20, 16 observations, ensemble of 8, 5 assimilation
    steps. Builds in well under a second on a laptop.
    """
    model = small_lorenz96
    n = model.get_number_of_variables()
    return Scenario.generate(
        model=model,
        operator_factory=lambda rng: LinearSelection(m=16, n_state=n, rng=rng),
        noise=IsotropicDiagonal(std=0.01, dim=16),
        ensemble_size=8,
        spinup_truth=1.0,
        pert_xb=0.5,
        spinup_xb=0.2,
        pert_ensemble=0.05,
        spinup_ensemble=0.1,
        obs_freq=0.1,
        end_time=0.5,
        seed=42,
    )


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Per-test temporary directory for IO roundtrips."""
    return tmp_path
