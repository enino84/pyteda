# -*- coding: utf-8 -*-
"""The square-root filters against the Kalman filter they approximate.

With a linear observation operator, Gaussian errors and the sample covariance
``P^b = DX DX^T / (N - 1)`` as background covariance, a deterministic
square-root filter must return exactly

    x^a = x^b + K (y - H x^b),      K = P^b H^T (H P^b H^T + R)^-1
    P^a = (I - K H) P^b

for the mean and for the covariance of the analysis ensemble. LETKF must do
the same when every local domain is the whole state.

These tests exist because ETKF, LETKF and EnSRF once used raw deviations where
the algebra needs them divided by ``sqrt(N - 1)``: they assimilated with a
background covariance ``N - 1`` times too large, which no other test noticed
because the filters still ran, still reduced the error and still agreed with
their own snapshots.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyteda.analysis.analysis_factory import AnalysisFactory
from pyteda.background import Background
from pyteda.models import Lorenz96
from pyteda.observation import IsotropicDiagonal, LinearSelection, Observation

N_STATE, N_ENS, N_OBS, OBS_STD = 13, 30, 5, 0.5      # odd: radius 6 covers the ring once


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    model = Lorenz96(n=N_STATE)
    mix = np.eye(N_STATE) + 0.5 * rng.standard_normal((N_STATE, N_STATE))
    Xb = mix @ rng.standard_normal((N_STATE, N_ENS)) + rng.standard_normal((N_STATE, 1))
    idx = np.sort(rng.choice(N_STATE, N_OBS, replace=False))
    op = LinearSelection(N_OBS, N_STATE, indices=idx, rng=rng)
    noise = IsotropicDiagonal(OBS_STD, dim=N_OBS)
    y = rng.standard_normal(N_OBS)
    bg = Background(model, ensemble_size=N_ENS)
    bg.Xb = Xb.copy()
    obs = Observation.from_arrays(y, op, noise)
    H = np.zeros((N_OBS, N_STATE))
    H[np.arange(N_OBS), idx] = 1.0
    return model, bg, obs, Xb, H, y


def _kalman(Xb, H, y, scale):
    xb = Xb.mean(axis=1)
    DX = Xb - xb[:, None]
    P = DX @ DX.T / scale
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + OBS_STD ** 2 * np.eye(N_OBS))
    return xb + K @ (y - H @ xb), (np.eye(N_STATE) - K @ H) @ P


@pytest.mark.parametrize("method, kwargs", [
    ("etkf", {}),
    ("ensrf", {}),
    ("letkf", dict(r=N_STATE // 2)),          # every local domain is the whole state
])
def test_square_root_filters_reproduce_the_kalman_update(method, kwargs):
    model, bg, obs, Xb, H, y = _setup()
    analysis = AnalysisFactory(method=method, model=model, **kwargs).create_analysis()
    Xa = np.asarray(analysis.perform_assimilation(bg, obs))
    xa_ref, Pa_ref = _kalman(Xb, H, y, scale=N_ENS - 1)
    assert np.allclose(Xa.mean(axis=1), xa_ref, atol=1e-9)
    assert np.allclose(np.cov(Xa), Pa_ref, atol=1e-9)
    # and it is not the update for a covariance N - 1 times too large
    xa_wrong, _ = _kalman(Xb, H, y, scale=1.0)
    assert np.abs(Xa.mean(axis=1) - xa_wrong).max() > 1e-3


def test_stochastic_enkf_agrees_in_the_mean():
    """The perturbed-observation EnKF has the same gain; its mean matches on average."""
    model, bg, obs, Xb, H, y = _setup()
    xa_ref, _ = _kalman(Xb, H, y, scale=N_ENS - 1)
    means = []
    for _ in range(300):
        analysis = AnalysisFactory(method="enkf", model=model).create_analysis()
        bg.Xb = Xb.copy()
        means.append(np.asarray(analysis.perform_assimilation(bg, obs)).mean(axis=1))
    assert np.abs(np.mean(means, axis=0) - xa_ref).max() < 0.05
