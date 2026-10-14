# -*- coding: utf-8 -*-
"""Whether observing one variable updates the others.

The localized filters take a ``cross`` flag that decides whether a local
domain (LETKF, LEnKF, obs-MC-local), a Cholesky predecessor set (EnKF-MC,
shrinkage, obs-MC, obs-Woodbury) or a decorrelation matrix (EnKF-BLoc)
reaches across variable blocks.

``cross=False`` is the default and gives a univariate analysis: observing
``h`` leaves ``u`` and ``v`` at exactly their forecast value. ``cross=True``
lets the observation reach the wind through the ensemble cross covariances.
These tests pin both branches down, because the difference is silent: both
produce a plausible-looking analysis of ``h``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.observation.grid_masks import strided_indices
from pyteda.observation.noise import IsotropicDiagonal
from pyteda.observation.operators import LinearSelection
from pyteda.analysis.analysis_factory import AnalysisFactory

warnings.filterwarnings("ignore", category=Warning)

swesphere = pytest.importorskip(
    "swesphere", reason="SWEModel needs the 'swe' extra: pip install pyteda[swe]"
)

#: filters whose localization can be made to cross variable blocks
CROSS_FILTERS = [
    ("letkf", dict(r=2)),
    ("lenkf", dict(r=2)),
    ("enkf-b-loc", dict(r=2)),
    ("enkf-modified-cholesky", dict(r=2)),
    ("enkf-shrinkage-precision", dict(r=2)),
]

ENSEMBLE_SIZE = 24


def _model():
    from pyteda.models import SWEModel

    return SWEModel(preset="two_jets", LMAX=4)


@pytest.fixture(scope="module")
def h_only_setup():
    """Ensemble and an observation network that touches the ``h`` block only."""
    model = _model()
    rng = np.random.default_rng(11)

    Xb = np.column_stack([
        model.get_initial_condition(seed=500 + e) for e in range(ENSEMBLE_SIZE)
    ])
    x_true = model.get_initial_condition(seed=9999)

    idx, _ = strided_indices(model, spacing=4, variables=["h"])
    m = idx.size
    op = LinearSelection(m, model.dim, indices=idx, rng=rng)
    noise = IsotropicDiagonal(1.0, dim=m)
    obs = Observation.from_arrays(op.apply(x_true) + noise.sample(rng),
                                  op, noise)

    bg = Background(model, ensemble_size=ENSEMBLE_SIZE)
    bg.Xb = Xb
    return model, bg, obs, Xb


def _block_increments(model, Xa, Xb):
    """Mean absolute increment per variable block."""
    fs = model.field_size
    d = np.asarray(Xa) - Xb
    return {
        name: float(np.abs(d[j * fs:(j + 1) * fs, :]).mean())
        for j, name in enumerate(model.var_blocks)
    }


def _run(method, kwargs, bg, obs, seed=0):
    """One analysis step, reproducibly.

    Several filters draw their synthetic observation perturbations from the
    legacy global numpy RNG, so without reseeding two runs of the same filter
    differ by sampling noise and nothing could be compared across runs.
    """
    # A fresh model per run: EnKF-BLoc caches the decorrelation matrix on it.
    model = _model()
    analysis = AnalysisFactory(
        method=method, model=model, **kwargs
    ).create_analysis()
    np.random.seed(seed)
    Xa = analysis.perform_assimilation(bg, obs)
    return model, Xa


# ----------------------------------------------------------------------
# The ensemble must have spread in every block, or the test below is void
# ----------------------------------------------------------------------
def test_every_block_has_ensemble_spread(h_only_setup):
    """A block with no spread cannot be updated, cross or not.

    ``two_jets`` is purely zonal, so ``v == 0`` in the reference state.
    ``SWEModel._perturb`` therefore scales ``u`` and ``v`` together; were it
    to scale each by its own standard deviation, ``v`` would stay identically
    zero across the ensemble and the cross-variable tests would pass for the
    wrong reason.
    """
    model, _, _, Xb = h_only_setup
    fs = model.field_size
    for j, name in enumerate(model.var_blocks):
        spread = Xb[j * fs:(j + 1) * fs, :].std(axis=1).mean()
        assert spread > 0.0, f"block {name!r} has no ensemble spread"


# ----------------------------------------------------------------------
# cross=False — univariate
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method,kwargs", CROSS_FILTERS)
def test_default_is_univariate(method, kwargs, h_only_setup):
    """Observing h leaves u and v untouched, exactly, with the default."""
    _, bg, obs, Xb = h_only_setup
    model, Xa = _run(method, kwargs, bg, obs)
    inc = _block_increments(model, Xa, Xb)

    assert inc["h"] > 0.0, "the observed block was not updated at all"
    assert inc["u"] == 0.0
    assert inc["v"] == 0.0


@pytest.mark.parametrize("method,kwargs", CROSS_FILTERS)
def test_cross_false_matches_the_default(method, kwargs, h_only_setup):
    """Passing cross=False explicitly is the same as not passing it."""
    _, bg, obs, Xb = h_only_setup
    model_a, Xa_default = _run(method, kwargs, bg, obs)
    model_b, Xa_false = _run(method, dict(kwargs, cross=False), bg, obs)
    assert np.allclose(np.asarray(Xa_default), np.asarray(Xa_false))


# ----------------------------------------------------------------------
# cross=True — multivariate
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method,kwargs", CROSS_FILTERS)
def test_cross_true_updates_the_unobserved_blocks(method, kwargs,
                                                  h_only_setup):
    """With cross=True an observation of h moves u and v."""
    _, bg, obs, Xb = h_only_setup
    model, Xa = _run(method, dict(kwargs, cross=True), bg, obs)
    inc = _block_increments(model, Xa, Xb)

    assert inc["h"] > 0.0
    assert inc["u"] > 0.0, "cross=True did not reach the u block"
    assert inc["v"] > 0.0, "cross=True did not reach the v block"


@pytest.mark.parametrize("method,kwargs", CROSS_FILTERS)
def test_cross_scale_zero_is_univariate(method, kwargs, h_only_setup):
    """cross_scale=0 tapers the cross-block links away again.

    Only EnKF-BLoc takes ``cross_scale``; the neighbourhood-based filters
    have no taper, so they are skipped.
    """
    if method != "enkf-b-loc":
        pytest.skip("cross_scale only applies to the decorrelation matrix")
    _, bg, obs, Xb = h_only_setup
    model, Xa = _run(method, dict(kwargs, cross=True, cross_scale=0.0),
                     bg, obs)
    inc = _block_increments(model, Xa, Xb)
    assert inc["u"] == 0.0
    assert inc["v"] == 0.0


# ----------------------------------------------------------------------
# The unlocalized EnKF has no cross flag: it is always multivariate
# ----------------------------------------------------------------------
def test_plain_enkf_is_always_multivariate(h_only_setup):
    """Plain EnKF uses the full sample covariance, so it needs no flag.

    This also covers the matrix-free (Woodbury) branch, which the state
    dimension here (600 >= SPARSE_THRESHOLD) selects.
    """
    _, bg, obs, Xb = h_only_setup
    model, Xa = _run("enkf", {}, bg, obs)
    inc = _block_increments(model, Xa, Xb)
    assert inc["h"] > 0.0
    assert inc["u"] > 0.0
    assert inc["v"] > 0.0


# ----------------------------------------------------------------------
# cross on a model that does not support it
# ----------------------------------------------------------------------
def test_cross_true_on_lorenz96_is_a_noop(small_lorenz96):
    """Lorenz96 has one block; cross is accepted and changes nothing."""
    n = small_lorenz96.n
    assert np.array_equal(
        small_lorenz96.get_ngb(5, 2),
        small_lorenz96.get_ngb(5, 2, cross=True),
    )
    small_lorenz96.create_decorrelation_matrix(2, cross=True, cross_scale=0.0)
    L_cross = small_lorenz96.get_decorrelation_matrix().copy()
    small_lorenz96.create_decorrelation_matrix(2)
    assert np.allclose(L_cross, small_lorenz96.get_decorrelation_matrix())


def test_cross_true_on_a_legacy_model_raises():
    """A model whose get_ngb takes only (i, r) gets a message, not silence."""
    from pyteda.analysis._loc_utils import call_ngb, supports_cross

    class LegacyModel:
        def get_ngb(self, i, r):
            return np.array([i])

    model = LegacyModel()
    assert not supports_cross(model.get_ngb)
    # cross=False still works: nothing is forwarded.
    assert np.array_equal(call_ngb(model, 3, 1), np.array([3]))
    with pytest.raises(TypeError, match="does not accept a `cross` keyword"):
        call_ngb(model, 3, 1, cross=True)
