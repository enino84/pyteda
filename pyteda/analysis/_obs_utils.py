# -*- coding: utf-8 -*-
"""
Helpers shared by analysis filters.

`linearize_at_mean` returns (H, HXb) given an Observation and an ensemble:

    H   : Jacobian of h at the ensemble mean (constant for linear ops).
    HXb : h applied to every ensemble member (column by column).

This decouples the filter math from operator linearity. For linear
operators the result is the original `H` and `H @ Xb`; for nonlinear
operators it is the Jacobian at xbar plus the *exact* h(X) — which keeps
the innovation residual unbiased and the linear update consistent.
"""

from __future__ import annotations

import numpy as np


def linearize_at_mean(observation, Xb: np.ndarray):
    """Return (H, h(Xb)) for use inside a linear-update filter.

    Parameters
    ----------
    observation : Observation
        The observation container.
    Xb : ndarray
        Background ensemble of shape (n, N_ens).

    Returns
    -------
    H   : ndarray, shape (m, n)
    HXb : ndarray, shape (m, N_ens)
    """
    if hasattr(observation, "apply") and hasattr(observation, "linearize"):
        xbar = np.mean(Xb, axis=1)
        H = observation.linearize(xbar)
        HXb = observation.apply(Xb)
        return H, HXb
    # Fallback for non-refactored observation objects.
    H = observation.get_observation_operator()
    return H, H @ Xb
