# -*- coding: utf-8 -*-
"""Framework-level adaptive multiplicative inflation.

This estimator lives in the CYCLING layer, not inside any single analysis
filter, so that every method (EnKF, LETKF, LEnKF, EnKF-BLoc, the modified
Cholesky filters, ...) receives the SAME adaptive-inflation treatment and
inter-method comparisons stay fair.

Method (innovation / observation-space diagnostic)
--------------------------------------------------
At each cycle the observed innovation is ``d = y - H xbar_b``. Under a
correctly specified filter its statistics satisfy

    E[ d^T d ] = tr(H P^b H^T) + tr(R),

i.e. the mean-square innovation equals the predicted observation-space
variance plus the observation-error variance. Writing the realised and
predicted mean-square innovations

    obs_ms  = (d^T d) / m,
    pred_ms = ( tr(H P^b H^T) + tr(R) ) / m,

a single multiplicative inflation factor ``lambda`` that would have made the
ENSEMBLE part consistent solves

    obs_ms = lambda * tr(H P^b H^T)/m + tr(R)/m,

so the per-cycle target is

    lambda_obs = ( obs_ms - tr(R)/m ) / ( tr(H P^b H^T)/m ).

This is the standard innovation-based multiplicative-inflation diagnostic
(Anderson 2007/2009; Miyoshi 2011; Desroziers et al. 2005 consistency
relations). The raw per-cycle estimate ``lambda_obs`` is noisy with small
ensembles, so it is smoothed by a first-order recursion

    lambda <- (1 - g) * lambda + g * sqrt(max(lambda_obs, eps)),

(the square root because the factor multiplies DEVIATIONS, whose square is
the variance the diagnostic constrains) and clipped to ``[lo, hi]``. The
filter then inflates its analysis (or background) deviations by the smoothed
``lambda`` exactly as for a fixed factor.

Matrix-free / sparse-safe
-------------------------
Only traces and a single matrix-vector style reduction are needed:
``tr(H P^b H^T) = sum over members ( ||H dxb^[e]||^2 ) / (N-1)`` is computed
from the ensemble in observation space without ever forming ``P^b`` or an
``m x m`` matrix, and ``tr(R)`` comes from the diagonal accessor of the noise
model. Everything is O(m N), so adaptive inflation adds negligible cost.
"""

from __future__ import annotations

import numpy as np


class AdaptiveInflation:
    """Innovation-based adaptive multiplicative inflation (framework-level).

    Parameters
    ----------
    lambda0 : float
        Initial inflation factor (applied to deviations). Default 1.04.
    gain : float
        Smoothing gain ``g`` of the first-order recursion, in (0, 1]. Small
        values track slowly but are robust to per-cycle noise. Default 0.1.
    lo, hi : float
        Clip bounds for the inflation factor. Defaults 1.0 and 2.0.
    eps : float
        Floor inside the square root to avoid NaNs when the realised
        innovation is smaller than the observation-error variance (which can
        happen by sampling noise). Default 1e-6.

    Notes
    -----
    The same instance is updated once per cycle via :meth:`update` and queried
    via :meth:`factor`. It is filter-agnostic: it only consumes the background
    ensemble in observation space, the innovation, and ``tr(R)``.
    """

    def __init__(self, lambda0: float = 1.04, gain: float = 0.1,
                 lo: float = 1.0, hi: float = 2.0, eps: float = 1e-6):
        self.lmbda = float(lambda0)
        self.gain = float(gain)
        self.lo = float(lo)
        self.hi = float(hi)
        self.eps = float(eps)
        # History for diagnostics / reporting.
        self.history = [self.lmbda]

    def factor(self) -> float:
        """Return the current (smoothed, clipped) inflation factor."""
        return float(np.clip(self.lmbda, self.lo, self.hi))

    def update(self, Xb: np.ndarray, H, y: np.ndarray, trR: float) -> float:
        """Update the inflation factor from this cycle's innovation.

        Parameters
        ----------
        Xb : ndarray, shape (n, N)
            Background ensemble.
        H : ndarray or sparse, shape (m, n)
            Linear observation operator (supports ``@``).
        y : ndarray, shape (m,)
            Observation vector.
        trR : float
            Trace of the observation-error covariance ``R`` (sum of its
            diagonal). For isotropic ``R = sigma^2 I`` this is ``m sigma^2``.

        Returns
        -------
        float
            The updated (smoothed, clipped) inflation factor.
        """
        n, N = Xb.shape
        xb = Xb.mean(axis=1)
        # Observation-space ensemble deviations: H dxb for each member.
        DX = Xb - xb[:, None]
        HDX = H @ DX                              # (m, N), matrix-free
        HDX = HDX.toarray() if hasattr(HDX, "toarray") else np.asarray(HDX)
        m = HDX.shape[0]

        # Predicted ensemble part: tr(H P^b H^T) = sum ||H dxb^[e]||^2 / (N-1).
        tr_HPbHt = float(np.sum(HDX * HDX)) / max(N - 1, 1)

        # Observed mean-square innovation.
        Hxb = H @ xb
        Hxb = Hxb.toarray().ravel() if hasattr(Hxb, "toarray") else np.asarray(Hxb).ravel()
        d = np.asarray(y).ravel() - Hxb
        obs_ss = float(d @ d)                     # sum of squares

        # Per-cycle target multiplier on the ENSEMBLE variance:
        #   obs_ss = lambda^2 * tr(H P^b H^T) + tr(R)
        # (lambda multiplies deviations, so lambda^2 multiplies variances).
        denom = tr_HPbHt
        if denom <= 0:
            return self.factor()
        lam2_obs = (obs_ss - trR) / denom
        lam_obs = np.sqrt(max(lam2_obs, self.eps))

        # First-order smoothing.
        self.lmbda = (1.0 - self.gain) * self.lmbda + self.gain * lam_obs
        f = self.factor()
        self.history.append(f)
        return f