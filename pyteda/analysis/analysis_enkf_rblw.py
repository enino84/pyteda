# -*- coding: utf-8 -*-
"""
EnKF based on the Rao-Blackwell Ledoit-Wolf (RBLW) covariance estimator.

Reference
---------
Nino-Ruiz, Guzman, Jabba (2021), "An ensemble Kalman filter implementation
based on the Ledoit and Wolf covariance matrix estimator", Journal of
Computational and Applied Mathematics 384, 113163.

The RBLW estimator (eq. 17 in the paper) is the optimal shrinkage weight
when prior ensemble members are Gaussian:

    α_RBLW = min( [(N-2)/n · tr(P^b²) + tr²(P^b)] /
                  [(N+2) · (tr(P^b²) - tr²(P^b)/n)] ,  1 )

The shrunk covariance combines the sample covariance with a scaled-identity
target T = (tr(P^b)/n) · I:

    B̂_RBLW = α · μ · I + (1-α) · P^b ,    μ = tr(P^b)/n

The analysis update (eq. 18a–c) is

    X^a = X^b + α·μ·H^T·Z + (1-α)·P^b·H^T·Z

with Z solving the matrix-free system

    [R + α·μ·H·H^T + (1-α)·H·P^b·H^T] · Z = D

This implementation never forms the n × n matrices P^b, B, or H·P^b·H^T.
The system is solved via the iterative Sherman-Morrison-Woodbury formula
(Algorithm 1 of the paper) by writing the bracketed operator as

    R + Q^(1) (Q^(1))^T + Q^(2) (Q^(2))^T

with Q^(1) = √(α·μ) · H ∈ R^(m×n)
     Q^(2) = √((1-α)/(N-1)) · H · ΔX ∈ R^(m×N)

For diagonal R the solver applies R^{-1} via row-scaling — so the cost is
linear in n and m. The full state-space covariance is never built.
"""

from __future__ import annotations

import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis
from ._woodbury_solver import woodbury_solve, diagonal_solver, dense_lu_solver


@register_analysis("enkf-rblw")
class AnalysisEnKFRBLW(Analysis):
    """EnKF based on the Rao-Blackwell Ledoit-Wolf shrinkage estimator.

    Suited to high-dimensional problems where the sample covariance is
    rank-deficient and Gaussian assumptions on prior errors are
    reasonable. The optimal shrinkage α is computed from the singular
    values of ΔX, never from P^b directly. The analysis update is
    matrix-free via Woodbury.

    Parameters
    ----------
    model : Model
        The dynamical model.
    """

    def __init__(self, model=None, **kwargs):
        self.model = model
        self.Xa = None

    # -- shrinkage weight --------------------------------------------------
    @staticmethod
    def _alpha_rblw(sigma: np.ndarray, N: int, n: int) -> tuple[float, float]:
        """Compute (μ, α_RBLW) from the singular values of ΔX.

        Uses the trace identities
            tr(P^b)   = (1/(N-1)) · Σ σ_e²
            tr(P^b²)  = (1/(N-1)²) · Σ σ_e^4
        from eqs. (20) of the paper, which avoid forming P^b explicitly.
        """
        s2 = np.asarray(sigma) ** 2
        s4 = s2 ** 2
        sum_s2 = float(s2.sum())
        sum_s4 = float(s4.sum())

        tr_Pb  = sum_s2 / (N - 1)
        tr_Pb2 = sum_s4 / (N - 1) ** 2
        mu = tr_Pb / n

        denom = (N + 2) * (tr_Pb2 - (tr_Pb ** 2) / n)
        if abs(denom) < 1e-300:
            alpha = 1.0
        else:
            num = ((N - 2) / n) * tr_Pb2 + tr_Pb ** 2
            alpha = min(num / denom, 1.0)
        # Guard against negative numerators producing α < 0 in pathological cases.
        alpha = max(0.0, float(alpha))
        return mu, alpha

    # -- main step --------------------------------------------------------
    def perform_assimilation(self, background, observation):
        """Run one EnKF-RBLW analysis step."""
        Xb = background.get_ensemble()                # (n, N)
        y = observation.get_observation()             # (m,)
        n, N = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)   # H may be sparse

        Ys = y[:, None] + observation.noise.sample_many_legacy(N)
        Dinn = Ys - HXb

        xb = Xb.mean(axis=1)
        DX = Xb - xb[:, None]
        sigma = np.linalg.svd(DX, full_matrices=False, compute_uv=False)
        mu, alpha = self._alpha_rblw(sigma, N=N, n=n)

        # Solve in OBSERVATION space (m × m, not n × n):
        #   [R + α μ · H H^T + ((1-α)/(N-1)) · (H ΔX)(H ΔX)^T] · Z = D_inn
        # Sherman-Morrison-Woodbury with:
        #   A_0 = R + α μ · H H^T   ∈ R^(m × m)
        #   Q   = √((1-α)/(N-1)) · H ΔX   ∈ R^(m × N)   (skinny)
        # For LinearSelection-type operators, H H^T = I_m and A_0 reduces
        # to a diagonal — no LU factorisation needed.
        m = HXb.shape[0]
        HDX = H @ DX                                                   # (m, N)
        if hasattr(HDX, "toarray"):
            HDX = HDX.toarray()
        Q_skinny = np.sqrt((1.0 - alpha) / (N - 1)) * np.asarray(HDX)  # (m, N)

        is_selection = type(observation.operator).__name__ == "LinearSelection"

        if is_selection:
            # H H^T = I_m. A_0 = diag(R_diag + α μ).
            if hasattr(observation.noise, "R_diag"):
                R_diag = observation.noise.R_diag
            else:
                R = observation.get_data_error_covariance()
                R_diag = np.diag(R) if R.ndim == 2 else np.asarray(R)
            A0_diag = np.asarray(R_diag) + alpha * mu
            A0_solver = diagonal_solver(A0_diag)
        else:
            H_dense = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
            HHt = H_dense @ H_dense.T
            R = observation.get_data_error_covariance()
            if hasattr(R, "toarray"):
                R = R.toarray()
            A0 = np.asarray(R) + alpha * mu * HHt
            A0_solver = dense_lu_solver(A0)

        Z_obs = woodbury_solve(A0_solver, [Q_skinny], [1.0], Dinn)     # (m, N)

        # Apply increments: X^a = X^b + B̂_RBLW · H^T · Z_obs
        # B̂_RBLW · H^T · Z = α μ · H^T Z + ((1-α)/(N-1)) · ΔX · (ΔX^T · H^T · Z)
        HtZ = H.T @ Z_obs                                              # (n, N)
        if hasattr(HtZ, "toarray"):
            HtZ = HtZ.toarray()
        HtZ = np.asarray(HtZ)
        increments = (
            alpha * mu * HtZ
            + ((1 - alpha) / (N - 1)) * (DX @ (DX.T @ HtZ))
        )
        self.Xa = Xb + increments
        self.alpha_ = alpha
        self.mu_ = mu
        return self.Xa

    # -- standard accessors ------------------------------------------------
    def get_analysis_state(self):
        return self.Xa.mean(axis=1)

    def get_ensemble(self):
        return self.Xa

    def get_error_covariance(self):
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        n, N = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - xa[:, None]
        self.Xa = xa[:, None] + inflation_factor * DXa
