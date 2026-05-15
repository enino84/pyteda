# -*- coding: utf-8 -*-
"""
EnKF based on the Ledoit-Wolf (LW) distribution-free covariance estimator.

Reference
---------
Nino-Ruiz, Guzman, Jabba (2021), "An ensemble Kalman filter implementation
based on the Ledoit and Wolf covariance matrix estimator", Journal of
Computational and Applied Mathematics 384, 113163.

The LW estimator (eq. 16 / 22b in the paper) does not assume Gaussianity
on the prior ensemble. It uses the same scaled-identity target as RBLW
but a different weight, computed in O(n·N) using the singular values of
ΔX:

    α_LW = min(  [(2-N)/(N-1)² · Σσ_e^4  +  Σ ‖Δx_e‖^4]
                 ⁄
                 [N² · ( Σσ_e^4/(N-1)²  −  (Σσ_e²/(N-1))² / n )]
              ,  1 )

The shrunk estimator combines the sample covariance with the target as in
RBLW:

    B̂_LW = α_LW · μ_LW · I + (1 − α_LW) · P^b ,    μ_LW = Σσ_e²/n

The analysis update follows eq. (23)–(25) of the paper, solved by the
matrix-free iterative Woodbury formula (Algorithm 1). No n × n matrices
are ever formed.
"""

from __future__ import annotations

import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis
from ._woodbury_solver import woodbury_solve, diagonal_solver, dense_lu_solver


@register_analysis("enkf-lw")
class AnalysisEnKFLW(Analysis):
    """EnKF based on the distribution-free Ledoit-Wolf shrinkage estimator.

    The LW estimator drops the Gaussian assumption on prior errors that
    RBLW imposes; this can improve performance when the model dynamics
    is strongly non-linear and the ensemble departs from Gaussianity.
    Like RBLW it is well-conditioned and full-rank by construction, so
    it does not require localisation.

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
    def _alpha_lw(sigma: np.ndarray, DX: np.ndarray,
                 N: int, n: int) -> tuple[float, float]:
        """Compute (μ_LW, α_LW) from σ and ΔX.

        Uses eqs. (22a–b) of the paper. ‖Δx_e‖^4 is computed directly
        from the columns of ΔX, never via the n × n matrix P^b.
        """
        s2 = np.asarray(sigma) ** 2
        s4 = s2 ** 2
        sum_s2 = float(s2.sum())
        sum_s4 = float(s4.sum())

        # Σ_e ‖Δx_e‖^4 — column-wise squared L2 norms, then squared and summed.
        col_sq = (DX ** 2).sum(axis=0)              # ‖Δx_e‖² for each column
        sum_col4 = float((col_sq ** 2).sum())       # Σ ‖Δx_e‖^4

        mu = sum_s2 / ((N - 1) * n)   # tr(P^b)/n  (note the (N-1) divisor)

        # Numerator and denominator of α_LW (paper eq. 22b).
        num = ((2 - N) / (N - 1) ** 2) * sum_s4 + sum_col4
        denom_in = (sum_s4 / (N - 1) ** 2) - ((sum_s2 / (N - 1)) ** 2) / n
        denom = (N ** 2) * denom_in

        if abs(denom) < 1e-300:
            alpha = 1.0
        else:
            alpha = min(num / denom, 1.0)
        alpha = max(0.0, float(alpha))
        return mu, alpha

    # -- main step --------------------------------------------------------
    def perform_assimilation(self, background, observation):
        Xb = background.get_ensemble()
        y = observation.get_observation()
        n, N = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)

        Ys = y[:, None] + observation.noise.sample_many_legacy(N)
        Dinn = Ys - HXb

        xb = Xb.mean(axis=1)
        DX = Xb - xb[:, None]
        sigma = np.linalg.svd(DX, full_matrices=False, compute_uv=False)
        mu, alpha = self._alpha_lw(sigma, DX, N=N, n=n)

        # We solve the system in OBSERVATION space (m × m, not n × n):
        #   [R + α μ · H H^T + ((1-α)/(N-1)) · (H ΔX)(H ΔX)^T] · Z = D_inn
        # Sherman-Morrison-Woodbury with:
        #   A_0 = R + α μ · H H^T   ∈ R^(m × m)
        #   Q   = √((1-α)/(N-1)) · H ΔX   ∈ R^(m × N)   (skinny!)
        # Note: H is sparse (LinearSelection); H ΔX is m × N which is the
        # only low-rank correction. For LinearSelection-type operators,
        # H H^T = I_m, so A_0 reduces to a diagonal matrix and a sparse LU
        # is not needed.
        m = HXb.shape[0]
        HDX = H @ DX                                                   # (m, N)
        if hasattr(HDX, "toarray"):
            HDX = HDX.toarray()
        Q_skinny = np.sqrt((1.0 - alpha) / (N - 1)) * np.asarray(HDX)  # (m, N)

        # Build A_0 = R + α μ · H H^T in observation space, m × m.
        # Detect the cheap LinearSelection case (H H^T = I_m).
        is_selection = type(observation.operator).__name__ == "LinearSelection"

        if is_selection:
            # H H^T = I_m. A_0 = diag(R) + α μ · I_m = diag(R_diag + α μ).
            if hasattr(observation.noise, "R_diag"):
                R_diag = observation.noise.R_diag
            else:
                R = observation.get_data_error_covariance()
                R_diag = np.diag(R) if R.ndim == 2 else np.asarray(R)
            A0_diag = np.asarray(R_diag) + alpha * mu
            A0_solver = diagonal_solver(A0_diag)
        else:
            # Generic case: form H H^T as an m × m matrix.
            H_dense = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
            HHt = H_dense @ H_dense.T
            R = observation.get_data_error_covariance()
            if hasattr(R, "toarray"):
                R = R.toarray()
            A0 = np.asarray(R) + alpha * mu * HHt
            A0_solver = dense_lu_solver(A0)

        Z_obs = woodbury_solve(A0_solver, [Q_skinny], [1.0], Dinn)     # (m, N)

        # Apply increments: X^a = X^b + B̂_LW · H^T · Z_obs
        # B̂_LW · H^T · Z = α μ · H^T Z + ((1-α)/(N-1)) · ΔX · (ΔX^T · H^T · Z)
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
