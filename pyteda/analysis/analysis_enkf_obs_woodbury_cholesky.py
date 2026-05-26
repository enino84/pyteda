# -*- coding: utf-8 -*-
"""Observation-space EnKF via modified-Cholesky background precision and
Sherman--Morrison--Woodbury.

This is the CORRECTED observation-space formulation. The earlier
``AnalysisEnKFObsModifiedCholesky`` tried to band the modified-Cholesky
factor of the *innovation* precision S^{-1} = (R + H B H^T)^{-1} directly in
observation space. That fails structurally: H B H^T is not banded in any
observation ordering (projecting the state-space locality of B through H
destroys it), so S^{-1} is not bandable and the Bickel--Levina hypothesis
does not hold in observation space.

Here we keep the modified-Cholesky estimator where it is legitimate --- on
the *state-space* background precision B^{-1}, ordered by the model grid,
where banding is genuine spatial locality (exactly as Nino-Ruiz, Sandu &
Deng do) --- and obtain the ACTION of the innovation precision S^{-1} in
observation space by the Sherman--Morrison--Woodbury identity

    S^{-1} = R^{-1} - R^{-1} H A^{-1} H^T R^{-1},
    A      = B^{-1} + H^T R^{-1} H        (state space, sparse SPD banded).

The Kalman gain K = B H^T S^{-1} applied to the innovation D never forms or
inverts an m x m matrix. Two equivalent routes are provided:

  * state-space (information form, default): the analysis increment is the
    single sparse solve
        A z = H^T R^{-1} D,    X_a = X_b + z,
    which is the cheapest route when m is comparable to or larger than n.
  * Woodbury (observation form): apply S^{-1} to a vector explicitly via the
    identity above (also a single sparse solve with A); exposed for the
    error-bound diagnostics, and cheaper when m << n.

Everything that must be banded lives in STATE space (B^{-1} and A); the
observation operator enters only through sparse mat-vecs and the diagonal
R^{-1}. R^{-1} is kept as a 1-D diagonal and never densified.

Error control. Because B^{-1} is estimated by the banded modified-Cholesky
estimator of Bickel & Levina in state space, its operator-norm error obeys
their rate; this propagates by elementary matrix perturbation to the Kalman
gain K and the analysis covariance P^a (see the accompanying theory). No
step requires S^{-1} itself to be bandable, so the bounds hold for general
linear H -- H enters only as the multiplicative factor ||H||.
"""

import numpy as np
from scipy.sparse import issparse, diags, csr_matrix, identity as sp_identity
from scipy.sparse.linalg import spsolve, factorized
from sklearn.linear_model import Ridge

from .analysis_core import Analysis
from .registry import register_analysis


@register_analysis("enkf-obs-woodbury-cholesky")
class AnalysisEnKFObsWoodburyCholesky(Analysis):
    """Observation-space EnKF: modified-Cholesky B^{-1} (state space) +
    Sherman--Morrison--Woodbury action of S^{-1} (observation space).

    Global (no domain localization). Sparse linear algebra is used whenever
    the state dimension crosses ``SPARSE_THRESHOLD``; below it dense solves
    are used. The modified-Cholesky banding lives entirely in state space.

    Attributes:
        model (Model): exposes ``get_pre(i, r)`` for the state-space
            predecessors used to estimate B^{-1}.
        r (int|dict|ndarray): localization radius / bandwidth for B^{-1}.
        alpha (float): Ridge penalty for the state-space regressions.
        cholesky_method (str): 'ridge' or 'svd' per-row solver for B^{-1}.
        cholesky_tol (float): relative SVD cutoff when method == 'svd'.

    Methods:
        get_background_precision(DX): banded modified-Cholesky B^{-1}.
        apply_Sinv(H, Rinv_diag, V, Binv): Woodbury action of S^{-1} on V.
        perform_assimilation(background, observation): analysis step.
    """

    def __init__(self, model=None, r=1, alpha: float = 0.01,
                 cholesky_method: str = "ridge", cholesky_tol: float = 0.1,
                 use_woodbury: bool = False, **kwargs):
        """
        Parameters
        ----------
        model : Model
            Provides ``get_pre(i, r)`` (state-space predecessors of B^{-1}).
        r : int | dict | ndarray
            Bandwidth/radius for the state-space modified-Cholesky B^{-1}.
        alpha : float
            Ridge penalty for the per-row state-space regressions.
        cholesky_method : {'ridge','svd'}
            Per-row conditional-regression solver for B^{-1}.
        cholesky_tol : float
            Relative singular-value cutoff for the 'svd' solver.
        use_woodbury : bool
            If True, compute the analysis through the explicit Woodbury
            action of S^{-1} in observation space (cheaper when m << n).
            If False (default), use the equivalent state-space information
            solve  A z = H^T R^{-1} D  (cheaper when m >~ n).
        """
        self.model = model
        self.r = r
        self.alpha = float(alpha)
        if cholesky_method not in ("ridge", "svd"):
            raise ValueError(
                f"cholesky_method must be 'ridge' or 'svd'; got {cholesky_method!r}."
            )
        self.cholesky_method = cholesky_method
        self.cholesky_tol = float(cholesky_tol)
        self.use_woodbury = bool(use_woodbury)

    # ------------------------------------------------------------------
    # State-space background precision B^{-1} (banded modified Cholesky)
    # ------------------------------------------------------------------
    def _local_regression(self, X, y, lr):
        if self.cholesky_method == "svd":
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            if s.size == 0 or s[0] == 0:
                return np.zeros(X.shape[1]), y.copy()
            keep = s / s[0] > self.cholesky_tol
            if not np.any(keep):
                return np.zeros(X.shape[1]), y.copy()
            Uk, sk, Vk = U[:, keep], s[keep], Vt[keep, :]
            coef = Vk.T @ ((Uk.T @ y) / sk)
            return coef, y - X @ coef
        lr_fit = lr.fit(X, y)
        return lr_fit.coef_, y - lr_fit.predict(X)

    def get_background_precision(self, DX):
        """Estimate B^{-1} = T^T D^{-1} T by banded modified Cholesky of the
        background deviations DX (n x N), each state component regressed on
        its grid predecessors via ``model.get_pre``.

        Returns a sparse CSR matrix (large n) or a dense ndarray (small n).
        """
        n, ensemble_size = DX.shape
        lr = Ridge(fit_intercept=False, alpha=self.alpha)
        from ..observation.noise import SPARSE_THRESHOLD
        sparse_path = n >= SPARSE_THRESHOLD
        max_pre = max(ensemble_size - 2, 1)
        var_floor = 1e-12 * max(float(np.mean(np.var(DX, axis=1))), 1e-300)

        if sparse_path:
            from scipy.sparse import lil_matrix
            L = lil_matrix((n, n), dtype=float)
            L.setdiag(1.0)
        else:
            L = np.eye(n)
        D_diag = np.empty(n, dtype=float)

        D_diag[0] = 1.0 / max(np.var(DX[0, :]), var_floor)
        for i in range(1, n):
            ind = np.asarray(self.model.get_pre(i, self.r), dtype=int)
            if ind.size > max_pre:
                ind = ind[-max_pre:]
            y = DX[i, :]
            if ind.size == 0:
                D_diag[i] = 1.0 / max(np.var(y), var_floor)
                continue
            X = DX[ind, :].T
            coef, err = self._local_regression(X, y, lr)
            k = ind.size
            # Unbiased residual variance (divide by N - k, not N).
            denom = max(ensemble_size - k, 1)
            var_res = float(err @ err) / denom
            D_diag[i] = 1.0 / max(var_res, var_floor)
            L[i, ind] = -coef

        if sparse_path:
            L = csr_matrix(L)
            Dm = diags(D_diag, format="csr")
            return (L.T @ Dm @ L).tocsr()
        return L.T @ (D_diag[:, None] * L)

    # ------------------------------------------------------------------
    # Woodbury action of S^{-1} in observation space
    # ------------------------------------------------------------------
    def apply_Sinv(self, H, Rinv_diag, V, Binv, A_solver=None):
        """Return S^{-1} V using Woodbury, where
            S^{-1} = R^{-1} - R^{-1} H A^{-1} H^T R^{-1},
            A      = B^{-1} + H^T R^{-1} H.
        ``V`` is (m, k). ``Rinv_diag`` is the 1-D diagonal of R^{-1}.
        ``A_solver`` optionally provides a prefactorized solve for A.
        Never forms an m x m matrix.
        """
        RinvV = Rinv_diag[:, None] * V                  # (m, k)
        HtRinvV = H.T @ RinvV                            # (n, k)
        HtRinvV = (HtRinvV.toarray() if hasattr(HtRinvV, "toarray")
                   else np.asarray(HtRinvV))
        if A_solver is not None:
            Z = A_solver(HtRinvV)
        elif issparse(Binv):
            Z = spsolve(self._A_sparse.tocsc(), HtRinvV)
            if Z.ndim == 1:
                Z = Z[:, None]
        else:
            Z = np.linalg.solve(self._A_dense, HtRinvV)
        HZ = H @ Z                                       # (m, k)
        HZ = HZ.toarray() if hasattr(HZ, "toarray") else np.asarray(HZ)
        return RinvV - Rinv_diag[:, None] * HZ

    # ------------------------------------------------------------------
    # Assimilation
    # ------------------------------------------------------------------
    def perform_assimilation(self, background, observation):
        """Analysis via state-space B^{-1} (modified Cholesky) and the
        Woodbury / information-form action of S^{-1}.

        Default (information form):
            A = B^{-1} + H^T R^{-1} H,
            X_a = X_b + A^{-1} H^T R^{-1} (Ys - H X_b).
        Woodbury form (use_woodbury=True): equivalent, applies S^{-1}
        explicitly in observation space; useful when m << n.
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        n, ensemble_size = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)
        # Keep H sparse if it is; only densify in the small-n branch.
        from ..observation.noise import SPARSE_THRESHOLD
        sparse_path = n >= SPARSE_THRESHOLD
        if not issparse(H):
            H = np.asarray(H)

        m = H.shape[0]

        # Perturbed observations and full innovations (used in the update).
        Ys = y[:, None] + observation.noise.sample_many_legacy(ensemble_size)
        D = Ys - HXb                                     # (m, N)

        # Background deviations and state-space precision B^{-1}.
        xb = np.mean(Xb, axis=1)
        DX = Xb - xb[:, None]
        Binv = self.get_background_precision(DX)

        # R^{-1} as a 1-D diagonal (never densified).
        if hasattr(observation.noise, "R_inv_diag"):
            Rinv_diag = np.asarray(observation.noise.R_inv_diag, dtype=float)
        else:
            R = observation.get_data_error_covariance()
            Rinv_diag = 1.0 / np.diag(np.asarray(R))

        # Assemble A = B^{-1} + H^T R^{-1} H  (state space, sparse SPD banded).
        if sparse_path or issparse(Binv) or issparse(H):
            Hs = H if issparse(H) else csr_matrix(H)
            Rinv_s = diags(Rinv_diag, format="csr")
            HtRinvH = (Hs.T @ Rinv_s @ Hs)
            Binv_s = Binv if issparse(Binv) else csr_matrix(Binv)
            A = (Binv_s + HtRinvH).tocsc()
            HtRinvD = Hs.T @ (Rinv_diag[:, None] * D)
            HtRinvD = (HtRinvD.toarray() if hasattr(HtRinvD, "toarray")
                       else np.asarray(HtRinvD))
            self._A_sparse = A
            if self.use_woodbury:
                A_solver = factorized(A)              # reuse for S^{-1} action
                U = self.apply_Sinv(Hs, Rinv_diag, D, Binv, A_solver=A_solver)
                # K D = B H^T U ; apply B by solving B^{-1} z = H^T U.
                W = Hs.T @ U
                W = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
                Z = spsolve(Binv_s.tocsc(), W)
                if Z.ndim == 1:
                    Z = Z[:, None]
                self.Xa = Xb + Z
            else:
                # Information form: single solve A z = H^T R^{-1} D.
                Z = spsolve(A, HtRinvD)
                if Z.ndim == 1:
                    Z = Z[:, None]
                self.Xa = Xb + Z
        else:
            # Dense small-n path.
            Hd = H
            HtRinvH = Hd.T @ (Rinv_diag[:, None] * Hd)
            Binv_d = Binv.toarray() if issparse(Binv) else Binv
            A = Binv_d + HtRinvH
            self._A_dense = A
            HtRinvD = Hd.T @ (Rinv_diag[:, None] * D)
            if self.use_woodbury:
                U = self.apply_Sinv(Hd, Rinv_diag, D, Binv_d)
                W = Hd.T @ U
                Z = np.linalg.solve(Binv_d, W)
                self.Xa = Xb + Z
            else:
                Z = np.linalg.solve(A, HtRinvD)
                self.Xa = Xb + Z
        return self.Xa

    def get_analysis_state(self):
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        return self.Xa

    def get_error_covariance(self):
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa