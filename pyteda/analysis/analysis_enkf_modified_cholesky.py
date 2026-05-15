# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge

from .analysis_core import Analysis
from .registry import register_analysis

@register_analysis("enkf-modified-cholesky")
class AnalysisEnKFModifiedCholesky(Analysis):
    """Analysis EnKF Modified Cholesky decomposition
    
    Attributes:
        model (Model object): An object that has all the methods and attributes of the model
        r (int): Value used in the process of removing correlations

    Methods:
        get_precision_matrix(DX, regularization_factor=0.01): Returns the computed precision matrix
        perform_assimilation(background, observation): Perform assimilation step given background and observations
        get_analysis_state(): Returns the computed column mean of ensemble Xa
        get_ensemble(): Returns ensemble Xa
        get_error_covariance(): Returns the computed covariance matrix of the ensemble Xa
        inflate_ensemble(inflation_factor): Computes new ensemble Xa given the inflation factor
    """

    def __init__(self, model, r=1, alpha: float = 0.01, **kwargs):
        """
        Initialize an instance of AnalysisEnKFModifiedCholesky.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
            r (int, dict, or ndarray, optional): Localisation radius used to
                pick predecessors via ``model.get_pre(i, r)``. Same int/dict/
                ndarray dispatch as LETKF/LEnKF.
            alpha (float, optional): Ridge regression regularisation passed to
                ``sklearn.linear_model.Ridge``. Default 0.01.
        """
        self.model = model
        self.r = r
        self.alpha = float(alpha)

    def get_precision_matrix(self, DX, regularization_factor=None):
        """
        Compute the precision matrix B^{-1} ≈ L^T D L via the modified
        Cholesky decomposition (per-component Ridge regressions of each
        component on its predecessors).

        For small problems (n < SPARSE_THRESHOLD) returns a dense ndarray.
        For large problems uses scipy.sparse internally — L is triangular
        with only ``r+1`` non-zeros per row by construction, and D is
        diagonal, so the dense storage cost would be n² for what is
        intrinsically O(n*r) information. Returns a sparse CSR matrix
        in that case.

        Parameters:
            DX (ndarray): Deviation matrix, shape (n, N_ens).
            regularization_factor (float, optional): Override for Ridge
                alpha. Default: self.alpha set at construction.

        Returns:
            Binv : ndarray (small n) or scipy.sparse.csr_matrix (large n).
        """
        alpha = float(regularization_factor) if regularization_factor is not None else self.alpha
        n, ensemble_size = DX.shape
        lr = Ridge(fit_intercept=False, alpha=alpha)

        # Decide format. Below threshold dense is faster and tested.
        from ..observation.noise import SPARSE_THRESHOLD
        sparse_path = n >= SPARSE_THRESHOLD

        if sparse_path:
            from scipy.sparse import lil_matrix, diags, csr_matrix
            # L is built row-wise; lil is the right scratch format.
            L = lil_matrix((n, n), dtype=float)
            L.setdiag(1.0)
            D_diag = np.empty(n, dtype=float)
        else:
            L = np.eye(n)
            D_dense = np.zeros((n, n))
            D_dense[0, 0] = 1.0 / np.var(DX[0, :])

        # First component has no predecessors — initialise directly.
        if sparse_path:
            D_diag[0] = 1.0 / np.var(DX[0, :])

        for i in range(1, n):
            ind_prede = self.model.get_pre(i, self.r)
            y = DX[i, :]
            if len(ind_prede) == 0:
                # No predecessors: residual is the deviation itself.
                if sparse_path:
                    D_diag[i] = 1.0 / np.var(y)
                else:
                    D_dense[i, i] = 1.0 / np.var(y)
                continue
            X = DX[ind_prede, :].T
            lr_fit = lr.fit(X, y)
            err_i = y - lr_fit.predict(X)
            if sparse_path:
                D_diag[i] = 1.0 / np.var(err_i)
                # lil supports fancy column indexing on a single row.
                L[i, ind_prede] = -lr_fit.coef_
            else:
                D_dense[i, i] = 1.0 / np.var(err_i)
                L[i, ind_prede] = -lr_fit.coef_

        if sparse_path:
            L = csr_matrix(L)
            D = diags(D_diag, format="csr")
            # Binv = L^T D L  — all sparse, result is sparse banded.
            return (L.T @ D @ L).tocsr()
        else:
            return L.T @ (D_dense @ L)

    def perform_assimilation(self, background, observation):
        """Perform assimilation step of ensemble Xa given the background and the observations.

        Parameters:
            background (Background Object): The background object defined in the class background
            observation (Observation Object): The observation object defined in the class observation

        Returns:
            Xa (Matrix): Matrix of ensemble
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        n, ensemble_size = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)

        # Sample observation noise efficiently — never materialise R.
        Ys = y[:, None] + observation.noise.sample_many_legacy(ensemble_size)
        xb = np.mean(Xb, axis=1)
        DX = Xb - np.outer(xb, np.ones(ensemble_size))

        Binv = self.get_precision_matrix(DX)
        Dinn = Ys - HXb

        # R_inv as a 1-D vector of diagonal entries — works for all
        # diagonal noise types (Isotropic, Heterogeneous, BlockDiagonal).
        # For non-diagonal R we fall back to the dense materialisation.
        if hasattr(observation.noise, "R_inv_diag"):
            r_inv_diag = observation.noise.R_inv_diag
            # H.T @ R_inv @ H computed cheaply: row-scale H by r_inv_diag.
            #   (H is sparse for high-dim setups, dense for small.)
            from ..observation.noise import SPARSE_THRESHOLD
            if n >= SPARSE_THRESHOLD:
                from scipy.sparse import diags
                Rinv_sparse = diags(r_inv_diag, format="csr")
                HtRinvH = H.T @ Rinv_sparse @ H
                HtRinvD = H.T @ (r_inv_diag[:, None] * Dinn)
            else:
                # Dense path: small enough that np.diag is fine.
                Rinv = np.diag(r_inv_diag)
                HtRinvH = H.T @ Rinv @ H
                if hasattr(HtRinvH, "toarray"):
                    HtRinvH = HtRinvH.toarray()
                HtRinvD = H.T @ (r_inv_diag[:, None] * Dinn)
                if hasattr(HtRinvD, "toarray"):
                    HtRinvD = HtRinvD.toarray()
        else:
            # General (dense) path — only used when R is genuinely dense.
            R = observation.get_data_error_covariance()
            Rinv = np.linalg.inv(R)
            HtRinvH = H.T @ (Rinv @ H)
            HtRinvD = H.T @ (Rinv @ Dinn)

        IN = Binv + HtRinvH

        # Solve IN @ Z = H^T R^{-1} D using the right backend.
        from scipy.sparse import issparse
        from scipy.sparse.linalg import spsolve
        if issparse(IN):
            # spsolve works column-by-column for multi-RHS; convert to
            # dense for the right-hand side which is always (n, N_ens).
            rhs = HtRinvD.toarray() if hasattr(HtRinvD, "toarray") else np.asarray(HtRinvD)
            # Use sparse LU for repeated solves — but here we only solve
            # once per assimilation step, so spsolve is fine.
            Z = spsolve(IN.tocsc(), rhs)
            if Z.ndim == 1:
                Z = Z[:, None]
        else:
            rhs = HtRinvD if not hasattr(HtRinvD, "toarray") else HtRinvD.toarray()
            Z = np.linalg.solve(IN, rhs)

        self.Xa = Xb + Z
        return self.Xa
        self.Xa = Xb + Z
        return self.Xa

    def get_analysis_state(self):
        """Compute column-wise mean vector of Matrix of ensemble Xa

        Parameters:
            None

        Returns:
            mean_vector (ndarray): Mean vector
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Returns ensemble Xa

        Parameters:
            None

        Returns:
            ensemble_matrix (ndarray): Ensemble matrix
        """
        return self.Xa

    def get_error_covariance(self):
        """Returns the computed covariance matrix of the ensemble Xa

        Parameters:
            None

        Returns:
            covariance_matrix (ndarray): Covariance matrix of the ensemble Xa
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Computes ensemble Xa given the inflation factor

        Parameters:
            inflation_factor (int): Double number indicating the inflation factor

        Returns:
            None
        """
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
