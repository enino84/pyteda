# -*- coding: utf-8 -*-
"""
EnKF with precision-space shrinkage estimator.

The estimator is a convex combination of two precisions, with a
trace-matching scale factor on the structured target:

    B̂⁻¹_shrunk(α) = α · (c · L^T D L)  +  (1-α) · W W^T,
            c = tr(W W^T) / tr(L^T D L)

where L^T D L is the modified-Cholesky precision target (sparse banded)
and W = √(N-1) U_k Σ_k⁻¹ is the factorised form of the truncated-SVD
pseudo-inverse of P^b (low-rank, n×k with k ≤ N-1). The scale factor
``c`` brings B_MC into the same trace magnitude as B_SVD; this is the
precision-space analogue of how the classical Ledoit–Wolf estimator
matches tr(μ·I) to tr(P^b). Without ``c``, the MSE criterion is biased
toward the SVD term because the squared Frobenius distance is dominated
by an artificial trace mismatch.

Four different criteria for α are supported through the ``criterion``
constructor argument:

  - ``"heuristic"``  α = γ/(1+γ) · min(1, N/n) with γ = tr(W^T B_MC W)/n.
                     This is the original, ad-hoc TEDA formula. Kept for
                     backwards compatibility.
  - ``"mse"``        Frobenius mean-squared error, the precision-space
                     analogue of Ledoit–Wolf:  α_MSE = V̂ / (V̂ + Ŝ).
  - ``"stein"``      Closed-form solution under Stein's loss.
                     Note: Stein's loss is scale-invariant, so this
                     criterion does not actually depend on ``c``; ``c``
                     is applied anyway for consistency of the estimator.
  - ``"da"``         Data-assimilation-aware: minimises tr(A(α)) where
                     A(α) = (B̂⁻¹_shrunk(α) + H^T R⁻¹ H)⁻¹, by closed-form
                     linearisation around α = 1/2.

All four variants share the same Woodbury solve for the analysis update.
See the companion paper for full derivations.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .analysis_core import Analysis
from .registry import register_analysis


# Valid criterion names exposed by the filter.
VALID_CRITERIA = {"heuristic", "mse", "stein", "da"}


@register_analysis("enkf-shrinkage-binv")
@register_analysis("enkf-shrinkage-binv-mse")
@register_analysis("enkf-shrinkage-binv-stein")
@register_analysis("enkf-shrinkage-binv-da")
@register_analysis("enkf-shrinkage-precision")  # legacy alias
class AnalysisEnKFShrinkageBinv(Analysis):
    """EnKF with shrinkage of the precision matrix B⁻¹.

    Combines a modified-Cholesky structured target with the truncated-SVD
    pseudo-inverse of the sample covariance. Four α-criteria are
    available via the ``criterion`` argument: ``heuristic``, ``mse``,
    ``stein``, ``da``.

    Parameters
    ----------
    model : object
        Numerical model exposing ``get_pre(i, r)`` for the modified
        Cholesky neighborhood.
    r : int, optional (default 1)
        Radius of influence for the modified Cholesky decomposition.
    criterion : str, optional (default ``"mse"``)
        How α is computed. One of ``"heuristic"``, ``"mse"``, ``"stein"``,
        ``"da"``.
    regularization_factor : float, optional (default 0.01)
        Ridge regularisation in the modified Cholesky fits.
    rtol_pseudo_inverse : float, optional (default 0.01)
        Relative tolerance for truncating singular values when building W.
    """

    def __init__(self, model, r=1, criterion="mse",
                 regularization_factor=0.01,
                 rtol_pseudo_inverse=0.01,
                 **kwargs):
        self.model = model
        self.r = r
        if criterion not in VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {sorted(VALID_CRITERIA)}; "
                f"got {criterion!r}.")
        self.criterion = criterion
        self.regularization_factor = float(regularization_factor)
        self.rtol_pseudo_inverse = float(rtol_pseudo_inverse)

    # ------------------------------------------------------------------
    #  Building blocks (target precision, SVD pseudo-inverse factor)
    # ------------------------------------------------------------------

    def get_target_precision_matrix(self, DX, regularization_factor=None):
        """Modified-Cholesky target precision L^T D L.

        Sparse-aware: for n >= SPARSE_THRESHOLD builds L sparse (each
        row has at most r+1 non-zeros) and D diagonal, returning a
        scipy.sparse CSR matrix. For small problems uses the original
        dense path.
        """
        if regularization_factor is None:
            regularization_factor = self.regularization_factor
        n, N = DX.shape
        lr = Ridge(fit_intercept=False, alpha=regularization_factor)

        # Sanitise DX defensively: if a previous filter in the benchmark
        # diverged it can leave NaN/Inf in the ensemble, which would
        # propagate through Ridge into NaN coefficients.
        if not np.all(np.isfinite(DX)):
            DX = np.nan_to_num(DX, nan=0.0, posinf=0.0, neginf=0.0)

        # Floor for the innovation variances so 1/var doesn't explode
        # when a row of DX is constant (degenerate ensemble).
        VAR_FLOOR = 1e-12

        from ..observation.noise import SPARSE_THRESHOLD
        sparse_path = n >= SPARSE_THRESHOLD

        if sparse_path:
            from scipy.sparse import lil_matrix, diags, csr_matrix
            L = lil_matrix((n, n), dtype=float)
            L.setdiag(1.0)
            D_diag = np.empty(n, dtype=float)
            D_diag[0] = 1.0 / max(np.var(DX[0, :]), VAR_FLOOR)
        else:
            L = np.eye(n)
            D = np.zeros((n, n))
            D[0, 0] = 1.0 / max(np.var(DX[0, :]), VAR_FLOOR)

        for i in range(1, n):
            ind_prede = self.model.get_pre(i, self.r)
            y = DX[i, :]
            if len(ind_prede) == 0:
                if sparse_path:
                    D_diag[i] = 1.0 / max(np.var(y), VAR_FLOOR)
                else:
                    D[i, i] = 1.0 / max(np.var(y), VAR_FLOOR)
                continue
            X = DX[ind_prede, :].T
            lr_fit = lr.fit(X, y)
            err_i = y - lr_fit.predict(X)
            if sparse_path:
                D_diag[i] = 1.0 / max(np.var(err_i), VAR_FLOOR)
                L[i, ind_prede] = -lr_fit.coef_
            else:
                D[i, i] = 1.0 / max(np.var(err_i), VAR_FLOOR)
                L[i, ind_prede] = -lr_fit.coef_

        if sparse_path:
            L_csr = csr_matrix(L)
            D_sp = diags(D_diag, format="csr")
            return (L_csr.T @ D_sp @ L_csr).tocsr()
        return L.T @ (D @ L)

    def get_pseudo_inverse_factor(self, DX, rtol_pseudo_inverse=None):
        """Factor W such that B̂⁻¹_SVD = W W^T.

        Uses the thin SVD of ΔX. Returns
            W = √(N-1) · U_k · diag(1/σ_e)   ∈ R^(n × k)
        and the retained singular values σ_e.
        """
        if rtol_pseudo_inverse is None:
            rtol_pseudo_inverse = self.rtol_pseudo_inverse
        n, N = DX.shape
        U, s, _ = np.linalg.svd(DX, full_matrices=False)
        if s.size == 0 or s[0] == 0:
            return np.zeros((n, 0)), np.zeros(0)
        keep = s / s[0] > rtol_pseudo_inverse
        Uk = U[:, keep]
        sk = s[keep]
        W = np.sqrt(N - 1.0) * Uk * (1.0 / sk)[None, :]
        return W, sk

    # ------------------------------------------------------------------
    #  Cheap shared quantities used by the four α criteria
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_trace_normalization(Binv_target, sigma_kept, N):
        """Scale ``Binv_target`` so its trace matches that of B_SVD.

        Returns ``(Binv_target_scaled, c)`` with
            c = tr(B_SVD) / tr(B_MC)
              = (N-1) · Σ_e σ_e⁻²   /   tr(L^T D L).

        This is the precision-space analogue of the trace-matching that
        the classical Ledoit–Wolf estimator hides inside the choice of
        target μ·I (with μ = tr(P^b)/n).  Without it, the Frobenius MSE
        criterion is dominated by an artificial scale mismatch between
        B_MC and B_SVD and converges to ``α ≈ 0``.

        For the Stein and DA criteria, the scaling is mathematically
        immaterial (Stein is scale-invariant; DA's H^T R⁻¹ H term sets
        an absolute reference), but we apply it uniformly so the four
        criteria optimise *the same* shrunken estimator.
        """
        from scipy.sparse import issparse

        if sigma_kept.size == 0:
            return Binv_target, 1.0

        tr_Bsvd = float((N - 1.0) * (1.0 / (sigma_kept ** 2)).sum())
        if issparse(Binv_target):
            tr_Bmc = float(Binv_target.diagonal().sum())
        else:
            tr_Bmc = float(np.trace(Binv_target))

        if tr_Bmc <= 0.0:
            return Binv_target, 1.0
        c = tr_Bsvd / tr_Bmc
        return c * Binv_target, c

    @staticmethod
    def _shared_traces(W, Binv_target, sigma_kept, N, n):
        """Pre-compute traces that appear in more than one criterion.

        Returns a dict with:
            tr_Bsvd     = (N-1) Σ 1/σ_e²
            tr_Bsvd2    = (N-1)² Σ 1/σ_e⁴
            tr_Bmc      = tr(B_MC)
            tr_Bmc2     = ||B_MC||_F²
            tr_Bmc_Bsvd = tr(W^T B_MC W)
            logdet_Bsvd = -2 Σ log σ_e + k log(N-1)
            BW          = B_MC @ W
        """
        from scipy.sparse import issparse
        k = W.shape[1]

        if sigma_kept.size > 0:
            inv_s2 = 1.0 / (sigma_kept ** 2)
            tr_Bsvd = float((N - 1.0) * inv_s2.sum())
            tr_Bsvd2 = float(((N - 1.0) ** 2) * (inv_s2 ** 2).sum())
            logdet_Bsvd = float(-2.0 * np.log(sigma_kept).sum()
                                + k * np.log(N - 1.0))
        else:
            tr_Bsvd = 0.0
            tr_Bsvd2 = 0.0
            logdet_Bsvd = 0.0

        if issparse(Binv_target):
            tr_Bmc = float(Binv_target.diagonal().sum())
            B_data = Binv_target.tocsr().data
            tr_Bmc2 = float((B_data ** 2).sum())
        else:
            tr_Bmc = float(np.trace(Binv_target))
            tr_Bmc2 = float((Binv_target ** 2).sum())

        if k > 0:
            BW = Binv_target @ W
            if hasattr(BW, "toarray"):
                BW = BW.toarray()
            BW = np.asarray(BW)
            tr_Bmc_Bsvd = float(np.einsum("ij,ij->", W, BW))
        else:
            BW = np.zeros((Binv_target.shape[0], 0))
            tr_Bmc_Bsvd = 0.0

        return dict(
            tr_Bsvd=tr_Bsvd, tr_Bsvd2=tr_Bsvd2,
            tr_Bmc=tr_Bmc, tr_Bmc2=tr_Bmc2,
            tr_Bmc_Bsvd=tr_Bmc_Bsvd,
            logdet_Bsvd=logdet_Bsvd,
            BW=BW,
        )

    @staticmethod
    def _clip01(alpha):
        return float(min(1.0, max(0.0, alpha)))

    # ------------------------------------------------------------------
    #  α formulas — one per criterion
    # ------------------------------------------------------------------

    @staticmethod
    def _alpha_heuristic(traces, sigma_kept, N, n):
        """Original heuristic γ/(1+γ) · min(1, N/n).

        γ = tr(B_MC · B_SVD) / n.
        """
        gamma = traces["tr_Bmc_Bsvd"] / max(n, 1)
        if gamma <= 0:
            return 1.0
        return AnalysisEnKFShrinkageBinv._clip01(
            (gamma / (1.0 + gamma)) * min(1.0, N / n)
        )

    @staticmethod
    def _alpha_mse(traces, sigma_kept, N, n):
        """Criterion A (Frobenius MSE): α_A = V̂ / (V̂ + Ŝ).

        Plug-in estimators:
            V̂ = (1/(N-1)) · (||B_SVD||_F² - (1/N)·tr(B_SVD)²)
            Ŝ = ||B_MC - B_SVD||_F²
              = ||B_MC||_F² + ||B_SVD||_F² - 2 tr(B_MC · B_SVD)
        """
        Vhat = max(
            (traces["tr_Bsvd2"]
             - (traces["tr_Bsvd"] ** 2) / N) / max(N - 1, 1),
            0.0,
        )
        Shat = max(
            traces["tr_Bmc2"] + traces["tr_Bsvd2"]
            - 2.0 * traces["tr_Bmc_Bsvd"],
            0.0,
        )
        denom = Vhat + Shat
        if denom <= 0:
            return 1.0
        return AnalysisEnKFShrinkageBinv._clip01(Vhat / denom)

    def _alpha_stein(self, traces, sigma_kept, N, n,
                     Binv_target, W):
        """Criterion B (Stein loss).

        Closed-form approximation from the paper:
            α_B ≈ 1/2 + (tr(B_MC B_SVD⁻¹) - n - log det(B_MC B_SVD⁻¹))
                       /  (2 · ||B_MC - B_SVD||²_F)

        Computed in the SVD subspace where everything is k×k.
        """
        k = W.shape[1]
        if k == 0:
            return 1.0

        # tr(B_MC · B_SVD⁻¹) and log det(B_MC · B_SVD⁻¹) on the rank-k
        # subspace. From W = √(N-1) U_k diag(1/σ),
        #   U_k = W · diag(σ) / √(N-1)
        # so U_k^T B_MC U_k = (1/(N-1)) · diag(σ) · W^T B_MC W · diag(σ).
        # And B_SVD⁻¹|_sub = (1/(N-1)) U_k Σ² U_k^T → tr identity below.
        WtBW = (W.T @ traces["BW"])                       # (k, k)
        UtBU = WtBW * np.outer(sigma_kept, sigma_kept) / (N - 1.0)

        # tr(B_MC · B_SVD⁻¹) on subspace:
        # = tr( B_MC|_sub · B_SVD⁻¹|_sub )
        # B_SVD⁻¹|_sub  in U_k basis is (1/(N-1)) · Σ²
        tr_Bmc_BsvdInv = float(
            np.trace(UtBU @ np.diag(sigma_kept ** 2) / (N - 1.0))
        )

        # log det B_MC|_sub  &  log det B_SVD|_sub
        sign, logdet_UtBU = np.linalg.slogdet(UtBU)
        if sign <= 0:
            # numerical fallback
            return self._alpha_mse(traces, sigma_kept, N, n)
        logdet_Bmc_sub = float(logdet_UtBU)
        logdet_Bsvd_sub = traces["logdet_Bsvd"]
        logdet_diff = logdet_Bmc_sub - logdet_Bsvd_sub

        denom_F = max(
            traces["tr_Bmc2"] + traces["tr_Bsvd2"]
            - 2.0 * traces["tr_Bmc_Bsvd"],
            1e-30,
        )
        alpha_B = 0.5 + (tr_Bmc_BsvdInv - k - logdet_diff) / (2.0 * denom_F)
        return self._clip01(alpha_B)

    @staticmethod
    def _alpha_da_closed(traces, sigma_kept, N, n, M0_solver_fn,
                        Binv_target, W, HtRinvH):
        """Criterion C (DA-aware), closed-form linearisation around α=1/2.

        α_C ≈ 1/2 + tr[(B_MC - B_SVD) M_0⁻²] /
                    (2 · tr[M_0⁻¹ (B_MC-B_SVD) M_0⁻² (B_MC-B_SVD)])

        with M_0 = 1/2 (B_MC + B_SVD) + H^T R⁻¹ H, M_0 ≈ A_0|_{α=1/2}.

        Hutchinson estimator for traces that cannot be reduced to rank-k.
        """
        n_state = Binv_target.shape[0]
        k = W.shape[1]

        # tr(M_0⁻¹ B_SVD M_0⁻¹) = ||M_0⁻¹ W||_F²
        if k > 0:
            M0invW = M0_solver_fn(W)
            t1 = float((M0invW ** 2).sum())
        else:
            t1 = 0.0

        # Hutchinson probes (deterministic seed for reproducibility)
        rng = np.random.default_rng(0)
        n_probes = 10
        Z = rng.choice([-1.0, 1.0], size=(n_state, n_probes))

        # tr(M_0⁻¹ B_MC M_0⁻¹) ≈ (1/m) Σ z^T M_0⁻¹ B_MC M_0⁻¹ z
        M0invZ = M0_solver_fn(Z)
        BMC_M0invZ = Binv_target @ M0invZ
        if hasattr(BMC_M0invZ, "toarray"):
            BMC_M0invZ = BMC_M0invZ.toarray()
        M0inv_BMC_M0invZ = M0_solver_fn(np.asarray(BMC_M0invZ))
        t2 = float(
            np.einsum("ij,ij->", Z, M0inv_BMC_M0invZ) / n_probes
        )

        numerator = t2 - t1

        # Denominator: 2 · tr[M_0⁻¹ Δ M_0⁻² Δ]  with Δ = B_MC - B_SVD
        WtZ = W.T @ Z
        BMCZ = Binv_target @ Z
        if hasattr(BMCZ, "toarray"):
            BMCZ = BMCZ.toarray()
        Delta_Z = np.asarray(BMCZ) - W @ WtZ
        u = M0_solver_fn(Delta_Z)
        Wtu = W.T @ u
        BMCu = Binv_target @ u
        if hasattr(BMCu, "toarray"):
            BMCu = BMCu.toarray()
        Delta_u = np.asarray(BMCu) - W @ Wtu
        M0inv_Delta_u = M0_solver_fn(Delta_u)
        denominator = 2.0 * float(
            np.einsum("ij,ij->", Z, M0inv_Delta_u) / n_probes
        )

        if abs(denominator) < 1e-30:
            return 0.5
        alpha_C = 0.5 + numerator / denominator
        return AnalysisEnKFShrinkageBinv._clip01(alpha_C)

    # ------------------------------------------------------------------
    #  Main analysis step
    # ------------------------------------------------------------------

    def perform_assimilation(self, background, observation):
        """One analysis step with shrinkage of the precision matrix.

        Shared computation (SVD, modified Cholesky, traces) runs first.
        The chosen criterion picks α. The analysis uses Woodbury with
        sparse banded A_0 and skinny factor √(1-α) · W.
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        n, ensemble_size = Xb.shape
        N = ensemble_size

        from ._obs_utils import linearize_at_mean
        from ..observation.noise import SPARSE_THRESHOLD
        H, HXb = linearize_at_mean(observation, Xb)

        Ys = y[:, None] + observation.noise.sample_many_legacy(N)
        Dinn = Ys - HXb
        xb = np.mean(Xb, axis=1)
        DX = Xb - xb[:, None]

        if n >= SPARSE_THRESHOLD:
            self._assimilate_sparse(Xb, DX, H, Dinn, observation, N, n)
        else:
            self._assimilate_dense(Xb, DX, H, Dinn, observation, N, n)
        return self.Xa

    # ----- sparse path (high-dim problems) ------------------------------

    def _assimilate_sparse(self, Xb, DX, H, Dinn, observation, N, n):
        from scipy.sparse import csr_matrix, csc_matrix, diags
        from ._woodbury_solver import woodbury_solve, sparse_lu_solver

        Binv_target = self.get_target_precision_matrix(DX)
        if not hasattr(Binv_target, "tocsr"):
            Binv_target = csr_matrix(Binv_target)
        W, sigma_kept = self.get_pseudo_inverse_factor(DX)
        # Trace-matching normalisation: compute α on a *trace-aligned*
        # copy of B_MC so the Frobenius MSE balance is meaningful. The
        # actual analysis solve below keeps the original B_MC; the
        # scaling only enters through ``alpha``. See class docstring.
        Binv_target_scaled, scale_c = self._apply_trace_normalization(
            Binv_target, sigma_kept, N,
        )
        self.scale_c_ = float(scale_c)
        traces = self._shared_traces(W, Binv_target_scaled, sigma_kept, N, n)

        # H^T R^{-1} H
        if hasattr(observation.noise, "R_inv_diag"):
            r_inv_diag = observation.noise.R_inv_diag
            Rinv = diags(r_inv_diag, format="csr")
            HtRinvH = (H.T @ Rinv @ H).tocsc()
            HtRinvD = H.T @ (r_inv_diag[:, None] * Dinn)
            if hasattr(HtRinvD, "toarray"):
                HtRinvD = HtRinvD.toarray()
        else:
            R = observation.get_data_error_covariance()
            Rinv = np.linalg.inv(R)
            HtRinvH_dense = H.T @ (Rinv @ H)
            if hasattr(HtRinvH_dense, "toarray"):
                HtRinvH_dense = HtRinvH_dense.toarray()
            HtRinvH = csc_matrix(HtRinvH_dense)
            HtRinvD = H.T @ (Rinv @ Dinn)
            if hasattr(HtRinvD, "toarray"):
                HtRinvD = HtRinvD.toarray()

        alpha = self._dispatch_alpha(
            traces, sigma_kept, N, n, Binv_target_scaled, W, HtRinvH,
            Binv_target_orig=Binv_target,
        )
        self.alpha_ = float(alpha)

        # Solve uses the *original* B_MC (without c-scaling). The scaling
        # only enters α to fix the trace-mismatch in the MSE criterion;
        # the actual shrunken estimator is α·B_MC + (1-α)·B_SVD.
        A0 = (alpha * Binv_target + HtRinvH).tocsc()
        A0_solver = sparse_lu_solver(A0)
        if W.shape[1] > 0 and (1.0 - alpha) > 0:
            Z = woodbury_solve(
                A0_solver, [W], [1.0 - alpha], np.asarray(HtRinvD),
            )
        else:
            Z = A0_solver(np.asarray(HtRinvD))
        self.Xa = Xb + Z

    # ----- dense path (small problems) ----------------------------------

    def _assimilate_dense(self, Xb, DX, H, Dinn, observation, N, n):
        from scipy.sparse import issparse
        Binv_target = self.get_target_precision_matrix(DX)
        if issparse(Binv_target):
            Binv_target = Binv_target.toarray()
        Binv_target = np.asarray(Binv_target)

        W, sigma_kept = self.get_pseudo_inverse_factor(DX)
        # Trace-matching normalisation (see class docstring): compute α
        # on a trace-aligned copy; keep the original B_MC for the solve.
        Binv_target_scaled, scale_c = self._apply_trace_normalization(
            Binv_target, sigma_kept, N,
        )
        self.scale_c_ = float(scale_c)
        Binv_target_scaled = np.asarray(Binv_target_scaled)
        traces = self._shared_traces(W, Binv_target_scaled, sigma_kept, N, n)

        if hasattr(observation.noise, "R_inv_diag"):
            r_inv_diag = observation.noise.R_inv_diag
            Rinv = np.diag(r_inv_diag)
            HtRinvH = H.T @ Rinv @ H
            if hasattr(HtRinvH, "toarray"):
                HtRinvH = HtRinvH.toarray()
            HtRinvD = H.T @ (r_inv_diag[:, None] * Dinn)
            if hasattr(HtRinvD, "toarray"):
                HtRinvD = HtRinvD.toarray()
        else:
            R = observation.get_data_error_covariance()
            Rinv = np.linalg.inv(R)
            HtRinvH = H.T @ (Rinv @ H)
            HtRinvD = H.T @ (Rinv @ Dinn)
            if hasattr(HtRinvH, "toarray"):
                HtRinvH = HtRinvH.toarray()
            if hasattr(HtRinvD, "toarray"):
                HtRinvD = HtRinvD.toarray()
        HtRinvH = np.asarray(HtRinvH)

        alpha = self._dispatch_alpha(
            traces, sigma_kept, N, n, Binv_target_scaled, W, HtRinvH,
            Binv_target_orig=Binv_target,
        )
        self.alpha_ = float(alpha)

        if W.shape[1] > 0:
            Pseu = W @ W.T
        else:
            Pseu = np.zeros((n, n))
        # Solve uses the *original* B_MC (without c-scaling).
        Binv_shrunk = alpha * Binv_target + (1.0 - alpha) * Pseu
        IN = Binv_shrunk + HtRinvH
        Z = np.linalg.solve(np.asarray(IN), np.asarray(HtRinvD))
        self.Xa = Xb + Z

    # ----- dispatch ----------------------------------------------------

    def _dispatch_alpha(self, traces, sigma_kept, N, n,
                        Binv_target_scaled, W, HtRinvH,
                        Binv_target_orig=None):
        """Pick α according to ``self.criterion``.

        ``Binv_target_scaled`` is the trace-aligned target (multiplied
        by c so its trace matches B_SVD); MSE and Stein use it to keep
        the criterion well-scaled. ``Binv_target_orig`` (default: the
        scaled one) is the unscaled B_MC; the DA criterion needs it
        because the H^T R⁻¹ H term defines an absolute reference scale
        that should not be matched to anything.
        """
        if Binv_target_orig is None:
            Binv_target_orig = Binv_target_scaled
        crit = self.criterion
        if crit == "heuristic":
            return self._alpha_heuristic(traces, sigma_kept, N, n)
        if crit == "mse":
            return self._alpha_mse(traces, sigma_kept, N, n)
        if crit == "stein":
            return self._alpha_stein(traces, sigma_kept, N, n,
                                     Binv_target_scaled, W)
        if crit == "da":
            # DA criterion uses the original (unscaled) B_MC so the
            # H^T R⁻¹ H term sets a meaningful absolute reference for
            # the curvature of tr(A(α)).
            from scipy.sparse import issparse
            A0_half = 0.5 * Binv_target_orig + HtRinvH
            if issparse(A0_half):
                from ._woodbury_solver import sparse_lu_solver
                M0_solver = sparse_lu_solver(A0_half.tocsc())
            else:
                from ._woodbury_solver import dense_lu_solver
                M0_solver = dense_lu_solver(np.asarray(A0_half))
            return self._alpha_da_closed(
                traces, sigma_kept, N, n, M0_solver,
                Binv_target_orig, W, HtRinvH,
            )
        raise ValueError(f"Unknown criterion {crit!r}")

    # ------------------------------------------------------------------
    #  Standard Analysis accessors
    # ------------------------------------------------------------------

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


# Backward-compatibility alias for code that imports the old class name.
AnalysisEnKFShrinkagePrecision = AnalysisEnKFShrinkageBinv
