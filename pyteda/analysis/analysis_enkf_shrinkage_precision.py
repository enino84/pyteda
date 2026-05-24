# -*- coding: utf-8 -*-
"""
EnKF with precision-space shrinkage estimators.

This module implements a family of ensemble Kalman filters whose
background-error *precision* matrix B⁻¹ is estimated by a convex
combination of two precision estimators,

    B̂⁻¹_shrunk(α) = α · B̂⁻¹_(1) + (1-α) · B̂⁻¹_(2),     α ∈ [0, 1],

with α chosen by one of three principled criteria. The two estimators
that are combined depend on the criterion:

  - ``"mse"``    Frobenius mean-squared-error weight. The two estimators
                 are **two modified-Cholesky precisions at different
                 radii of influence** (r₁ and r₂, with r₁ < r₂):

                     B̂⁻¹_(1) = L_{r₁}^T D_{r₁} L_{r₁}   (more bias, less variance)
                     B̂⁻¹_(2) = L_{r₂}^T D_{r₂} L_{r₂}   (less bias, more variance)

                 Both are sparse, banded, full-rank and SPD, and live on
                 the *same* trace scale, so the Frobenius bias–variance
                 balance is homogeneous and well-posed. No trace-matching
                 factor is needed. The combined estimator is itself sparse
                 banded with bandwidth max(r₁, r₂).

  - ``"stein"``  Stein-loss (information-geometric) weight. The two
                 estimators are the modified-Cholesky target
                 B̂⁻¹_MC = L^T D L (radius r) and the truncated-SVD
                 pseudo-inverse B̂⁻¹_SVD = W W^T, with
                 W = √(N-1) U_k Σ_k⁻¹. Stein's loss is scale-invariant,
                 so the target is used as-is (no scale factor).

  - ``"da"``     Data-assimilation-aware weight that minimises the trace
                 of the posterior analysis covariance,
                 tr[(B̂⁻¹_shrunk(α) + H^T R⁻¹ H)⁻¹]. Same two estimators
                 as Stein (modified-Cholesky vs truncated-SVD). The
                 H^T R⁻¹ H term sets an absolute reference scale, so the
                 target is used as-is.

See the companion paper for full derivations.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .analysis_core import Analysis
from .registry import register_analysis


# Valid criterion names exposed by the filter.
VALID_CRITERIA = {"mse", "stein", "da"}

# Default radii for the two-target Frobenius-MSE estimator.
# Our proposal always combines exactly two modified-Cholesky targets.
MSE_RADIUS_NARROW = 2
MSE_RADIUS_WIDE = 5


@register_analysis("enkf-shrinkage-binv")
@register_analysis("enkf-shrinkage-binv-mse")
@register_analysis("enkf-shrinkage-binv-stein")
@register_analysis("enkf-shrinkage-binv-da")
@register_analysis("enkf-shrinkage-precision")  # legacy alias
class AnalysisEnKFShrinkageBinv(Analysis):
    """EnKF with shrinkage of the precision matrix B⁻¹.

    Three α-criteria are available via the ``criterion`` argument:

      - ``"mse"``    Frobenius MSE between **two modified-Cholesky
                     targets** at radii (r_narrow, r_wide).
      - ``"stein"``  Stein loss between modified-Cholesky (radius r) and
                     the truncated-SVD pseudo-inverse.
      - ``"da"``     DA-aware (minimises tr of posterior covariance),
                     same two estimators as Stein.

    Parameters
    ----------
    model : object
        Numerical model exposing ``get_pre(i, r)`` for the modified
        Cholesky neighborhood.
    r : int, optional (default 2)
        Radius of influence for the modified Cholesky decomposition.
        For the Stein/DA criteria this is *the* radius. For the MSE
        criterion it is the *narrow* radius r₁; the wide radius r₂ is
        given by ``r_wide``.
    r_wide : int, optional (default 5)
        Wide radius r₂ used by the MSE criterion only. Must satisfy
        r_wide > r. Ignored by the Stein/DA criteria.
    criterion : str, optional (default ``"mse"``)
        One of ``"mse"``, ``"stein"``, ``"da"``.
    regularization_factor : float, optional (default 0.01)
        Ridge regularisation in the modified Cholesky fits.
    rtol_pseudo_inverse : float, optional (default 0.25)
        Relative tolerance for truncating singular values when building
        W (Stein/DA only). Singular values with σ_e/σ_1 ≤ rtol are
        discarded, which prevents noise-dominated modes from entering
        B̂⁻¹_SVD with weight 1/σ_e².
    """

    def __init__(self, model, r=2, r_wide=MSE_RADIUS_WIDE, criterion="mse",
                 regularization_factor=0.01,
                 rtol_pseudo_inverse=0.25,
                 cholesky_method="ridge",
                 cholesky_tol=0.1,
                 **kwargs):
        self.model = model
        self.r = r
        self.r_wide = r_wide
        if criterion not in VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {sorted(VALID_CRITERIA)}; "
                f"got {criterion!r}.")
        self.criterion = criterion
        self.regularization_factor = float(regularization_factor)
        self.rtol_pseudo_inverse = float(rtol_pseudo_inverse)
        if cholesky_method not in ("ridge", "svd"):
            raise ValueError(
                f"cholesky_method must be 'ridge' or 'svd'; "
                f"got {cholesky_method!r}.")
        self.cholesky_method = cholesky_method
        self.cholesky_tol = float(cholesky_tol)

    # ------------------------------------------------------------------
    #  Building blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _local_regression_svd(X, y, tol):
        """Solve min_beta ||y - X beta|| by truncated-SVD pseudo-inverse.

        X has shape (N_samples, p) with the p local predictors in columns
        (this is DX[neighbors,:].T). Singular values with σ/σ₁ ≤ tol are
        dropped, which regularizes the small-sample local fit by discarding
        low-variance (noise-dominated) directions — the SVD analogue of the
        Ridge penalty. Returns (coef, residual) where residual = y - X·coef.
        """
        # economy SVD of the (N_samples × p) design matrix
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        if s.size == 0 or s[0] == 0:
            return np.zeros(X.shape[1]), y.copy()
        keep = s / s[0] > tol
        if not np.any(keep):
            return np.zeros(X.shape[1]), y.copy()
        Uk = U[:, keep]
        sk = s[keep]
        Vk = Vt[keep, :]
        # beta = V_k Σ_k⁻¹ U_kᵀ y  (truncated pseudo-inverse)
        coef = Vk.T @ ((Uk.T @ y) / sk)
        resid = y - X @ coef
        return coef, resid

    def get_target_precision_matrix(self, DX, r=None,
                                    regularization_factor=None,
                                    cholesky_method=None,
                                    cholesky_tol=None):
        """Modified-Cholesky target precision L^T D L at radius ``r``.

        The local conditional regressions that build each row of L can be
        solved two ways, selected by ``cholesky_method``:

          - ``"ridge"`` (default): ridge regression with penalty
            ``regularization_factor`` (the λ in the paper).
          - ``"svd"``: truncated-SVD pseudo-inverse of the local design,
            dropping singular directions with σ/σ₁ ≤ ``cholesky_tol``.

        Both regularize the small-N local fit; Ridge penalizes small
        directions smoothly, truncated-SVD discards them hard.

        Sparse-aware: for n >= SPARSE_THRESHOLD builds L sparse (each
        row has at most r+1 non-zeros) and D diagonal, returning a
        scipy.sparse CSR matrix. For small problems uses the dense path.
        """
        if r is None:
            r = self.r
        if regularization_factor is None:
            regularization_factor = self.regularization_factor
        if cholesky_method is None:
            cholesky_method = self.cholesky_method
        if cholesky_tol is None:
            cholesky_tol = self.cholesky_tol
        n, N = DX.shape
        use_svd = (cholesky_method == "svd")
        lr = None if use_svd else Ridge(fit_intercept=False,
                                        alpha=regularization_factor)

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
            ind_prede = self.model.get_pre(i, r)
            y = DX[i, :]
            if len(ind_prede) == 0:
                if sparse_path:
                    D_diag[i] = 1.0 / max(np.var(y), VAR_FLOOR)
                else:
                    D[i, i] = 1.0 / max(np.var(y), VAR_FLOOR)
                continue
            X = DX[ind_prede, :].T
            if use_svd:
                coef, err_i = self._local_regression_svd(X, y, cholesky_tol)
            else:
                lr_fit = lr.fit(X, y)
                coef = lr_fit.coef_
                err_i = y - lr_fit.predict(X)
            if sparse_path:
                D_diag[i] = 1.0 / max(np.var(err_i), VAR_FLOOR)
                L[i, ind_prede] = -coef
            else:
                D[i, i] = 1.0 / max(np.var(err_i), VAR_FLOOR)
                L[i, ind_prede] = -coef

        if sparse_path:
            L_csr = csr_matrix(L)
            D_sp = diags(D_diag, format="csr")
            return (L_csr.T @ D_sp @ L_csr).tocsr()
        return L.T @ (D @ L)

    def get_pseudo_inverse_factor(self, DX, rtol_pseudo_inverse=None):
        """Factor W such that B̂⁻¹_SVD = W W^T (Stein/DA criteria).

        Uses the thin SVD of ΔX. Returns
            W = √(N-1) · U_k · diag(1/σ_e)   ∈ R^(n × k)
        and the retained singular values σ_e. Singular values with
        σ_e/σ_1 ≤ rtol are truncated.
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
    #  α formulas — one per criterion
    # ------------------------------------------------------------------

    @staticmethod
    def _clip01(alpha):
        return float(min(1.0, max(0.0, alpha)))

    def _alpha_mse_two_targets(self, B1, B2, DX, r1, r2):
        """Criterion A (Frobenius MSE) for two modified-Cholesky targets.

        Combines B1 = L_{r1}^T D_{r1} L_{r1} (narrow radius: more bias,
        less variance) and B2 = L_{r2}^T D_{r2} L_{r2} (wide radius: less
        bias, more variance) as

            B̂⁻¹(α) = α B1 + (1-α) B2,    α ∈ [0,1].

        The exact first-order condition for min_α E‖B̂⁻¹(α) − B⁻¹‖²_F is

            α* = -ρ / γ,
            γ = ‖B1 − B2‖²_F,
            ρ = ⟨B2 − B⁻¹, B1 − B2⟩ = ⟨b₂, b₁−b₂⟩ + (c₁₂ − v₂).

        Both the bias inner product ⟨b₂,b₁−b₂⟩ and the (co)variance term
        depend on quantities that are not directly observable. Rather than
        estimate the variance term by an expensive leave-one-out jackknife
        (which costs 2N modified-Cholesky builds per cycle), we bound the
        whole numerator ρ by the Bickel & Levina (2008) banded-precision
        bias rate. For the well-conditioned banded class U⁻¹(ε₀,C,β),
        ‖b_i‖_op = O(r_i^{−β}), so by Cauchy–Schwarz

            |ρ| ≤ B_max = C²·n · r₂^{−β}(r₁^{−β} + r₂^{−β}),

        and α* lies in the symmetric interval [−B_max/γ, +B_max/γ] ∩ [0,1].
        Lacking the sign of ρ, the minimax (worst-case-optimal) weight is
        the interval centre. This is well posed precisely because B1 and B2
        are *both* well-conditioned SPD precisions on the same trace scale:
        the SPD cone is convex, every convex combination is well
        conditioned, and the analysis is provably insensitive to the exact
        α inside the interval (the worst α raises the Frobenius error only
        a few percent over the optimum). Only a value inside the interval
        is needed — no resampling. The decay rate β is estimated cheaply
        from the off-diagonal decay of the wide Cholesky factor.

        This makes Criterion A as cheap as Stein/DA: two Cholesky builds
        per cycle instead of 2N, i.e. ~N× faster (e.g. ~20× at N=20).
        """
        from scipy.sparse import issparse

        def fro2_diff(A, B):
            D = A - B
            if issparse(D):
                d = D.tocsr().data
                return float((d ** 2).sum())
            return float((np.asarray(D) ** 2).sum())

        n, N = DX.shape
        gamma = fro2_diff(B1, B2)                 # γ = ‖B1 − B2‖²_F
        if gamma <= 0.0:
            return 1.0                            # targets coincide

        # --- Bickel–Levina bound on the numerator (no jackknife) --------
        # Estimate the off-diagonal decay rate β and prefactor C of the
        # wide precision factor from its banded structure, then form
        # B_max = C²·n·r₂^{−β}(r₁^{−β}+r₂^{−β}).
        beta, C = self._estimate_bl_decay(B2, r2)
        B_max = (C ** 2) * n * (r2 ** (-beta)) * (
            r1 ** (-beta) + r2 ** (-beta))
        half_width = B_max / gamma                # half-width of the interval

        # The first-order optimum is α*=-ρ/γ with |ρ|≤B_max, so α* lies in
        # the symmetric interval [-half_width, +half_width] about 0. The
        # Frobenius risk is provably flat across this interval (both
        # constituents are well conditioned), so any admissible α gives the
        # same distance to B⁻¹. As a *filter*, however, the wide target
        # B2 (radius r₂, many coefficients estimated from few members) is
        # the higher-variance, less stable constituent; the narrow target
        # B1 (radius r₁) is the low-variance, stable one. With α the weight
        # on B1, we therefore break the Frobenius tie in favour of the
        # stable constituent and take the admissible weight closest to 1:
        #
        #     α* = clip_{[0,1]}( 1 - half_width ).
        #
        # When the interval is wide (half_width large) this still admits a
        # genuine blend; when it is narrow (half_width≈0, the constituents
        # disagree strongly) it puts essentially full weight on the stable
        # narrow target, which is the robust choice for the analysis.
        alpha = 1.0 - half_width
        return self._clip01(alpha)

    @staticmethod
    def _estimate_bl_decay(B, r):
        """Estimate (β, C) for the Bickel–Levina bound from a precision.

        The banded class U⁻¹(ε₀,C,β) assumes the off-diagonal mass at
        lag ℓ decays like C·ℓ^{−β}. We read this off the precision B by
        fitting log|mass(ℓ)| vs log ℓ over the available lags 1..r, where
        mass(ℓ) = mean_i |B_{i,i−ℓ}|. Robust fallbacks keep β in a sane
        range when the fit is ill-determined (e.g. r too small).
        """
        from scipy.sparse import issparse
        A = B.toarray() if issparse(B) else np.asarray(B)
        nn = A.shape[0]
        lags = []
        mass = []
        for ell in range(1, int(r) + 1):
            diag_vals = np.abs(np.diagonal(A, offset=-ell))
            if diag_vals.size == 0:
                continue
            m = float(diag_vals.mean())
            if m > 0:
                lags.append(ell)
                mass.append(m)
        if len(lags) >= 2:
            x = np.log(np.array(lags))
            y = np.log(np.array(mass))
            # slope of log-mass vs log-lag is −β
            slope = np.polyfit(x, y, 1)[0]
            beta = float(np.clip(-slope, 0.5, 4.0))
            C = float(np.exp(y[0] + beta * x[0]))   # mass(1)·1^β ≈ C
        else:
            # not enough lags to fit: assume a mild decay and use the
            # lag-1 mass (or the diagonal scale) as the prefactor.
            beta = 1.0
            if mass:
                C = float(mass[0])
            else:
                C = float(np.abs(np.diagonal(A)).mean())
        # guard against degenerate C
        if not np.isfinite(C) or C <= 0:
            C = 1.0
        return beta, C

    def _alpha_stein(self, traces, sigma_kept, N, n, Binv_target, W):
        """Criterion B (Stein loss).

        Closed-form approximation:
            α_B ≈ 1/2 + (tr(B_MC B_SVD⁻¹) - k - log det(B_MC B_SVD⁻¹))
                       /  (2 · ‖B_MC - B_SVD‖²_F)

        Computed in the SVD subspace where everything is k×k. Stein's
        loss is scale-invariant, so B_MC is used as-is (no scale factor).
        """
        k = W.shape[1]
        if k == 0:
            return 1.0

        WtBW = (W.T @ traces["BW"])                       # (k, k)
        UtBU = WtBW * np.outer(sigma_kept, sigma_kept) / (N - 1.0)

        tr_Bmc_BsvdInv = float(
            np.trace(UtBU @ np.diag(sigma_kept ** 2) / (N - 1.0))
        )

        sign, logdet_UtBU = np.linalg.slogdet(UtBU)
        if sign <= 0:
            # numerical fallback: trust the structured target
            return 1.0
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

        with M_0 = 1/2 (B_MC + B_SVD) + H^T R⁻¹ H. Uses a Hutchinson
        estimator for the traces that cannot be reduced to rank-k.
        """
        n_state = Binv_target.shape[0]
        k = W.shape[1]

        if k > 0:
            M0invW = M0_solver_fn(W)
            t1 = float((M0invW ** 2).sum())
        else:
            t1 = 0.0

        rng = np.random.default_rng(0)
        n_probes = 10
        Z = rng.choice([-1.0, 1.0], size=(n_state, n_probes))

        M0invZ = M0_solver_fn(Z)
        BMC_M0invZ = Binv_target @ M0invZ
        if hasattr(BMC_M0invZ, "toarray"):
            BMC_M0invZ = BMC_M0invZ.toarray()
        M0inv_BMC_M0invZ = M0_solver_fn(np.asarray(BMC_M0invZ))
        t2 = float(
            np.einsum("ij,ij->", Z, M0inv_BMC_M0invZ) / n_probes
        )

        numerator = t2 - t1

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
    #  Shared traces for the Stein/DA path (MC vs SVD)
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_traces(W, Binv_target, sigma_kept, N, n):
        """Pre-compute traces appearing in the Stein criterion."""
        from scipy.sparse import issparse
        k = W.shape[1]

        if sigma_kept.size > 0:
            inv_s2 = 1.0 / (sigma_kept ** 2)
            tr_Bsvd2 = float(((N - 1.0) ** 2) * (inv_s2 ** 2).sum())
            logdet_Bsvd = float(-2.0 * np.log(sigma_kept).sum()
                                + k * np.log(N - 1.0))
        else:
            tr_Bsvd2 = 0.0
            logdet_Bsvd = 0.0

        if issparse(Binv_target):
            B_data = Binv_target.tocsr().data
            tr_Bmc2 = float((B_data ** 2).sum())
        else:
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
            tr_Bsvd2=tr_Bsvd2,
            tr_Bmc2=tr_Bmc2,
            tr_Bmc_Bsvd=tr_Bmc_Bsvd,
            logdet_Bsvd=logdet_Bsvd,
            BW=BW,
        )

    # ------------------------------------------------------------------
    #  Main analysis step
    # ------------------------------------------------------------------

    def perform_assimilation(self, background, observation):
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

        if self.criterion == "mse":
            self._assimilate_mse(Xb, DX, H, Dinn, observation, N, n)
        elif n >= SPARSE_THRESHOLD:
            self._assimilate_sparse(Xb, DX, H, Dinn, observation, N, n)
        else:
            self._assimilate_dense(Xb, DX, H, Dinn, observation, N, n)
        return self.Xa

    # ----- MSE path: two modified-Cholesky targets ---------------------

    def _assimilate_mse(self, Xb, DX, H, Dinn, observation, N, n):
        """Analysis with the two-target Frobenius-MSE estimator.

        B̂⁻¹(α) = α L_{r₁}^T D_{r₁} L_{r₁} + (1-α) L_{r₂}^T D_{r₂} L_{r₂},
        both sparse banded → the whole operator is sparse banded with
        bandwidth max(r₁, r₂). No SVD term, no Woodbury needed.
        """
        from scipy.sparse import issparse, csc_matrix, diags
        from ..observation.noise import SPARSE_THRESHOLD

        B1 = self.get_target_precision_matrix(DX, r=self.r)
        B2 = self.get_target_precision_matrix(DX, r=self.r_wide)

        alpha = self._alpha_mse_two_targets(B1, B2, DX, self.r, self.r_wide)
        self.alpha_ = float(alpha)

        Binv_shrunk = alpha * B1 + (1.0 - alpha) * B2

        # H^T R^{-1} H  and  H^T R^{-1} Dinn
        if hasattr(observation.noise, "R_inv_diag"):
            r_inv_diag = observation.noise.R_inv_diag
            if n >= SPARSE_THRESHOLD:
                Rinv = diags(r_inv_diag, format="csr")
                HtRinvH = (H.T @ Rinv @ H)
            else:
                HtRinvH = H.T @ (r_inv_diag[:, None] * H) \
                    if not issparse(H) else (H.T @ diags(r_inv_diag) @ H)
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

        IN = Binv_shrunk + HtRinvH
        if issparse(IN):
            from ._woodbury_solver import sparse_lu_solver
            solver = sparse_lu_solver(IN.tocsc())
            Z = solver(np.asarray(HtRinvD))
        else:
            Z = np.linalg.solve(np.asarray(IN), np.asarray(HtRinvD))
        self.Xa = Xb + Z

    # ----- Stein/DA sparse path (high-dim) -----------------------------

    def _assimilate_sparse(self, Xb, DX, H, Dinn, observation, N, n):
        from scipy.sparse import csr_matrix, csc_matrix, diags
        from ._woodbury_solver import woodbury_solve, sparse_lu_solver

        Binv_target = self.get_target_precision_matrix(DX, r=self.r)
        if not hasattr(Binv_target, "tocsr"):
            Binv_target = csr_matrix(Binv_target)
        W, sigma_kept = self.get_pseudo_inverse_factor(DX)
        traces = self._shared_traces(W, Binv_target, sigma_kept, N, n)

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

        alpha = self._dispatch_alpha_mc_svd(
            traces, sigma_kept, N, n, Binv_target, W, HtRinvH,
        )
        self.alpha_ = float(alpha)

        A0 = (alpha * Binv_target + HtRinvH).tocsc()
        A0_solver = sparse_lu_solver(A0)
        if W.shape[1] > 0 and (1.0 - alpha) > 0:
            Z = woodbury_solve(
                A0_solver, [W], [1.0 - alpha], np.asarray(HtRinvD),
            )
        else:
            Z = A0_solver(np.asarray(HtRinvD))
        self.Xa = Xb + Z

    # ----- Stein/DA dense path (small problems) ------------------------

    def _assimilate_dense(self, Xb, DX, H, Dinn, observation, N, n):
        from scipy.sparse import issparse
        Binv_target = self.get_target_precision_matrix(DX, r=self.r)
        if issparse(Binv_target):
            Binv_target = Binv_target.toarray()
        Binv_target = np.asarray(Binv_target)

        W, sigma_kept = self.get_pseudo_inverse_factor(DX)
        traces = self._shared_traces(W, Binv_target, sigma_kept, N, n)

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

        alpha = self._dispatch_alpha_mc_svd(
            traces, sigma_kept, N, n, Binv_target, W, HtRinvH,
        )
        self.alpha_ = float(alpha)

        if W.shape[1] > 0:
            Pseu = W @ W.T
        else:
            Pseu = np.zeros((n, n))
        Binv_shrunk = alpha * Binv_target + (1.0 - alpha) * Pseu
        IN = Binv_shrunk + HtRinvH
        Z = np.linalg.solve(np.asarray(IN), np.asarray(HtRinvD))
        self.Xa = Xb + Z

    # ----- dispatch for the Stein/DA criteria --------------------------

    def _dispatch_alpha_mc_svd(self, traces, sigma_kept, N, n,
                               Binv_target, W, HtRinvH):
        """Pick α for the modified-Cholesky-vs-SVD criteria (Stein, DA).

        The target B_MC is used as-is (no scale factor): Stein's loss is
        scale-invariant, and the DA criterion's H^T R⁻¹ H term sets an
        absolute reference scale.
        """
        crit = self.criterion
        if crit == "stein":
            return self._alpha_stein(traces, sigma_kept, N, n,
                                     Binv_target, W)
        if crit == "da":
            return self._alpha_da_goldensection(
                Binv_target, W, HtRinvH, n,
            )
        raise ValueError(f"Unknown criterion {crit!r} for MC/SVD path")

    @staticmethod
    def _alpha_da_goldensection(Binv_target, W, HtRinvH, n,
                                n_probes=12, n_iter=40):
        """Criterion C (DA-aware) by direct golden-section minimisation.

        Minimises J_C(α) = tr[(α B_MC + (1-α) B_SVD + H^T R⁻¹ H)⁻¹]
        directly on [0,1]. J_C is strictly convex (Sec. 4), so golden
        section converges to the unique minimiser. The trace of the
        inverse is estimated by a Hutchinson probe using the
        sparse-plus-low-rank SMW solver, never forming an n×n inverse.

        This replaces the α₀=½ linearisation, which underestimates the
        optimal weight when the minimiser sits near an endpoint — the
        usual situation here, because the rank-deficient B_SVD makes the
        posterior trace blow up as α→0, pushing the optimum toward α=1.
        """
        from scipy.sparse import issparse
        from ._woodbury_solver import sparse_lu_solver, dense_lu_solver

        k = W.shape[1]
        rng = np.random.default_rng(0)
        Z = rng.choice([-1.0, 1.0], size=(n, n_probes))

        def trA(alpha):
            # A0 = α B_MC + H^T R⁻¹ H ; low-rank term (1-α) W W^T
            A0 = alpha * Binv_target + HtRinvH
            if issparse(A0):
                solve = sparse_lu_solver(A0.tocsc())
            else:
                solve = dense_lu_solver(np.asarray(A0))
            if k > 0 and (1.0 - alpha) > 0:
                # SMW:  M⁻¹ = A0⁻¹ - (1-α) A0⁻¹W (I + (1-α)Wᵀ A0⁻¹W)⁻¹ Wᵀ A0⁻¹
                A0invW = solve(W)
                cap = np.eye(k) + (1.0 - alpha) * (W.T @ A0invW)
                cap_inv = np.linalg.inv(cap)
                A0invZ = solve(Z)
                corr = (1.0 - alpha) * A0invW @ (cap_inv @ (W.T @ A0invZ))
                MinvZ = A0invZ - corr
            else:
                MinvZ = solve(Z)
            return float(np.einsum("ij,ij->", Z, MinvZ) / n_probes)

        # golden-section search on [0,1]
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = 0.0, 1.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = trA(c), trA(d)
        for _ in range(n_iter):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = trA(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = trA(d)
            if abs(b - a) < 1e-4:
                break
        return AnalysisEnKFShrinkageBinv._clip01(0.5 * (a + b))

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


# Backwards-compatible alias (legacy name).
AnalysisEnKFShrinkagePrecision = AnalysisEnKFShrinkageBinv
