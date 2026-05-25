# -*- coding: utf-8 -*-
"""Observation-space modified Cholesky EnKF.

This filter estimates the *innovation precision*

    S^{-1} = (R + H B H^T)^{-1}

directly in observation space, by a banded modified Cholesky (regression)
decomposition of the perturbed innovations

    d^[e] = eps^[e] - H dxb^[e],     eps^[e] ~ N(0, R),  dxb^[e] ~ N(0, B).

For a *linear* observation operator H these d^[e] are exact draws from
N(0, S) (linear image of a Gaussian plus an independent Gaussian), so the
Bickel & Levina banded modified Cholesky estimator applied to them returns
a sparse, symmetric-positive-definite factor of S^{-1} with no inversion.
The Kalman gain K = B H^T S^{-1} is then applied matrix-free through the
ensemble, never forming or inverting an m x m matrix.

This is the observation-space counterpart of
``AnalysisEnKFModifiedCholesky`` (which bands the *state* precision B^{-1}
via ``model.get_pre``). Here the banding lives among the *observations*:
predecessors are taken in a fixed observation ordering. When the operator
is a ``LinearSelection`` each observation inherits the grid index of the
state component it samples, so the model's own ``get_pre`` defines spatial
predecessors among observations; otherwise a sequential observation order
is used as a fallback.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .analysis_core import Analysis
from .registry import register_analysis


@register_analysis("enkf-obs-modified-cholesky")
class AnalysisEnKFObsModifiedCholesky(Analysis):
    """Analysis EnKF via observation-space modified Cholesky.

    Estimates the innovation precision S^{-1} = (R + H B H^T)^{-1} in
    observation space from the perturbed innovations, then applies the
    gain K = B H^T S^{-1} matrix-free.

    Attributes:
        model (Model object): Model providing ``get_pre`` for the
            spatial ordering of observations (used when the operator is a
            ``LinearSelection`` so observations carry grid indices).
        r (int, dict, or ndarray): Localisation radius / bandwidth used to
            pick observation predecessors. Same int/dict/ndarray dispatch
            as the state-space modified-Cholesky filter.
        alpha (float): Ridge regression regularisation for the per-row
            regressions. Default 0.01.

    Methods:
        get_innovation_precision(D, order, pre): Returns S^{-1} as
            L^T D_diag L from banded regressions of the innovations.
        perform_assimilation(background, observation): Assimilation step.
        get_analysis_state(): Column mean of ensemble Xa.
        get_ensemble(): Ensemble Xa.
        get_error_covariance(): Sample covariance of ensemble Xa.
        inflate_ensemble(inflation_factor): Inflate Xa about its mean.
    """

    def __init__(self, model=None, r=1, alpha: float = 0.01,
                 b_via_cholesky: bool = False, r_state=None,
                 cholesky_method: str = "ridge", cholesky_tol: float = 0.1,
                 alpha_state=None, cholesky_method_state=None,
                 cholesky_tol_state=None,
                 **kwargs):
        """
        Initialize an instance of AnalysisEnKFObsModifiedCholesky.

        Parameters:
            model (Model object, optional): Model exposing ``get_pre``.
                Required for the spatial observation ordering and, when
                ``b_via_cholesky`` is True, for the state-space predecessors
                used to estimate B^{-1}.
            r (int, dict, or ndarray, optional): Radius for picking
                observation-space predecessors (innovation precision).
                Default 1.
            alpha (float, optional): Ridge regularisation (the lambda) used
                when ``cholesky_method == 'ridge'``, for the OBSERVATION-space
                S^{-1} regressions. Default 0.01.
            b_via_cholesky (bool, optional): If True, the factor B in the
                gain K = B H^T S^{-1} is applied NOT through the (dense,
                rank-N) sample P^b, but by estimating the state-space
                precision B^{-1} = T_B^T D_B T_B via modified Cholesky and
                SOLVING the linear system B^{-1} z = H^T u for z = B H^T u.
                This removes the spurious long-range sample correlations of
                P^b from the gain and keeps both factors localized and
                matrix-free. Default False (use sample P^b).
            r_state (int, dict, or ndarray, optional): Radius for the
                state-space predecessors of B^{-1}. Defaults to ``r``.
            cholesky_method (str, optional): How each OBSERVATION-space
                conditional regression (a row of the S^{-1} factor) is
                solved: ``'ridge'`` (penalty ``alpha``) or ``'svd'``
                (truncated pseudo-inverse with cutoff ``cholesky_tol``).
                Default 'ridge'.
            cholesky_tol (float, optional): Relative singular-value cutoff
                for the truncated-SVD solver in OBSERVATION space (used only
                when ``cholesky_method == 'svd'``). Default 0.1.
            alpha_state (float, optional): Ridge penalty for the STATE-space
                B^{-1} regressions in the gain. If None, inherits ``alpha``.
            cholesky_method_state (str, optional): Regression method for the
                STATE-space B^{-1} of the gain. If None, inherits
                ``cholesky_method``. Lets the gain's B^{-1} use a different
                solver/tolerance than the innovation S^{-1}, since the two
                live in different spaces with different conditioning.
            cholesky_tol_state (float, optional): Truncated-SVD cutoff for
                the STATE-space B^{-1}. If None, inherits ``cholesky_tol``.
        """
        self.model = model
        self.r = r
        self.alpha = float(alpha)
        self.b_via_cholesky = bool(b_via_cholesky)
        self.r_state = r if r_state is None else r_state
        if cholesky_method not in ("ridge", "svd"):
            raise ValueError(
                f"cholesky_method must be 'ridge' or 'svd'; got {cholesky_method!r}."
            )
        self.cholesky_method = cholesky_method
        self.cholesky_tol = float(cholesky_tol)

        # State-space (gain B^{-1}) regression controls; inherit from the
        # observation-space settings when not given explicitly.
        self.alpha_state = (
            float(alpha) if alpha_state is None else float(alpha_state)
        )
        cms = cholesky_method if cholesky_method_state is None \
            else cholesky_method_state
        if cms not in ("ridge", "svd"):
            raise ValueError(
                f"cholesky_method_state must be 'ridge' or 'svd'; got {cms!r}."
            )
        self.cholesky_method_state = cms
        self.cholesky_tol_state = (
            float(cholesky_tol) if cholesky_tol_state is None
            else float(cholesky_tol_state)
        )

    # ------------------------------------------------------------------
    # Local conditional regression (shared by S^{-1} and B^{-1})
    # ------------------------------------------------------------------
    def _local_regression(self, X, y, lr, method, tol):
        """Solve the local regression of ``y`` on the columns of ``X`` by
        ``method`` ('ridge' or 'svd'), returning ``(coef, residual)``.

        ``ridge`` uses the supplied scikit-learn estimator ``lr``; ``svd``
        uses a truncated-SVD pseudo-inverse dropping singular directions
        with sigma/sigma_1 <= ``tol``.
        """
        if method == "svd":
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            if s.size == 0 or s[0] == 0:
                return np.zeros(X.shape[1]), y.copy()
            keep = s / s[0] > tol
            if not np.any(keep):
                return np.zeros(X.shape[1]), y.copy()
            Uk, sk, Vk = U[:, keep], s[keep], Vt[keep, :]
            coef = Vk.T @ ((Uk.T @ y) / sk)
            return coef, y - X @ coef
        # ridge
        lr_fit = lr.fit(X, y)
        return lr_fit.coef_, y - lr_fit.predict(X)


    # ------------------------------------------------------------------
    # State-space background precision B^{-1} (for the consistent gain)
    # ------------------------------------------------------------------
    def get_background_precision(self, DX, regularization_factor=None):
        """Estimate the state-space background precision B^{-1} = L^T D L by
        a banded modified Cholesky decomposition of the background
        deviations (each state component regressed on its spatial
        predecessors via ``model.get_pre``).

        Returns a sparse CSR matrix (large n) or a dense ndarray (small n),
        suitable for SOLVING the system B^{-1} z = w rather than forming B.

        Parameters:
            DX (ndarray): Background deviations, shape (n, N_ens).
            regularization_factor (float, optional): Override Ridge alpha.

        Returns:
            Binv : ndarray or scipy.sparse.csr_matrix.
        """
        from sklearn.linear_model import Ridge
        alpha = float(regularization_factor) if regularization_factor is not None \
            else self.alpha_state
        n, ensemble_size = DX.shape
        lr = Ridge(fit_intercept=False, alpha=alpha)

        from ..observation.noise import SPARSE_THRESHOLD
        sparse_path = n >= SPARSE_THRESHOLD
        max_pre = max(ensemble_size - 2, 1)
        var_floor = 1e-6 * float(np.mean(np.var(DX, axis=1)))

        if sparse_path:
            from scipy.sparse import lil_matrix, diags, csr_matrix
            L = lil_matrix((n, n), dtype=float)
            L.setdiag(1.0)
            D_diag = np.empty(n, dtype=float)
        else:
            L = np.eye(n)
            D_diag = np.empty(n, dtype=float)

        D_diag[0] = 1.0 / max(np.var(DX[0, :]), var_floor)
        for i in range(1, n):
            ind_prede = np.asarray(self.model.get_pre(i, self.r_state), dtype=int)
            if ind_prede.size > max_pre:
                ind_prede = ind_prede[-max_pre:]
            y = DX[i, :]
            if ind_prede.size == 0:
                D_diag[i] = 1.0 / max(np.var(y), var_floor)
                continue
            X = DX[ind_prede, :].T
            coef, err_i = self._local_regression(
                X, y, lr, self.cholesky_method_state, self.cholesky_tol_state
            )
            D_diag[i] = 1.0 / max(np.var(err_i), var_floor)
            L[i, ind_prede] = -coef

        if sparse_path:
            L = csr_matrix(L)
            Dm = diags(D_diag, format="csr")
            return (L.T @ Dm @ L).tocsr()
        else:
            return L.T @ (D_diag[:, None] * L)

    # ------------------------------------------------------------------
    # Observation ordering and predecessors
    # ------------------------------------------------------------------
    def _obs_predecessors(self, observation, m):
        """Return, for each observation row i, the list of its predecessor
        rows (indices < i in the chosen ordering) within the bandwidth.

        Two regimes:

        The bandwidth ``k`` is taken from ``self.r`` (an int, or the rounded
        mean of a dict/array radius). In every regime the predecessor set of
        the k-th observation in the ordering is the block of its ``k``
        immediately preceding observations in that ordering --- exactly the
        ``k``-banded modified Cholesky factor the theory assumes. This
        guarantees ``k`` genuine predecessors (subject to the start of the
        ordering) and avoids the gaps that arise when a state-space radius is
        mapped through a partial observation network.

        * ``LinearSelection`` operator — observations are ordered by the
          state grid index they sample, so "immediately preceding in the
          ordering" means spatially nearest among the observed locations.
          This makes the banding a genuine spatial localisation in
          observation space.

        * otherwise — a sequential ordering 0,1,...,m-1 is used, and the
          band is over consecutive observation rows.

        Returns:
            order (ndarray): permutation of range(m) giving the ordering.
            pre (list of ndarray): pre[k] holds the ORIGINAL observation
                rows that precede the k-th observation in ``order`` AND are
                spatial neighbours of it. May be empty when the network is
                sparse near an observation (then that row's factor is purely
                diagonal).
        """
        # Radius / bandwidth from the spec.
        if isinstance(self.r, (int, float)):
            radius = int(round(self.r))
        else:
            radius = int(round(np.mean(np.asarray(self.r, dtype=float))))
        radius = max(radius, 1)

        # Spatial ordering when the operator exposes state indices.
        state_index = None
        try:
            state_index = np.asarray(
                observation.get_observation_operator_index(), dtype=int
            )
        except Exception:
            state_index = None

        if state_index is None:
            # No grid information: fall back to natural order with a band
            # over consecutive observation rows (predecessors are genuine
            # neighbours in this regime by construction).
            order = np.arange(m)
            pre = [order[max(0, j - radius):j] for j in range(m)]
            return order, pre

        # Order observations by the grid index they sample.
        order = np.argsort(state_index, kind="stable")
        # Observation rows present at each state index.
        from collections import defaultdict
        rows_at = defaultdict(list)
        for row in range(m):
            rows_at[int(state_index[row])].append(row)

        # Neighbourhood test. Delegate the geometry to the model whenever it
        # is available: ``model.get_ngb(s, r)`` returns the state indices
        # within radius r of s under the model's OWN metric (1-D cyclic for
        # Lorenz-96, 2-D lat/lon-cyclic for SWE, etc.). This is what lets the
        # SAME filter localise correctly across models without assuming the
        # linear index encodes distance. Only if no model geometry is exposed
        # do we fall back to a cyclic distance on the linear index.
        use_model_geom = (
            self.model is not None and hasattr(self.model, "get_ngb")
        )

        if not use_model_geom:
            L = None
            if self.model is not None:
                for attr in ("n", "get_number_of_variables"):
                    try:
                        val = (self.model.get_number_of_variables()
                               if attr == "get_number_of_variables"
                               else getattr(self.model, attr))
                        L = int(val() if callable(val) else val)
                        break
                    except Exception:
                        L = None

            def neighbours_of(s):
                lo, hi = s - radius, s + radius
                cand = set()
                for ss in range(lo, hi + 1):
                    key = ss % L if L is not None else ss
                    cand.update(rows_at.get(int(key), []))
                return cand
        else:
            def neighbours_of(s):
                cand = set()
                try:
                    ngb = np.atleast_1d(self.model.get_ngb(int(s), radius))
                except Exception:
                    ngb = np.array([s], dtype=int)
                for ss in ngb:
                    cand.update(rows_at.get(int(ss), []))
                return cand

        # A row's predecessors are the observations that (a) come earlier in
        # the ordering AND (b) are spatial neighbours of it under the model's
        # metric. Observations with no neighbour inside the radius get an
        # EMPTY predecessor set, so their factor row is purely diagonal
        # (residual = innovation), rather than being regressed on spurious
        # far-away observations.
        pre = [None] * m
        for j in range(m):
            row = order[j]
            s = int(state_index[row])
            nb = neighbours_of(s)
            preds = [order[t] for t in range(j) if order[t] in nb]
            pre[j] = np.asarray(preds, dtype=int)
        return order, pre
        # so their factor row is purely diagonal (residual = innovation),
        # rather than being regressed on spurious far-away observations.
    # ------------------------------------------------------------------
    # Innovation precision via banded modified Cholesky
    # ------------------------------------------------------------------
    def get_innovation_precision(self, D, order, pre, regularization_factor=None):
        """Compute S^{-1} ≈ L^T D_diag L by a banded modified Cholesky
        decomposition of the innovation samples.

        Each innovation component (taken in ``order``) is regressed on its
        predecessors ``pre``; the residual variance gives the diagonal of
        D_diag and the negated coefficients populate the unit
        lower-triangular factor L. By construction L^T D_diag L is
        symmetric positive-definite whenever the residual variances are
        positive (Theorem: positive-definiteness by construction).

        Parameters:
            D (ndarray): Innovation samples, shape (m, N_ens). Row i is the
                i-th observation component across the ensemble.
            order (ndarray): Observation ordering (permutation of range(m)).
            pre (list of ndarray): Predecessor rows for each position in
                ``order`` (expressed as ORIGINAL observation rows).
            regularization_factor (float, optional): Override Ridge alpha.

        Returns:
            Sinv : ndarray (dense, m x m) or scipy.sparse.csr_matrix.
                The estimated innovation precision in the ORIGINAL
                observation ordering.
        """
        alpha = float(regularization_factor) if regularization_factor is not None \
            else self.alpha
        m, ensemble_size = D.shape
        lr = Ridge(fit_intercept=False, alpha=alpha)

        from ..observation.noise import SPARSE_THRESHOLD
        sparse_path = m >= SPARSE_THRESHOLD

        if sparse_path:
            from scipy.sparse import lil_matrix, diags, csr_matrix
            L = lil_matrix((m, m), dtype=float)
            L.setdiag(1.0)
            D_diag = np.empty(m, dtype=float)
        else:
            L = np.eye(m)
            D_diag = np.empty(m, dtype=float)

        # Floor on residual variances. With N_ens samples and up to k
        # regressors, a near-interpolating regression (k close to N_ens)
        # drives Var(err) -> 0 and 1/Var -> inf, blowing up the precision.
        # The consistency theory requires k small relative to N_ens; we
        # additionally floor the residual variance at a small fraction of
        # the marginal innovation variance to keep S^ well conditioned
        # whatever bandwidth the caller picks.
        var_floor = 1e-6 * float(np.mean(np.var(D, axis=1)))

        # Cap predecessors so each regression stays over-determined:
        # at most N_ens - 2 regressors. When the caller asks for a wider
        # band than the ensemble can support, keep the NEAREST predecessors
        # (the tail of the predecessor block, which are closest in the
        # ordering). This is the small-k regime the theory prescribes.
        max_pre = max(ensemble_size - 2, 1)

        for k, i in enumerate(order):
            y = D[i, :]
            ind_prede = np.asarray(pre[k], dtype=int)
            if ind_prede.size > max_pre:
                ind_prede = ind_prede[-max_pre:]
            if ind_prede.size == 0:
                # No predecessors: residual is the innovation itself.
                D_diag[i] = 1.0 / max(np.var(y), var_floor)
                continue
            X = D[ind_prede, :].T
            coef, err_i = self._local_regression(
                X, y, lr, self.cholesky_method, self.cholesky_tol
            )
            D_diag[i] = 1.0 / max(np.var(err_i), var_floor)
            # Row i of L gets the negated regression coefficients at the
            # predecessor columns. L stays unit lower-triangular in
            # ``order`` because every predecessor precedes i.
            L[i, ind_prede] = -coef

        if sparse_path:
            L = csr_matrix(L)
            Dm = diags(D_diag, format="csr")
            return (L.T @ Dm @ L).tocsr()
        else:
            return L.T @ (D_diag[:, None] * L)

    # ------------------------------------------------------------------
    # Assimilation
    # ------------------------------------------------------------------
    def perform_assimilation(self, background, observation):
        """Perform the assimilation step in observation space.

        Steps:
            1. Build background deviations DX and perturbed innovations
               D = Ys - H Xb, where Ys = y + eps are perturbed observations
               (so each column of D is a draw from N(0, S)).
            2. Estimate the innovation precision Sinv = S^{-1} by banded
               modified Cholesky of D in observation space.
            3. Apply the gain matrix-free:
                   Xa = Xb + B H^T Sinv D
                      = Xb + (1/(N-1)) DX (H DX)^T (Sinv D).

        Parameters:
            background (Background Object): Background ensemble container.
            observation (Observation Object): Observation container.

        Returns:
            Xa (ndarray): Assimilated ensemble, shape (n, N_ens).
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        n, ensemble_size = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)
        H_dense = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
        m = H_dense.shape[0]

        # Perturbed observations and innovations. Each column of D is an
        # exact draw from N(0, S) for linear H (Theorem 1):
        #   D[:, e] = (y + eps^[e]) - H xb^[e]
        # Perturbed observations and full innovations (used in the update):
        #   D[:, e] = (y + eps^[e]) - H xb^[e].
        Ys = y[:, None] + observation.noise.sample_many_legacy(ensemble_size)
        D = Ys - HXb                                   # (m, N_ens)

        # Samples of N(0, S) are the innovation DEVIATIONS about their mean
        # (Theorem 1). The full innovations D have nonzero mean y - H xbar,
        # so we centre them before estimating the precision; using the raw
        # D would bias S^ by the outer product of the mean innovation.
        Dc = D - D.mean(axis=1, keepdims=True)         # (m, N_ens) ~ N(0, S)

        # Background deviations (n x N).
        xb = np.mean(Xb, axis=1)
        DX = Xb - xb[:, None]

        # Observation ordering and predecessors, then innovation precision
        # estimated from the CENTRED innovations.
        order, pre = self._obs_predecessors(observation, m)
        Sinv = self.get_innovation_precision(Dc, order, pre)

        # Apply S^{-1} to the FULL innovations: U = Sinv @ D   (m x N).
        from scipy.sparse import issparse
        if issparse(Sinv):
            U = Sinv @ D
            U = U.toarray() if hasattr(U, "toarray") else np.asarray(U)
        else:
            U = Sinv @ D

        # Apply B H^T to U, i.e. compute Z = B (H^T U).
        W = H_dense.T @ U                                # (n, N) = H^T U
        if self.b_via_cholesky:
            # Consistent gain: B applied by SOLVING B^{-1} Z = W, with
            # B^{-1} the banded state-space modified-Cholesky precision.
            # The dense rank-(N-1) sample P^b never enters the gain.
            Binv = self.get_background_precision(DX)
            if issparse(Binv):
                from scipy.sparse.linalg import spsolve
                Z = spsolve(Binv.tocsc(), W)
                if Z.ndim == 1:
                    Z = Z[:, None]
            else:
                Z = np.linalg.solve(Binv, W)
            self.Xa = Xb + Z
        else:
            # Low-rank ensemble gain: B H^T U = (1/(N-1)) DX (H DX)^T U,
            # which uses the dense sample P^b implicitly.
            HDX = H_dense @ DX                           # (m, N)
            self.Xa = Xb + (DX @ (HDX.T @ U)) / (ensemble_size - 1)
        return self.Xa

    def get_analysis_state(self):
        """Compute the column-wise mean vector of the ensemble Xa.

        Returns:
            mean_vector (ndarray): Mean vector of Xa.
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Return the ensemble Xa.

        Returns:
            ensemble_matrix (ndarray): Ensemble matrix Xa.
        """
        return self.Xa

    def get_error_covariance(self):
        """Return the sample covariance matrix of the ensemble Xa.

        Returns:
            covariance_matrix (ndarray): Covariance matrix of Xa.
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Inflate the ensemble Xa about its mean.

        Parameters:
            inflation_factor (float): Multiplicative inflation factor.

        Returns:
            None
        """
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
