# -*- coding: utf-8 -*-
"""Local observation-space modified Cholesky EnKF.

This is the LOCAL counterpart of ``AnalysisEnKFObsModifiedCholesky``. Where
the global filter bands the modified-Cholesky factor of the *global*
innovation precision ``S^{-1} = (R + H B H^T)^{-1}`` --- and therefore relies
on ``S^{-1}`` being approximately banded in the chosen observation ordering
(see Remark "bandable class" in the companion paper) --- this local variant
removes that structural hypothesis.

Following the LETKF, the state is partitioned into overlapping local domains
(one per state component, ``model.get_ngb(i, r)``). For each domain we:

    1. gather only the observations that fall inside the domain;
    2. form the local perturbed innovations
           d_local^[e] = eps_local^[e] - H_local dxb^[e],   Cov = S_local,
       which are exact draws from N(0, S_local) for linear H (Theorem 1 of
       the companion paper, applied to the local sub-operator);
    3. estimate the local innovation precision S_local^{-1} by a modified
       Cholesky (regression) decomposition that, because the domain is small
       (m_local <= a few times the radius), uses the FULL local band --- i.e.
       no off-band truncation. The local precision is therefore estimated
       essentially exactly (up to sampling error), NOT under a banding
       assumption on its inverse;
    4. apply the local Kalman gain K_local = B_local H_local^T S_local^{-1}
       through the ensemble and write back only the centre component.

The motivation is exactly the regime where the global filter is expected to
degrade: SPARSE / inhomogeneous observation meshes, where the global S^{-1}
is not banded in any natural observation ordering. By solving an EXACT local
problem in each window (as the LETKF does), localisation becomes a domain
decomposition (R-localisation, in the sense of Sakov & Bertino) rather than
a sparsity hypothesis on the precision. The two filters thus bracket the two
localisation philosophies with a comparable modified-Cholesky estimator at
their core:

    * global  (AnalysisEnKFObsModifiedCholesky)      -> dense meshes
    * local   (AnalysisEnKFObsModifiedCholeskyLocal) -> sparse meshes

Residual-variance correction
-----------------------------
Each local conditional regression of d_i on its k predecessors over N_ens
samples uses the unbiased residual variance estimate ||err||^2 / (N_ens - k)
(rather than the maximum-likelihood ||err||^2 / N_ens of the global filter's
Eq. (8)). With small ensembles and several regressors the ML denominator
under-estimates the residual variance and hence OVER-confides the precision;
the (N_ens - k) correction removes that finite-sample optimism, which we
found to be a leading contributor to the under-dispersion of the global
filter.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .analysis_core import Analysis
from .registry import register_analysis


@register_analysis("enkf-obs-modified-cholesky-local")
class AnalysisEnKFObsModifiedCholeskyLocal(Analysis):
    """Local observation-space modified-Cholesky EnKF.

    Performs an independent local analysis on each state-centred domain
    ``model.get_ngb(i, r)``. Inside every domain the local innovation
    precision is estimated by a full (untruncated) modified-Cholesky
    decomposition of the local perturbed innovations and the local gain is
    applied through the ensemble. Only the centre component of each local
    analysis is written back, exactly as in the LETKF.

    Attributes:
        model (Model object): Model exposing ``get_ngb(i, r)`` (the local
            domain of state component ``i``) and ``get_number_of_variables``.
        r (int, dict, or ndarray): Localisation radius. Same int/dict/ndarray
            dispatch as the LETKF (the model resolves it in ``get_ngb``).
        alpha (float): Ridge regularisation for the local conditional
            regressions. Default 0.01.
        cholesky_method (str): 'ridge' (penalty ``alpha``) or 'svd'
            (truncated pseudo-inverse with cutoff ``cholesky_tol``) for the
            per-row local regressions. Default 'ridge'.
        cholesky_tol (float): Relative singular-value cutoff for 'svd'.
        ddof_correction (bool): If True (default) use the unbiased residual
            variance ||err||^2/(N_ens - k); if False use the ML
            ||err||^2/N_ens of the global filter.
    """

    def __init__(self, model=None, r=1, alpha: float = 0.01,
                 cholesky_method: str = "ridge", cholesky_tol: float = 0.1,
                 ddof_correction: bool = True, **kwargs):
        self.model = model
        self.r = r
        self.alpha = float(alpha)
        if cholesky_method not in ("ridge", "svd"):
            raise ValueError(
                f"cholesky_method must be 'ridge' or 'svd'; got {cholesky_method!r}."
            )
        self.cholesky_method = cholesky_method
        self.cholesky_tol = float(cholesky_tol)
        self.ddof_correction = bool(ddof_correction)

    # ------------------------------------------------------------------
    # Local conditional regression (one row of the local Cholesky factor)
    # ------------------------------------------------------------------
    def _local_regression(self, X, y, lr):
        """Solve the regression of ``y`` on the columns of ``X`` by the
        configured method, returning ``(coef, residual)``."""
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

    # ------------------------------------------------------------------
    # Full (untruncated) local innovation precision S_local^{-1}
    # ------------------------------------------------------------------
    def _local_innovation_precision(self, Dloc):
        """Estimate the FULL local innovation precision S_local^{-1} by a
        modified-Cholesky decomposition of the local innovation samples.

        ``Dloc`` has shape (m_local, N_ens): row j is local observation j
        across the ensemble. Predecessors of observation j are ALL earlier
        local observations 0..j-1 (full band: the domain is small, so no
        off-band truncation is imposed -- this is the key difference from the
        global filter, which truncates to a fixed bandwidth).

        Returns the dense (m_local x m_local) precision S_local^{-1}.
        """
        m_local, N = Dloc.shape
        lr = Ridge(fit_intercept=False, alpha=self.alpha)
        L = np.eye(m_local)
        D_diag = np.empty(m_local)

        # Floor on residual variances to keep S_local^{-1} well conditioned.
        var_floor = 1e-12 * max(float(np.mean(np.var(Dloc, axis=1))), 1e-300)

        # Cap predecessors so each regression stays over-determined.
        max_pre = max(N - 2, 1)

        # First local observation: no predecessors.
        D_diag[0] = 1.0 / max(np.var(Dloc[0, :]), var_floor)

        for j in range(1, m_local):
            y = Dloc[j, :]
            pre = np.arange(j)                # full band: all earlier obs
            if pre.size > max_pre:
                pre = pre[-max_pre:]          # keep nearest predecessors
            k = pre.size
            X = Dloc[pre, :].T
            coef, err = self._local_regression(X, y, lr)
            # Unbiased residual variance: divide by (N - k), not N.
            if self.ddof_correction and N - k > 0:
                var_res = float(err @ err) / (N - k)
            else:
                var_res = float(np.var(err))
            D_diag[j] = 1.0 / max(var_res, var_floor)
            L[j, pre] = -coef

        return L.T @ (D_diag[:, None] * L)

    # ------------------------------------------------------------------
    # One local analysis (centred on state component i)
    # ------------------------------------------------------------------
    def _local_analysis(self, Xb, state_to_obs, R_value, y_full, eps_full, i,
                        ensemble_size):
        """Assimilate the observations inside the domain of state ``i`` and
        return the updated local ensemble together with the centre index.

        Parameters mirror the LETKF helper for drop-in comparability.
        """
        n = Xb.shape[0]
        # Local state domain and centre.
        si = np.atleast_1d(self.model.get_ngb(i, self.r)).astype(int)
        center_pos = int(np.where(si == i)[0][0])

        Xbi = Xb[si, :]                                   # (n_local, N)
        xbi = np.mean(Xbi, axis=1, keepdims=True)
        DXi = Xbi - xbi                                   # (n_local, N)

        # Local observations: those whose observed state index lies in si.
        # ``state_to_obs`` is precomputed ONCE per assimilation (passed in),
        # so the per-component cost is only over the small local domain si,
        # not over all m observations -- essential for high-dim models (SWE).
        obs_state_idx = [s for s in si.tolist() if s in state_to_obs]
        if len(obs_state_idx) == 0:
            return Xbi, center_pos              # no local obs: background

        obs_rows = np.array([state_to_obs[s] for s in obs_state_idx], dtype=int)
        # Local rows within the domain (position of each observed state in si).
        si_pos = {int(s): p for p, s in enumerate(si)}
        local_rows = np.array([si_pos[s] for s in obs_state_idx], dtype=int)

        m_local = obs_rows.size

        # Local linearised operator restricted to the domain: selection.
        # H_local maps the local state (size n_local) to local obs (m_local).
        # Row t selects local state position local_rows[t].
        # Local background in observation space: H_local @ Xbi = Xbi[local_rows].
        HXbi = Xbi[local_rows, :]                          # (m_local, N)
        H_dxb = DXi[local_rows, :]                         # (m_local, N)

        # Local observations and perturbed innovations.
        yi = y_full[obs_rows]                              # (m_local,)
        # Perturbed observation noise: SELECT the local rows from the
        # per-cycle perturbation matrix sampled ONCE in perform_assimilation.
        # We never draw randomness inside the per-domain loop, so the global
        # numpy RNG stream is consumed exactly once per cycle (as in the other
        # filters) and the benchmark stays reproducible and method-comparable.
        eps = eps_full[obs_rows, :]                        # (m_local, N)
        # Full innovations for the update (mean retained).
        Di = (yi[:, None] + eps) - HXbi                    # (m_local, N)
        # Centred innovations ~ N(0, S_local) for precision estimation.
        # Cov(eps - H dxb) = R + H B H^T = S_local exactly (linear H).
        Dci = eps - H_dxb                                  # (m_local, N)

        # Local innovation precision (full band, exact local problem).
        Sinv_loc = self._local_innovation_precision(Dci)

        # Local gain applied through the ensemble:
        #   K_local = B_local H_local^T S_local^{-1}
        #   B_local H_local^T u = (1/(N-1)) DXi (H_dxb)^T u
        U = Sinv_loc @ Di                                  # (m_local, N)
        Xai = Xbi + (DXi @ (H_dxb.T @ U)) / (ensemble_size - 1)
        return Xai, center_pos

    # ------------------------------------------------------------------
    # Assimilation (loop over state-centred domains)
    # ------------------------------------------------------------------
    def perform_assimilation(self, background, observation):
        """Perform the local assimilation step.

        Loops over every state component, performs an independent local
        analysis on its domain, and writes back the centre component, exactly
        as in the LETKF but with a modified-Cholesky local innovation
        precision in place of the ensemble-transform solve.

        Returns:
            Xa (ndarray): Assimilated ensemble, shape (n, N_ens).
        """
        Xb = background.get_ensemble()
        n, ensemble_size = Xb.shape

        # Observation indices (state components observed) and values.
        H_index = observation.H_index
        y_full = observation.get_observation()

        # Local isotropic observation-error variance (scalar).
        if hasattr(observation, "noise") and hasattr(observation.noise, "R_diag"):
            R_value = float(observation.noise.R_diag[0])
        else:
            R_value = float(observation.R[0, 0])

        # Sample the observation-error perturbations ONCE for the whole cycle,
        # using the framework's noise model (same global-RNG stream as the
        # other filters via sample_many_legacy). Each local domain then SELECTS
        # its rows from this matrix; no randomness is drawn inside the loop, so
        # the RNG stream is not desynchronised across methods. Falls back to a
        # direct draw only if the noise model lacks the legacy sampler.
        if hasattr(observation, "noise") and \
                hasattr(observation.noise, "sample_many_legacy"):
            eps_full = observation.noise.sample_many_legacy(ensemble_size)
        else:
            eps_full = np.sqrt(R_value) * np.random.standard_normal(
                (len(np.asarray(H_index)), ensemble_size)
            )

        Xa = np.empty_like(Xb)
        # Precompute the state-index -> observation-row map ONCE.
        H_index_arr = np.asarray(H_index, dtype=int)
        state_to_obs = {int(s): r for r, s in enumerate(H_index_arr)}
        for i in range(n):
            Xai, center_pos = self._local_analysis(
                Xb, state_to_obs, R_value, y_full, eps_full, i, ensemble_size
            )
            Xa[i, :] = Xai[center_pos, :]

        self.Xa = Xa
        return self.Xa

    def get_analysis_state(self):
        """Column-wise mean vector of the ensemble Xa."""
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Return the ensemble Xa."""
        return self.Xa

    def get_error_covariance(self):
        """Sample covariance matrix of the ensemble Xa."""
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Inflate the ensemble Xa about its mean."""
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa