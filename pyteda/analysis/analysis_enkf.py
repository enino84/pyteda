# -*- coding: utf-8 -*-

import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis

@register_analysis("enkf")
class AnalysisEnKF(Analysis):
    """
    Analysis EnKF full covariance matrix
    
    Methods
    -------
    perform_assimilation(background, observation)
        Perform the assimilation step given the background and observations
    get_analysis_state()
        Return the computed column mean of the ensemble Xa
    get_ensemble()
        Return the ensemble Xa
    get_error_covariance()
        Return the computed covariance matrix of the ensemble Xa
    inflate_ensemble(inflation_factor)
        Compute the new ensemble Xa given the inflation factor
    """

    def __init__(self, **kwargs):
        """
        Initialize the AnalysisEnKF object.
        
        Parameters
        ----------
        None
        """
        pass
  
    def perform_assimilation(self, background, observation):
        """
        Perform the assimilation step of the ensemble Xa given the background and observations.

        For high-dimensional problems (n ≥ SPARSE_THRESHOLD), uses the
        iterative Sherman-Morrison-Woodbury formula to avoid forming
        P^b ∈ R^(n×n) explicitly. The system

            [R + H · P^b · H^T] · Z = D

        is solved with P^b = (1/(N-1)) · ΔX · ΔX^T factorised, so that

            R + Q · Q^T,    Q = H · ΔX / sqrt(N-1)   ∈ R^(m × N)

        only requires a low-rank correction. The full state-space
        covariance is never built. For small problems (n < SPARSE_THRESHOLD)
        the original dense path is preserved.

        Parameters
        ----------
        background : Background Object
        observation : Observation Object

        Returns
        -------
        Xa : ndarray
            Assimilated ensemble Xa.
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        n, ensemble_size = Xb.shape

        from ._obs_utils import linearize_at_mean
        from ..observation.noise import SPARSE_THRESHOLD
        H, HXb = linearize_at_mean(observation, Xb)

        # Sample observation noise efficiently — avoid materialising R for large m.
        Ys = y[:, None] + observation.noise.sample_many_legacy(ensemble_size)
        D = Ys - HXb

        # Deviations (always cheap — n × N).
        xb = Xb.mean(axis=1)
        DX = Xb - xb[:, None]

        if n >= SPARSE_THRESHOLD:
            # Matrix-free path: solve in observation space with Woodbury.
            #   [R + (1/(N-1)) (HΔX)(HΔX)^T] Z = D
            from ._woodbury_solver import (
                woodbury_solve, diagonal_solver, dense_lu_solver,
            )
            H_dense = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
            HDX = H_dense @ DX                                # (m, N)
            Q = HDX / np.sqrt(ensemble_size - 1)              # (m, N)

            if hasattr(observation.noise, "R_inv_diag"):
                r_inv_diag = observation.noise.R_inv_diag
                A0_solver = diagonal_solver(1.0 / r_inv_diag)
            else:
                R = observation.get_data_error_covariance()
                A0_solver = dense_lu_solver(R)

            Z = woodbury_solve(A0_solver, [Q], [1.0], D)      # (m, N)

            # Apply increment: X^a = X^b + P^b · H^T · Z
            #                       = X^b + (1/(N-1)) · ΔX · (ΔX^T H^T Z)
            self.Xa = Xb + (DX @ (H_dense.T @ Z)) / (ensemble_size - 1)
        else:
            # Original dense path for small problems.
            Pb = background.get_covariance_matrix()
            R = observation.get_data_error_covariance()
            IN = R + H @ (Pb @ H.T)
            if hasattr(IN, "toarray"):
                IN = IN.toarray()
            Z = np.linalg.solve(np.asarray(IN), D)
            self.Xa = Xb + Pb @ (H.T @ Z)
        return self.Xa
  
    def get_analysis_state(self):
        """
        Compute the column-wise mean vector of the ensemble Xa.
        
        Returns
        -------
        mean vector : array
            Column-wise mean vector of Xa
        """
        return np.mean(self.Xa, axis=1)
  
    def get_ensemble(self):
        """
        Return the ensemble Xa.
        
        Returns
        -------
        Xa : matrix
            Ensemble matrix Xa
        """
        return self.Xa
  
    def get_error_covariance(self):
        """
        Return the computed covariance matrix of the ensemble Xa.
        
        Returns
        -------
        covariance matrix : matrix
            Covariance matrix of Xa
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """
        Compute the ensemble Xa given the inflation factor.
        
        Parameters
        ----------
        inflation_factor : int or float
            Double number indicating the inflation factor
        
        Returns
        -------
        None
        """
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
