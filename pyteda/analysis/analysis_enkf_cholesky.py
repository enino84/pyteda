# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci

from .analysis_core import Analysis
from .registry import register_analysis

@register_analysis("enkf-cholesky")
class AnalysisEnKFCholesky(Analysis):
    """EnKF implementation Cholesky (ensemble space)
  
    Attributes:
        None

    Methods:
        perform_assimilation(background, observation): Perform assimilation step given background and observations
        get_analysis_state(): Returns the computed column mean of ensemble Xa
        get_ensemble(): Returns ensemble Xa
        get_error_covariance(): Returns the computed covariance matrix of the ensemble Xa
        inflate_ensemble(inflation_factor): Computes new ensemble Xa given the inflation factor
    """
    def __init__(self, **kwargs):
        """
        Parameters:
            None
        """
        pass
  
    def perform_assimilation(self, background, observation):
        """Perform assimilation step of ensemble Xa given the background and the observations

        Parameters:
            background (Background Object): The background object defined in the class background
            observation (Observation Object): The observation object defined in the class observation
        
        Returns:
            Xa (Matrix): Matrix of ensemble
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        R = observation.get_data_error_covariance()
        n, ensemble_size = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)

        Rinv = np.diag(np.reciprocal(np.diag(R)))
        # Sample observation noise efficiently — avoid materialising R for large m.
        # multivariate_normal does an SVD of R internally; for diagonal R that
        # is wasteful and at high m can run out of memory. Use the noise
        # object's sample_many(), which knows the right per-class shortcut.
        Ys = y[:, None] + observation.noise.sample_many_legacy(ensemble_size)
        D = Ys - HXb
        xb = np.mean(Xb, axis=1)
        DX = Xb - np.outer(xb, np.ones(ensemble_size))
        Q = H @ DX
        IN = (ensemble_size - 1) * np.eye(ensemble_size, ensemble_size) + Q.T @ (Rinv @ Q)
        L = np.linalg.cholesky(IN)
        DG = Q.T @ (Rinv @ D)
        ZG = sci.linalg.solve_triangular(L, DG, lower=True)
        Z = sci.linalg.solve_triangular(L, ZG, trans='T', lower=True)
        self.Xa = Xb + DX @ Z
        return self.Xa
  
    def get_analysis_state(self):
        """Compute column-wise mean vector of Matrix of ensemble Xa

        Parameters:
            None

        Returns:
            mean_vector: Mean vector
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Returns ensemble Xa

        Parameters:
            None

        Returns:
            ensemble_matrix: Ensemble matrix
        """
        return self.Xa
  
    def get_error_covariance(self):
        """Returns the computed covariance matrix of the ensemble Xa

        Parameters:
            None

        Returns:
            covariance_matrix: Covariance matrix of the ensemble Xa
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Computes ensemble Xa given the inflation factor

        Parameters:
            inflation_factor (int): Double number indicating the inflation factor

        Returns:
            None
        """
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
