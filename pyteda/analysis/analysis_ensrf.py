# -*- coding: utf-8 -*-

import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis

@register_analysis("ensrf")
class AnalysisEnSRF(Analysis):
    """
    Analysis EnSRF
    
    Methods
    -------
    perform_assimilation(background, observation)
        Performs the assimilation step given the background and observations.
    get_analysis_state()
        Returns the computed column mean of the ensemble Xa.
    get_ensemble()
        Returns the ensemble Xa.
    get_error_covariance()
        Returns the computed covariance matrix of the ensemble Xa.
    inflate_ensemble(inflation_factor)
        Computes the new ensemble Xa given the inflation factor.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AnalysisEnKF object.
        """
        pass
  
    def perform_assimilation(self, background, observation):
        """
        Performs the assimilation step of the ensemble Xa given the background and observations.
        
        Parameters
        ----------
        background : Background Object
            The background object defined in the Background class.
        observation : Observation Object
            The observation object defined in the Observation class.
        
        Returns
        -------
        Xa : Matrix
            Assimilated ensemble Xa.
        """
        xb = background.get_background_state()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()
        DX = background.get_member_deviations()

        _, N = DX.shape

        I = np.eye(N)
        d = y - H @ xb
        # Deviations scaled so that P^b = DX DX^T, i.e. divided by sqrt(N - 1).
        # With raw deviations the filter assimilates with a background
        # covariance N - 1 times too large.
        DX = DX / np.sqrt(N - 1)
        V = H @ DX
        IN = R + V @ V.T
        dxa = DX @ V.T @ np.linalg.solve(IN, d)
        xa = xb + dxa
        
        Pat = I - V.T @ np.linalg.solve(IN, V)

        U, S, V = np.linalg.svd(Pat)

        Pat_sqrt = U @ np.diag(np.sqrt(S)) @ U.T

        self.xa = xa
        self.Xa = xa.reshape(-1, 1) + np.sqrt(N - 1) * (DX @ Pat_sqrt)

        return self.Xa
  
    def get_analysis_state(self):
        """
        Computes the column-wise mean vector of the ensemble Xa.
        
        Returns
        -------
        mean vector : array
            Column-wise mean vector of Xa.
        """
        return self.xa
  
    def get_ensemble(self):
        """
        Returns the ensemble Xa.
        
        Returns
        -------
        Xa : matrix
            Ensemble matrix Xa.
        """
        return self.Xa
  
    def get_error_covariance(self):
        """
        Returns the computed covariance matrix of the ensemble Xa.
        
        Returns
        -------
        covariance matrix : matrix
            Covariance matrix of Xa.
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """
        Computes the ensemble Xa given the inflation factor.
        
        Parameters
        ----------
        inflation_factor : int or float
            Double number indicating the inflation factor.
        
        Returns
        -------
        None
        """
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
