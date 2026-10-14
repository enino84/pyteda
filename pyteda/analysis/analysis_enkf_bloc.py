import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis
from ._loc_utils import build_decorrelation

@register_analysis("enkf-b-loc")
class AnalysisEnKFBLoc(Analysis):
    """Analysis EnKF B-Loc"""

    def __init__(self, model, r=1, cross=False, cross_scale=1.0, **kwargs):
        """
        Initialize the AnalysisEnKFBLoc object.

        Parameters
        ----------
        model : Model object
            An object that has all the methods and attributes of the given model.
        r : int, optional
            Value used in the process of removing correlations (default is 1).
        cross : bool, optional
            Whether the decorrelation matrix keeps entries that link different
            variable blocks (default False, which zeroes them and makes the
            analysis univariate: observing one variable leaves the others at
            their forecast value). With ``cross=True`` an observation of one
            variable updates the others through the ensemble cross
            covariances. Has no effect on single-variable models.
        cross_scale : float, optional
            Taper applied to the cross-block entries when ``cross=True``
            (default 1.0, trusting them as much as the within-block ones).
            See :mod:`pyteda.analysis._loc_utils`.
        """
        self.model = model
        self.r = r
        self.cross = bool(cross)
        self.cross_scale = float(cross_scale)
        build_decorrelation(model, r, cross=self.cross,
                            cross_scale=self.cross_scale)

    def perform_assimilation(self, background, observation):
        """
        Perform the assimilation step of the ensemble Xa given the background and observations.

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
        Xb = background.get_ensemble()
        Pb = background.get_covariance_matrix()
        y = observation.get_observation()
        R = observation.get_data_error_covariance()
        n, matrix_size = Xb.shape

        from ._obs_utils import linearize_at_mean
        H, HXb = linearize_at_mean(observation, Xb)

        # Sample observation noise efficiently — avoid materialising R for large m.
        # multivariate_normal does an SVD of R internally; for diagonal R that
        # is wasteful and at high m can run out of memory. Use the noise
        # object's sample_many(), which knows the right per-class shortcut.
        Ys = y[:, None] + observation.noise.sample_many_legacy(matrix_size)
        L = self.model.get_decorrelation_matrix()
        Pb = L * np.cov(Xb)
        D = Ys - HXb
        IN = R + H @ (Pb @ H.T)
        Z = np.linalg.solve(IN, D)
        self.Xa = Xb + Pb @ (H.T @ Z)
        return self.Xa

    def get_analysis_state(self):
        """
        Compute the column-wise mean vector of the ensemble Xa.

        Returns
        -------
        mean vector : array
            Column-wise mean vector of Xa.
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """
        Return the ensemble Xa.

        Returns
        -------
        Xa : matrix
            Ensemble matrix Xa.
        """
        return self.Xa

    def get_error_covariance(self):
        """
        Return the computed covariance matrix of the ensemble Xa.

        Returns
        -------
        covariance matrix : matrix
            Covariance matrix of Xa.
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """
        Compute the ensemble Xa given the inflation factor.

        Parameters
        ----------
        inflation_factor : int or float
            Double number indicating the inflation factor.

        Returns
        -------
        None
        """
        n, matrix_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(matrix_size))
        self.Xa = np.outer(xa, np.ones(matrix_size)) + inflation_factor * DXa
