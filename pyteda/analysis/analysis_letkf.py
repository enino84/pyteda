import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis

@register_analysis("letkf")
class AnalysisLETKF(Analysis):
    """Analysis Local Ensemble Transform Kalman Filter (LETKF)"""

    def __init__(self, model, r=1, **kwargs):
        """
        Initialize the AnalysisLETKF object.

        Parameters
        ----------
        model : Model object
            An object that has all the methods and attributes of the given model.
        r : int, dict, or ndarray, optional
            Localization radius (default 1). Three forms are accepted:
              * ``int`` / ``float`` — single radius for all components.
              * ``dict``            — per variable block, e.g. ``{'q': 2, 'psi': 4}``
                                      for QGModel, or ``{'x': 2}`` for Lorenz96.
              * ``ndarray``         — per state component (length ``n``).
            The model handles the dispatch in ``get_ngb``.
        """
        self.model = model
        self.r = r

    def local_analysis_LETKF(self, Xb, H, R, y, n, N, i, r):
        # Subdomain decomposition
        si = self.model.get_ngb(i, r) #[(i+j) % n for j in range(-r, r+1)]
        center_index = np.where(si == i)[0][0]  # Index of the center element in the subdomain
        Xbi = Xb[:, si].T
        xbi = np.mean(Xbi, axis=1).reshape(-1,1)
        DXi = Xbi - xbi

        # Observations
        oi = np.array([s_i for s_i in si if s_i in H])  # Global index
        Hi = np.array([i for i, s_i in enumerate(si) if s_i in H])  # Local indexes
        mi = len(Hi)  # Number of local observations
        yz = np.zeros((n,))
        yz[H] = y

        if mi > 0:
            yi = yz[oi].reshape(-1,1)  # We take the local observations from the model state
            Ri = R * np.eye(mi, mi)  # Local error covariance matrix - diagonal (R is the scalar variance)
            di = yi - xbi[Hi]  # Innovation matrix (local)
            di = di.reshape(-1,1)

            Qi = DXi[Hi, :]

            projection_onto_ensemble_space = Qi.T @ np.linalg.solve(Ri, Qi)
            Ui, Si, _ = np.linalg.svd(projection_onto_ensemble_space)
            Pa_ens_invi = Ui @ np.diag(1 / (Si + 1)) @ Ui.T

            rhsi = Qi.T @ np.linalg.solve(Ri, di)
            dxai = DXi @ Pa_ens_invi @ rhsi
            xai = xbi + dxai

            Pat_sqrti = Ui @ np.diag(np.sqrt(1 / (Si + 1))) @ Ui.T

            """
            Pat = I - V.T @ np.linalg.solve(IN, V)
            U, S, V = np.linalg.svd(Pat)
            Pat_sqrt = U @ np.diag(np.sqrt(S)) @ U.T
            """

            DXA_inci = DXi @ Pat_sqrti
            Xai = xai.reshape(-1, 1) + DXA_inci
            Xai = Xai.T

        else:
            Xai = Xbi.T

        return Xai, center_index

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
        Xb = background.Xb.T
        H = observation.H_index
        # Avoid materialising the full m×m R matrix at high obs dim.
        # All R access in the local analyses needs only the scalar
        # variance (we assume isotropic noise locally), so we extract it
        # from the cheap noise.R_diag accessor when available.
        if hasattr(observation, "noise") and hasattr(observation.noise, "R_diag"):
            R_value = float(observation.noise.R_diag[0])
        else:
            # Fallback: materialise the (small) full matrix
            R_value = float(observation.R[1, 1])
        y = observation.y
        n = self.model.get_number_of_variables()
        ensemble_size = background.ensemble_size

        Xa = np.zeros((ensemble_size, n))  # Local analysis for each model component i
        for i in range(0, n):
            Xai, center_index = self.local_analysis_LETKF(Xb, H, R_value, y, n, ensemble_size, i, self.r)
            Xa[:, i] = Xai[:, center_index]
            #Xa[:, i] = Xai[:, self.r]

        self.Xa = Xa.T

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
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
