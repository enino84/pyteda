# -*- coding: utf-8 -*-
"""Tests for pyteda.models.

Each model must satisfy the Model contract:
    - get_number_of_variables() -> int
    - get_initial_condition() -> ndarray of size n
    - propagate(x, T) -> ndarray of size n
    - get_ngb(i, r), get_pre(i, r) -> integer ndarray
    - create_decorrelation_matrix(r), get_decorrelation_matrix() -> (n, n)
"""

import warnings

import numpy as np
import pytest

from pyteda.models import Lorenz96, QGModel, SWEModel


warnings.filterwarnings("ignore", category=Warning)


# ----------------------------------------------------------------------
# Lorenz96
# ----------------------------------------------------------------------
class TestLorenz96:
    def test_dimension(self):
        m = Lorenz96(n=40)
        assert m.get_number_of_variables() == 40

    def test_var_blocks(self):
        m = Lorenz96(n=40)
        assert "x" in m.var_blocks
        assert m.var_blocks["x"] == slice(0, 40)

    def test_propagate_shape(self):
        m = Lorenz96(n=20)
        x = m.get_initial_condition()
        x1 = m.propagate(x, np.array([0.0, 0.5]))
        assert x1.shape == (20,)
        assert np.isfinite(x1).all()

    def test_propagate_changes_state(self):
        m = Lorenz96(n=20)
        x = m.get_initial_condition()
        x1 = m.propagate(x, np.array([0.0, 0.5]))
        # Trajectory has moved
        assert not np.allclose(x, x1)

    def test_decorrelation_matrix_int(self):
        m = Lorenz96(n=20)
        m.create_decorrelation_matrix(r=2)
        L = m.get_decorrelation_matrix()
        assert L.shape == (20, 20)
        # Symmetric
        assert np.allclose(L, L.T)
        # Diagonal is 1 (gaussian kernel at zero distance)
        assert np.allclose(np.diag(L), 1.0)

    def test_decorrelation_three_forms_equivalent(self):
        """L matrix is identical for r=2, r={'x':2}, r=full(20, 2)."""
        m = Lorenz96(n=20)
        m.create_decorrelation_matrix(r=2)
        L1 = m.get_decorrelation_matrix()
        m.create_decorrelation_matrix(r={"x": 2})
        L2 = m.get_decorrelation_matrix()
        m.create_decorrelation_matrix(r=np.full(20, 2.0))
        L3 = m.get_decorrelation_matrix()
        assert np.allclose(L1, L2)
        assert np.allclose(L1, L3)

    def test_get_ngb_cyclic(self):
        m = Lorenz96(n=10)
        ngb_0 = m.get_ngb(0, r=2)
        # With cyclic boundary, index 0 with r=2 sees [-2, -1, 0, 1, 2] mod 10 = [8, 9, 0, 1, 2]
        assert set(ngb_0) == {0, 1, 2, 8, 9}

    def test_get_ngb_heterogeneous_radius(self):
        m = Lorenz96(n=20)
        r_arr = np.full(20, 2.0)
        r_arr[10:15] = 5.0
        # Index 0 has r=2 -> 5 neighbours
        assert len(m.get_ngb(0, r_arr)) == 5
        # Index 12 has r=5 -> 11 neighbours
        assert len(m.get_ngb(12, r_arr)) == 11

    def test_decorrelation_not_built_raises(self):
        m = Lorenz96(n=10)
        with pytest.raises(RuntimeError, match="not built"):
            m.get_decorrelation_matrix()


# ----------------------------------------------------------------------
# QGModel
# ----------------------------------------------------------------------
class TestQGModel:
    @pytest.fixture(scope="class")
    def qg(self):
        # mrefin=4 -> grid 25x25 = 625 per field, dim 1250
        return QGModel(mrefin=4, scheme="rk4", dt=1.0, ic_kind="fourier",
                       verbose=False)

    def test_dimension(self, qg):
        assert qg.get_number_of_variables() == 2 * qg.field_size

    def test_var_blocks(self, qg):
        assert set(qg.var_blocks.keys()) == {"q", "psi"}
        # q and psi together cover the full state
        q_size = qg.var_blocks["q"].stop - qg.var_blocks["q"].start
        psi_size = qg.var_blocks["psi"].stop - qg.var_blocks["psi"].start
        assert q_size + psi_size == qg.dim

    def test_lists_integrators(self):
        ints = QGModel.list_available_integrators()
        # Should at least include the standard ones
        for name in ("rk4", "euler", "ab2", "ssprk3"):
            assert name in ints

    def test_lists_ics(self):
        ics = QGModel.list_available_ics()
        for name in ("zero", "fourier", "vortex"):
            assert name in ics

    def test_propagate(self, qg):
        x = qg.get_initial_condition(seed=0)
        assert x.shape == (qg.dim,)
        x1 = qg.propagate(x, np.array([0.0, 5.0]))
        assert x1.shape == (qg.dim,)
        assert np.isfinite(x1).all()
        assert not np.allclose(x, x1)

    def test_decorrelation_per_block(self, qg):
        qg.create_decorrelation_matrix(r={"q": 2, "psi": 4})
        L = qg.get_decorrelation_matrix()
        assert L.shape == (qg.dim, qg.dim)
        assert np.allclose(L, L.T)
        # Cross-block (q-psi) is zero by default
        q_idx = 0
        psi_idx = qg.field_size  # first index of psi
        assert L[q_idx, psi_idx] == 0.0


# ----------------------------------------------------------------------
# SWEModel
# ----------------------------------------------------------------------
class TestSWEModel:
    @pytest.fixture(scope="class")
    def swe(self):
        # Small grid for fast tests
        return SWEModel(LMAX=4, dt=120.0, state_vars=["u", "v", "h"])

    def test_dimension(self, swe):
        # LMAX=4 -> Nlat=10, Nlon=20, field=200, dim=600
        assert swe.field_size == 200
        assert swe.get_number_of_variables() == 600

    def test_var_blocks(self, swe):
        assert list(swe.var_blocks.keys()) == ["u", "v", "h"]

    def test_state_vars_subset(self):
        m = SWEModel(LMAX=4, state_vars=["h"])
        assert m.get_number_of_variables() == m.field_size
        assert list(m.var_blocks.keys()) == ["h"]

    def test_state_vars_validation(self):
        with pytest.raises(ValueError, match="Unknown"):
            SWEModel(LMAX=4, state_vars=["foo"])
        with pytest.raises(ValueError, match="duplicates"):
            SWEModel(LMAX=4, state_vars=["u", "u"])

    def test_state_vars_canonical_order(self):
        # Even if the user passes them out of order, they're stored canonically
        m = SWEModel(LMAX=4, state_vars=["h", "u"])
        assert m.state_vars == ["u", "h"]

    def test_propagate(self, swe):
        x = swe.get_initial_condition()
        x1 = swe.propagate(x, np.array([0.0, 360.0]))  # 3 dt steps
        assert x1.shape == x.shape
        assert np.isfinite(x1).all()

    def test_decorrelation_per_block(self, swe):
        swe.create_decorrelation_matrix(r={"u": 2, "v": 2, "h": 4})
        L = swe.get_decorrelation_matrix()
        assert L.shape == (swe.dim, swe.dim)
        assert np.allclose(L, L.T)

    def test_diagnostic_fields(self, swe):
        x = swe.get_initial_condition()
        h = swe.get_field(x, "h")
        u = swe.get_field(x, "u")
        # h is layer height, should be O(2800) for default H0
        assert h.mean() > 1000
        # u_max from Williamson TC2 base flow ~ U0 = 38 m/s
        assert u.max() < 50
