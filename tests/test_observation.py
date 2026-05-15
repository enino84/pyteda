# -*- coding: utf-8 -*-
"""Tests for pyteda.observation."""

import numpy as np
import pytest

from pyteda.observation import (
    LinearSelection,
    LinearMatrix,
    NonlinearOperator,
    IsotropicDiagonal,
    HeterogeneousDiagonal,
    DenseCovariance,
)


class TestLinearSelection:
    def test_apply_with_explicit_indices(self):
        op = LinearSelection(m=3, n_state=10, indices=np.array([0, 2, 4]))
        x = np.arange(10, dtype=float)
        assert np.array_equal(op.apply(x), [0.0, 2.0, 4.0])

    def test_apply_with_random_selection(self):
        op = LinearSelection(m=4, n_state=10, rng=np.random.default_rng(42))
        x = np.arange(10, dtype=float)
        y = op.apply(x)
        assert y.shape == (4,)
        assert op.dim_obs == 4
        assert op.dim_state == 10

    def test_apply_on_ensemble(self):
        op = LinearSelection(m=3, n_state=10, indices=np.array([0, 2, 4]))
        X = np.arange(50, dtype=float).reshape(10, 5)
        Y = op.apply(X)
        assert Y.shape == (3, 5)

    def test_jacobian_matches_apply(self):
        op = LinearSelection(m=3, n_state=10, indices=np.array([0, 2, 4]))
        H = op.linearize(np.zeros(10))
        assert H.shape == (3, 10)
        x = np.arange(10, dtype=float)
        # H is sparse but `@` returns a dense ndarray
        assert np.array_equal(H @ x, op.apply(x))

    def test_linearize_returns_sparse(self):
        """At high state dimension the dense H would be huge; LinearSelection
        must store it sparsely."""
        from scipy.sparse import issparse
        op = LinearSelection(m=100, n_state=10000,
                             indices=np.arange(100))
        H = op.linearize()
        assert issparse(H)
        # The sparse representation must hold exactly m=100 nonzeros
        assert H.nnz == 100

    def test_sparse_H_supports_typical_filter_ops(self):
        """The filters do H @ x, H @ X, H.T @ y, H @ Pb @ H.T — all of
        which must work with sparse H against dense operands."""
        op = LinearSelection(m=5, n_state=20,
                             indices=np.array([0, 3, 7, 12, 17]))
        H = op.linearize()
        rng = np.random.default_rng(0)
        x = rng.standard_normal(20)
        X = rng.standard_normal((20, 8))     # ensemble
        Pb = rng.standard_normal((20, 20))
        y = rng.standard_normal(5)

        # Each operation must (a) succeed and (b) match the dense baseline
        H_dense = np.zeros((5, 20))
        H_dense[np.arange(5), [0, 3, 7, 12, 17]] = 1.0

        # H @ x
        assert np.allclose(np.asarray(H @ x).ravel(), H_dense @ x)
        # H @ X
        assert np.allclose(np.asarray(H @ X), H_dense @ X)
        # H.T @ y
        assert np.allclose(np.asarray(H.T @ y).ravel(), H_dense.T @ y)
        # H @ Pb @ H.T  (innovation covariance shape)
        result = H @ Pb @ H.T
        if hasattr(result, "toarray"):
            result = result.toarray()
        assert np.allclose(np.asarray(result), H_dense @ Pb @ H_dense.T)

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError, match="Cannot observe"):
            LinearSelection(m=20, n_state=10)


class TestLinearMatrix:
    def test_apply(self):
        rng = np.random.default_rng(0)
        H = rng.standard_normal((4, 10))
        op = LinearMatrix(H)
        x = rng.standard_normal(10)
        assert np.allclose(op.apply(x), H @ x)
        assert np.allclose(op.linearize(x), H)


class TestNonlinearOperator:
    def test_apply(self):
        op = NonlinearOperator(h=lambda x: x[:3] ** 2, n_state=10, dim_obs=3)
        x = np.arange(1, 11, dtype=float)
        assert np.allclose(op.apply(x), [1.0, 4.0, 9.0])

    def test_apply_on_ensemble(self):
        op = NonlinearOperator(h=lambda x: x[:3] ** 2, n_state=10, dim_obs=3)
        X = np.arange(50, dtype=float).reshape(10, 5)
        Y = op.apply(X)
        assert Y.shape == (3, 5)

    def test_finite_difference_jacobian(self):
        op = NonlinearOperator(h=lambda x: 2.0 * x[:3], n_state=5, dim_obs=3)
        H = op.linearize(np.zeros(5))
        expected = np.zeros((3, 5))
        expected[0, 0] = expected[1, 1] = expected[2, 2] = 2.0
        assert np.allclose(H, expected, atol=1e-5)

    def test_provided_jacobian_used(self):
        def h(x): return x[:2] ** 2
        def jac(x):
            J = np.zeros((2, len(x)))
            J[0, 0] = 2 * x[0]
            J[1, 1] = 2 * x[1]
            return J
        op = NonlinearOperator(h=h, n_state=5, dim_obs=2, jacobian=jac)
        H = op.linearize(np.array([3.0, 4.0, 0.0, 0.0, 0.0]))
        assert H[0, 0] == 6.0
        assert H[1, 1] == 8.0


class TestIsotropicDiagonal:
    def test_dim_and_R(self):
        n = IsotropicDiagonal(std=0.5, dim=4)
        assert n.dim == 4
        assert np.allclose(np.diag(n.R), 0.25)
        assert np.allclose(n.R - np.diag(np.diag(n.R)), 0.0)

    def test_sample_shape(self):
        n = IsotropicDiagonal(std=0.5, dim=4)
        assert n.sample(np.random.default_rng(0)).shape == (4,)

    def test_sample_reproducible(self):
        n = IsotropicDiagonal(std=0.5, dim=4)
        s1 = n.sample(np.random.default_rng(42))
        s2 = n.sample(np.random.default_rng(42))
        assert np.array_equal(s1, s2)

    def test_R_inv(self):
        n = IsotropicDiagonal(std=2.0, dim=4)
        assert np.allclose(np.diag(n.R_inv), 0.25)

    def test_dim_lazy_binding(self):
        n = IsotropicDiagonal(std=0.5)
        n.bind_dim(7)
        assert n.dim == 7


class TestHeterogeneousDiagonal:
    def test_dim_and_R(self):
        stds = np.array([0.1, 0.2, 0.5])
        n = HeterogeneousDiagonal(stds=stds)
        assert n.dim == 3
        assert np.allclose(np.diag(n.R), stds ** 2)

    def test_R_inv(self):
        stds = np.array([0.5, 2.0])
        n = HeterogeneousDiagonal(stds=stds)
        assert np.allclose(np.diag(n.R_inv), [4.0, 0.25])

    def test_negative_stds_raises(self):
        with pytest.raises(ValueError):
            HeterogeneousDiagonal(stds=np.array([0.1, -0.2]))


class TestDenseCovariance:
    def test_dim(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4))
        R = A @ A.T + 0.1 * np.eye(4)
        n = DenseCovariance(R)
        assert n.dim == 4
        assert np.allclose(n.R, R)

    def test_sample_empirical_cov_close_to_R(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 3))
        R = A @ A.T + 0.5 * np.eye(3)
        n = DenseCovariance(R)
        samples = np.array([n.sample(rng) for _ in range(5000)])
        emp_cov = np.cov(samples.T)
        assert np.allclose(emp_cov, R, atol=0.15)
