# -*- coding: utf-8 -*-
"""
Numerical correctness tests for the new EnKF-LW, EnKF-RBLW and the
Woodbury solver. These pin the implementation to the closed-form
formulas of Nino-Ruiz, Guzman, Jabba (2021):

  - α_LW   from eq. (16) (paper) / eq. (22b) (computational form)
  - α_RBLW from eq. (17)
  - The iterative Woodbury solver to its dense reference inverse.
"""

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Direct, slow reference implementations (paper eqs.)
# ----------------------------------------------------------------------
def _alpha_LW_direct(DX: np.ndarray) -> tuple[float, float]:
    """α_LW computed via the original eq. (16): O(n²) but unambiguous."""
    n, N = DX.shape
    Pb = (DX @ DX.T) / (N - 1)

    num = 0.0
    for e in range(N):
        dx = DX[:, e:e+1]
        Z = dx @ dx.T
        num += np.linalg.norm(Pb - Z, ord="fro") ** 2

    tr_Pb2 = np.trace(Pb @ Pb)
    tr_Pb = np.trace(Pb)
    denom = N ** 2 * (tr_Pb2 - (tr_Pb ** 2) / n)
    if abs(denom) < 1e-300:
        alpha = 1.0
    else:
        alpha = min(num / denom, 1.0)
    mu = tr_Pb / n
    return float(mu), float(alpha)


def _alpha_RBLW_direct(DX: np.ndarray) -> tuple[float, float]:
    """α_RBLW computed via eq. (17) directly with the dense covariance."""
    n, N = DX.shape
    Pb = (DX @ DX.T) / (N - 1)
    tr_Pb2 = np.trace(Pb @ Pb)
    tr_Pb = np.trace(Pb)
    num = ((N - 2) / n) * tr_Pb2 + tr_Pb ** 2
    denom = (N + 2) * (tr_Pb2 - (tr_Pb ** 2) / n)
    if abs(denom) < 1e-300:
        alpha = 1.0
    else:
        alpha = min(num / denom, 1.0)
    mu = tr_Pb / n
    return float(mu), float(alpha)


# ----------------------------------------------------------------------
# Fast SVD-based formulas in the implementation
# ----------------------------------------------------------------------
class TestAlphaLW:
    def test_matches_direct_formula(self):
        from pyteda.analysis.analysis_enkf_lw import AnalysisEnKFLW
        rng = np.random.default_rng(0)
        n, N = 30, 8
        DX = rng.standard_normal((n, N))
        DX -= DX.mean(axis=1, keepdims=True)

        mu_d, alpha_d = _alpha_LW_direct(DX)
        sigma = np.linalg.svd(DX, full_matrices=False, compute_uv=False)
        mu_f, alpha_f = AnalysisEnKFLW._alpha_lw(sigma, DX, N=N, n=n)

        assert abs(mu_f - mu_d) < 1e-10
        assert abs(alpha_f - alpha_d) < 1e-10

    def test_alpha_in_unit_interval(self):
        from pyteda.analysis.analysis_enkf_lw import AnalysisEnKFLW
        rng = np.random.default_rng(1)
        for trial in range(10):
            n = rng.integers(20, 100)
            N = rng.integers(5, 20)
            DX = rng.standard_normal((n, N))
            DX -= DX.mean(axis=1, keepdims=True)
            sigma = np.linalg.svd(DX, full_matrices=False, compute_uv=False)
            _, alpha = AnalysisEnKFLW._alpha_lw(sigma, DX, N=N, n=n)
            assert 0.0 <= alpha <= 1.0


class TestAlphaRBLW:
    def test_matches_direct_formula(self):
        from pyteda.analysis.analysis_enkf_rblw import AnalysisEnKFRBLW
        rng = np.random.default_rng(2)
        n, N = 25, 6
        DX = rng.standard_normal((n, N))
        DX -= DX.mean(axis=1, keepdims=True)

        mu_d, alpha_d = _alpha_RBLW_direct(DX)
        sigma = np.linalg.svd(DX, full_matrices=False, compute_uv=False)
        mu_f, alpha_f = AnalysisEnKFRBLW._alpha_rblw(sigma, N=N, n=n)

        assert abs(mu_f - mu_d) < 1e-10
        assert abs(alpha_f - alpha_d) < 1e-10

    def test_alpha_in_unit_interval(self):
        from pyteda.analysis.analysis_enkf_rblw import AnalysisEnKFRBLW
        rng = np.random.default_rng(3)
        for trial in range(10):
            n = rng.integers(20, 100)
            N = rng.integers(5, 20)
            DX = rng.standard_normal((n, N))
            DX -= DX.mean(axis=1, keepdims=True)
            sigma = np.linalg.svd(DX, full_matrices=False, compute_uv=False)
            _, alpha = AnalysisEnKFRBLW._alpha_rblw(sigma, N=N, n=n)
            assert 0.0 <= alpha <= 1.0


# ----------------------------------------------------------------------
# Woodbury solver vs np.linalg.solve
# ----------------------------------------------------------------------
class TestWoodburySolver:
    def test_diagonal_A0_one_correction(self):
        from pyteda.analysis._woodbury_solver import (
            woodbury_solve, diagonal_solver,
        )
        rng = np.random.default_rng(10)
        n, k, m = 60, 5, 4
        diag = rng.uniform(1.0, 5.0, n)
        Q = rng.standard_normal((n, k))
        rhs = rng.standard_normal((n, m))

        A_full = np.diag(diag) + 0.7 * (Q @ Q.T)
        Z_ref = np.linalg.solve(A_full, rhs)
        Z_wb = woodbury_solve(diagonal_solver(diag), [Q], [0.7], rhs)
        assert np.max(np.abs(Z_wb - Z_ref)) < 1e-10

    def test_diagonal_A0_two_corrections(self):
        from pyteda.analysis._woodbury_solver import (
            woodbury_solve, diagonal_solver,
        )
        rng = np.random.default_rng(11)
        n, k1, k2, m = 80, 6, 4, 3
        diag = rng.uniform(2.0, 8.0, n)
        Q1 = rng.standard_normal((n, k1))
        Q2 = rng.standard_normal((n, k2))
        rhs = rng.standard_normal((n, m))

        A_full = np.diag(diag) + 1.3 * (Q1 @ Q1.T) + 0.4 * (Q2 @ Q2.T)
        Z_ref = np.linalg.solve(A_full, rhs)
        Z_wb = woodbury_solve(diagonal_solver(diag), [Q1, Q2],
                              [1.3, 0.4], rhs)
        assert np.max(np.abs(Z_wb - Z_ref)) < 1e-10

    def test_dense_A0(self):
        from pyteda.analysis._woodbury_solver import (
            woodbury_solve, dense_lu_solver,
        )
        rng = np.random.default_rng(12)
        n, k, m = 40, 5, 2
        A0 = rng.standard_normal((n, n))
        A0 = A0 @ A0.T + 5.0 * np.eye(n)
        Q = rng.standard_normal((n, k))
        rhs = rng.standard_normal((n, m))

        A_full = A0 + 0.9 * (Q @ Q.T)
        Z_ref = np.linalg.solve(A_full, rhs)
        Z_wb = woodbury_solve(dense_lu_solver(A0), [Q], [0.9], rhs)
        assert np.max(np.abs(Z_wb - Z_ref)) < 1e-10

    def test_sparse_A0(self):
        from pyteda.analysis._woodbury_solver import (
            woodbury_solve, sparse_lu_solver,
        )
        from scipy.sparse import diags
        rng = np.random.default_rng(13)
        n, k, m = 100, 5, 3
        # Tridiagonal A_0
        main = rng.uniform(2.0, 5.0, n)
        off  = rng.uniform(-1.0, 1.0, n - 1)
        A0 = diags([off, main, off], [-1, 0, 1], format="csc")
        Q = rng.standard_normal((n, k))
        rhs = rng.standard_normal((n, m))

        A_full = A0.toarray() + 0.6 * (Q @ Q.T)
        Z_ref = np.linalg.solve(A_full, rhs)
        Z_wb = woodbury_solve(sparse_lu_solver(A0), [Q], [0.6], rhs)
        assert np.max(np.abs(Z_wb - Z_ref)) < 1e-10

    def test_one_d_rhs(self):
        from pyteda.analysis._woodbury_solver import (
            woodbury_solve, diagonal_solver,
        )
        rng = np.random.default_rng(14)
        n, k = 50, 4
        diag = rng.uniform(1.0, 3.0, n)
        Q = rng.standard_normal((n, k))
        rhs = rng.standard_normal(n)
        A_full = np.diag(diag) + Q @ Q.T
        Z_ref = np.linalg.solve(A_full, rhs)
        Z_wb = woodbury_solve(diagonal_solver(diag), [Q], [1.0], rhs)
        assert Z_wb.shape == (n,)
        assert np.max(np.abs(Z_wb - Z_ref)) < 1e-10


# ----------------------------------------------------------------------
# End-to-end: filters run and reduce error on a small synthetic problem
# ----------------------------------------------------------------------
class TestNewFiltersEndToEnd:
    """The new filters must run on a small Lorenz96 scenario."""

    def test_enkf_lw_runs(self, small_scenario, small_lorenz96):
        from pyteda.simulation import Simulation
        from pyteda.analysis.analysis_factory import AnalysisFactory
        analysis = AnalysisFactory("enkf-lw", model=small_lorenz96).create_analysis()
        sim = Simulation.from_scenario(small_scenario, analysis)
        sim.run()
        assert np.all(np.isfinite(sim.error_a))

    def test_enkf_rblw_runs(self, small_scenario, small_lorenz96):
        from pyteda.simulation import Simulation
        from pyteda.analysis.analysis_factory import AnalysisFactory
        analysis = AnalysisFactory("enkf-rblw", model=small_lorenz96).create_analysis()
        sim = Simulation.from_scenario(small_scenario, analysis)
        sim.run()
        assert np.all(np.isfinite(sim.error_a))

    def test_enkf_shrinkage_binv_alias_works(self, small_scenario, small_lorenz96):
        """The legacy registry key 'enkf-shrinkage-precision' still works."""
        from pyteda.simulation import Simulation
        from pyteda.analysis.analysis_factory import AnalysisFactory
        analysis = AnalysisFactory(
            "enkf-shrinkage-precision", model=small_lorenz96,
        ).create_analysis()
        sim = Simulation.from_scenario(small_scenario, analysis)
        sim.run()
        assert np.all(np.isfinite(sim.error_a))

    def test_enkf_shrinkage_binv_new_key(self, small_scenario, small_lorenz96):
        from pyteda.simulation import Simulation
        from pyteda.analysis.analysis_factory import AnalysisFactory
        analysis = AnalysisFactory(
            "enkf-shrinkage-binv", model=small_lorenz96,
        ).create_analysis()
        sim = Simulation.from_scenario(small_scenario, analysis)
        sim.run()
        assert np.all(np.isfinite(sim.error_a))


# ----------------------------------------------------------------------
# Three-criteria shrinkage-Binv tests
# ----------------------------------------------------------------------
class TestShrinkageBinvCriteria:
    """The 3 criteria (mse/stein/da) must:
       - produce α ∈ [0, 1]
       - produce *different* α in general (not all identical)
       - run end-to-end on a small Lorenz96 scenario
    """

    def test_alpha_in_unit_interval(self, small_scenario, small_lorenz96):
        """All four criteria must give α ∈ [0,1] over a real run."""
        from pyteda.simulation import Simulation
        from pyteda.analysis.analysis_factory import AnalysisFactory
        for crit in ["mse", "stein", "da"]:
            analysis = AnalysisFactory(
                "enkf-shrinkage-binv", model=small_lorenz96,
                criterion=crit,
            ).create_analysis()
            sim = Simulation.from_scenario(small_scenario, analysis)
            sim.run()
            # alpha_ is stored after each analysis step
            assert 0.0 <= analysis.alpha_ <= 1.0, (
                f"criterion={crit}, alpha_={analysis.alpha_}"
            )
            assert np.all(np.isfinite(sim.error_a))

    def test_factory_auto_selects_criterion(self, small_lorenz96):
        """Each registry key picks the right criterion automatically."""
        from pyteda.analysis.analysis_factory import AnalysisFactory
        cases = {
            "enkf-shrinkage-binv":      "mse",          # default
            "enkf-shrinkage-binv-mse":  "mse",
            "enkf-shrinkage-binv-stein": "stein",
            "enkf-shrinkage-binv-da":   "da",
            "enkf-shrinkage-precision": "mse",   # legacy alias
        }
        for key, expected in cases.items():
            analysis = AnalysisFactory(
                key, model=small_lorenz96,
            ).create_analysis()
            assert analysis.criterion == expected, (
                f"key={key}: expected criterion {expected!r}, "
                f"got {analysis.criterion!r}"
            )

    def test_factory_override_criterion(self, small_lorenz96):
        """The user can override the implicit criterion."""
        from pyteda.analysis.analysis_factory import AnalysisFactory
        # Even though the key implies 'mse', explicit kwarg wins.
        analysis = AnalysisFactory(
            "enkf-shrinkage-binv-mse", model=small_lorenz96,
            criterion="stein",
        ).create_analysis()
        assert analysis.criterion == "stein"

    def test_criteria_produce_different_alphas(self, small_scenario,
                                                 small_lorenz96):
        """The four criteria should NOT all give the same α — that would
        mean they're not really different methods. This is a sanity check
        on a real run."""
        from pyteda.simulation import Simulation
        from pyteda.analysis.analysis_factory import AnalysisFactory
        alphas = {}
        for crit in ["mse", "stein", "da"]:
            analysis = AnalysisFactory(
                "enkf-shrinkage-binv", model=small_lorenz96,
                criterion=crit,
            ).create_analysis()
            sim = Simulation.from_scenario(small_scenario, analysis)
            sim.run()
            alphas[crit] = analysis.alpha_
        # At least one pair must differ noticeably
        values = list(alphas.values())
        max_diff = max(abs(v1 - v2)
                       for v1 in values for v2 in values)
        assert max_diff > 1e-4, (
            f"All criteria gave essentially the same α: {alphas}"
        )

    def test_invalid_criterion_raises(self, small_lorenz96):
        from pyteda.analysis.analysis_enkf_shrinkage_precision import (
            AnalysisEnKFShrinkageBinv,
        )
        with pytest.raises(ValueError, match="criterion"):
            AnalysisEnKFShrinkageBinv(
                model=small_lorenz96, criterion="not_a_real_criterion",
            )
