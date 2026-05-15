# -*- coding: utf-8 -*-
"""
Iterative Sherman–Morrison–Woodbury solver.

Solves linear systems of the form

    [A_0 + sum_q w_q · Q^(q) · (Q^(q))^T] · Z = rhs                 (*)

where:
  - A_0 is large and structured (diagonal, sparse banded, or anything
    a caller can solve cheaply);
  - each Q^(q) is a "skinny" matrix (n × k_q with k_q ≪ n);
  - the weights w_q are non-negative scalars.

The solver never forms the full n × n system. Instead, it applies the
Woodbury identity once per Q^(q) (Algorithm 1 from Nino-Ruiz, Guzman,
Jabba 2021), reducing the problem to a tree of trivial systems whose
leaves are A_0^{-1} · (skinny matrix). The caller provides an A_0_solver
callable so the solver doesn't need to know whether A_0 is dense, sparse,
factorised, or special-structured.

Usage
-----

>>> # Solve [R + Q1 Q1^T + Q2 Q2^T] Z = rhs with R diagonal
>>> R_inv_diag = 1.0 / np.diag(R)
>>> A0_solver = lambda b: R_inv_diag[:, None] * b if b.ndim == 2 \\
...                       else R_inv_diag * b
>>> Z = woodbury_solve(A0_solver, [Q1, Q2], [1.0, 1.0], rhs)

The solver is the matrix-free engine behind EnKF, EnKF-LW, EnKF-RBLW,
and EnKF-Shrinkage-Binv in pyteda.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np


def woodbury_solve(
    A0_solver: Callable[[np.ndarray], np.ndarray],
    Q_list: Sequence[np.ndarray],
    weights: Sequence[float],
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve [A_0 + sum_q w_q Q^(q) Q^(q)^T] Z = rhs without forming the
    n × n left-hand side.

    Parameters
    ----------
    A0_solver : callable
        Function ``b -> A_0^{-1} @ b``. Must accept ``b`` of shape
        ``(n,)``, ``(n, m)``, or ``(n, k)`` and return the same shape.
        The solver is responsible for whatever internal representation
        of A_0 is fastest (dense LU, sparse splu, diagonal scaling, etc).
    Q_list : list of ndarray
        Skinny matrices Q^(q) of shape ``(n, k_q)`` with ``k_q ≪ n``.
        For typical EnKF use, Q^(q) is built from H · ΔX or similar.
    weights : list of float
        Non-negative scalar weights ``w_q``. Must have the same length
        as ``Q_list``. Pass ``[1.0] * len(Q_list)`` if no weighting.
    rhs : ndarray
        Right-hand side, shape ``(n,)`` or ``(n, m)``.

    Returns
    -------
    Z : ndarray
        Solution to (*), with the same shape as ``rhs``.

    Notes
    -----
    Builds the solution by iteratively applying the Woodbury identity:
    if A^{(q-1)} is the partial sum and we add w_q · Q^(q) · Q^(q)^T,
    then for any RHS,

        A^{(q)}^{-1} · b = A^{(q-1)}^{-1} · b
                         - A^{(q-1)}^{-1} · Q^(q) · S_q^{-1}
                                       · Q^(q)^T · A^{(q-1)}^{-1} · b

    with S_q = I/w_q + Q^(q)^T · A^{(q-1)}^{-1} · Q^(q) ∈ R^{k_q × k_q}.

    Each step replaces a single n × n solve by:
      - one A^{(q-1)}-solve on rhs   (1 RHS, returns shape rhs);
      - one A^{(q-1)}-solve on Q^(q) (k_q RHS, returns n × k_q);
      - one solve in R^{k_q × k_q}.

    Total work after L levels: 1 + sum_q k_q  applications of A_0^{-1},
    one k_q × k_q solve per level. For typical EnKF use n ≫ k_q ≪ N,
    so the cost is O(N · cost_of_A0_solve) which is far less than O(n^3).
    """
    if len(Q_list) != len(weights):
        raise ValueError(
            f"Q_list has length {len(Q_list)} but weights has "
            f"length {len(weights)}."
        )

    # Promote rhs to 2-D for uniform handling, remember original shape.
    rhs = np.asarray(rhs)
    rhs_was_1d = (rhs.ndim == 1)
    if rhs_was_1d:
        rhs = rhs[:, None]

    # z is "current" A^{(q)^{-1}} rhs across iterations.
    # We start with q=0: A^{(0)} = A_0, so z = A_0^{-1} rhs.
    z = A0_solver(rhs)
    if z.ndim == 1:
        z = z[:, None]

    # As we incorporate each Q^(q) we also update solves of all prior
    # right-hand-sides we'll need for the next level. To avoid quadratic
    # bookkeeping we follow the simpler iterative form: at level q,
    # update z directly using the solver A^{(q-1)} we have at hand —
    # which is itself maintained recursively as a closure that knows
    # how to apply previous Woodbury corrections.

    # Closure holding the current A^{(q)^{-1}} as a function.
    apply_inv: Callable[[np.ndarray], np.ndarray] = (
        lambda b, _solve=A0_solver: _solve(b)
    )

    for Q, w in zip(Q_list, weights):
        Q = np.asarray(Q)
        if Q.ndim != 2:
            raise ValueError(f"Q must be 2-D, got shape {Q.shape}.")
        n, k = Q.shape
        if k == 0 or w == 0:
            continue   # No-op term.

        # Apply current A^{(q-1)^{-1}} to Q. This is k solves.
        Q_inv = apply_inv(Q)
        if Q_inv.ndim == 1:
            Q_inv = Q_inv[:, None]

        # Build the k × k Schur complement
        #   S = I / w + Q^T · A^{(q-1)^{-1}} · Q
        S = np.eye(k) / w + Q.T @ Q_inv

        # Update z = A^{(q)^{-1}} rhs:
        #   z_new = z - Q_inv · S^{-1} · (Q^T · z)
        QtZ = Q.T @ z                         # (k × m)
        S_inv_QtZ = np.linalg.solve(S, QtZ)   # (k × m)
        z = z - Q_inv @ S_inv_QtZ             # (n × m)

        # Build the new closure that applies A^{(q)^{-1}} so that the
        # next iteration sees the right operator. Capture by default
        # arguments to avoid late-binding of loop variables.
        def make_next(prev_apply, Q_=Q, Q_inv_=Q_inv, S_=S, w_=w):
            def apply_q(b):
                b = np.asarray(b)
                squeeze = (b.ndim == 1)
                if squeeze:
                    b = b[:, None]
                base = prev_apply(b)
                if base.ndim == 1:
                    base = base[:, None]
                Qtb = Q_.T @ base
                S_inv_Qtb = np.linalg.solve(S_, Qtb)
                out = base - Q_inv_ @ S_inv_Qtb
                return out[:, 0] if squeeze else out
            return apply_q

        apply_inv = make_next(apply_inv)

    if rhs_was_1d:
        z = z[:, 0]
    return z


# ----------------------------------------------------------------------
# Convenience builders for common A_0 cases
# ----------------------------------------------------------------------

def diagonal_solver(diag: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """A_0 is a diagonal matrix; return a closure that solves A_0 x = b.

    Parameters
    ----------
    diag : ndarray, shape (n,)
        The diagonal entries of A_0. All must be non-zero.

    Returns
    -------
    solver : callable
        ``solver(b)`` returns ``b / diag`` for ``b`` of shape (n,) or (n, m).
    """
    inv = 1.0 / np.asarray(diag, dtype=float)

    def solve(b):
        b = np.asarray(b)
        if b.ndim == 1:
            return inv * b
        return inv[:, None] * b

    return solve


def dense_lu_solver(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """A_0 is a dense matrix; factorise once and reuse for many solves."""
    from scipy.linalg import lu_factor, lu_solve
    lu_piv = lu_factor(np.asarray(A, dtype=float))

    def solve(b):
        return lu_solve(lu_piv, np.asarray(b, dtype=float))

    return solve


def sparse_lu_solver(A) -> Callable[[np.ndarray], np.ndarray]:
    """A_0 is a sparse matrix; factorise once with splu and reuse."""
    from scipy.sparse import csc_matrix, issparse
    from scipy.sparse.linalg import splu
    if not issparse(A):
        A = csc_matrix(np.asarray(A, dtype=float))
    else:
        A = A.tocsc()
    fact = splu(A)

    def solve(b):
        b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            return fact.solve(b)
        # splu.solve handles 2-D input column-by-column already
        return fact.solve(b)

    return solve
