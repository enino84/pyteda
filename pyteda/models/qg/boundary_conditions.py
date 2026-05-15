"""
boundary_conditions.py
======================
Boundary condition implementations for the 1.5-layer QG model.

Supported types
---------------
'dirichlet'  — psi = 0 on all four walls (closed basin, original QG-C behaviour)
'channel'    — periodic in x, Dirichlet (psi=0) at j=0 and j=m-1 (zonal channel)

Each BC object exposes the same interface:

    bc = make_bc('channel', m, n, h, F)

    psi  = bc.solve(q)           # Helmholtz inversion: (nabla^2 - F) psi = q
    J    = bc.arakawa(h, psi, q) # Arakawa Jacobian J(psi, q)
    L    = bc.laplacian(h, A)    # Centred Laplacian nabla^2 A
    dy   = bc.dpsidy(h, psi)     # d(psi)/dy (centred differences)

Physics notes
-------------
Dirichlet (closed basin)
    psi = 0 on all four boundaries. The streamlines follow the walls.
    This is the geometry of the original QG-C model (Sakov 2024).

Channel (zonal)
    Periodic in the zonal (x) direction, rigid walls at north/south.
    The zonal periodicity is enforced in the Helmholtz solver via a
    block-circulant matrix structure, and in the differential operators
    via wrap-around stencils at i=0 and i=n-1.
    This is the standard geometry for ocean jet and baroclinic instability
    studies (Pedlosky 1987; Vallis 2006).

References
----------
Arakawa, A. (1966). Computational design for long-term numerical integration
    of the equations of fluid motion. J. Comput. Phys. 1, 119-143.

Pedlosky, J. (1987). Geophysical Fluid Dynamics. Springer.

Sakov, P. (2024). QG-C: A quasi-geostrophic model in C.
    https://github.com/sakov/qg-c

Vallis, G.K. (2006). Atmospheric and Oceanic Fluid Dynamics. Cambridge UP.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import diags, block_diag, eye, lil_matrix
from scipy.sparse.linalg import factorized


# =============================================================================
# Public factory
# =============================================================================

def make_bc(kind: str, m: int, n: int, h: float, F: float):
    """
    Instantiate a boundary-condition handler.

    Parameters
    ----------
    kind : 'dirichlet' or 'channel'
    m, n : grid dimensions (rows=y, cols=x)
    h    : uniform grid spacing
    F    : Froude number parameter (stratification)

    Returns
    -------
    DirichletBC or ChannelBC instance.
    """
    kind = kind.lower()
    if kind == 'dirichlet':
        return DirichletBC(m, n, h, F)
    elif kind == 'channel':
        return ChannelBC(m, n, h, F)
    else:
        raise ValueError(
            f"Unknown BC kind '{kind}'. Choose 'dirichlet' or 'channel'."
        )


# =============================================================================
# Dirichlet (closed basin) — original QG-C geometry
# =============================================================================

class DirichletBC:
    """
    Homogeneous Dirichlet boundary conditions: psi = 0 on all four walls.

    The Helmholtz equation (nabla^2 - F) psi = q is solved via direct
    sparse LU factorisation (SuperLU via scipy). The boundary rows of the
    linear system are replaced by identity rows enforcing psi=0.
    """

    kind = 'dirichlet'

    def __init__(self, m: int, n: int, h: float, F: float):
        self.m = m
        self.n = n
        self._lu = factorized(_build_helmholtz_dirichlet(m, n, h, F))

    # ── Helmholtz solver ─────────────────────────────────────────────────────

    def solve(self, q: np.ndarray) -> np.ndarray:
        """Invert (nabla^2 - F) psi = q with psi=0 on all walls."""
        m, n = self.m, self.n
        mn   = m * n
        rhs  = q.ravel().copy()

        # enforce zero on boundary rows
        border = _dirichlet_border_idx(m, n)
        rhs[border] = 0.0
        return self._lu(rhs).reshape(m, n)

    # ── differential operators ───────────────────────────────────────────────

    @staticmethod
    def arakawa(h: float, psi: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Arakawa Jacobian with zero-flux boundary (psi=0 on walls)."""
        return _arakawa_dirichlet(h, psi, q)

    @staticmethod
    def laplacian(h: float, A: np.ndarray) -> np.ndarray:
        """Centred Laplacian with Dirichlet boundary (zero outside)."""
        return _laplacian_dirichlet(h, A)

    @staticmethod
    def dpsidy(h: float, psi: np.ndarray) -> np.ndarray:
        """d(psi)/dy with Dirichlet stencil (interior only)."""
        return _dpsidy_dirichlet(h, psi)


# =============================================================================
# Channel (zonal) — periodic in x, Dirichlet in y
# =============================================================================

class ChannelBC:
    """
    Channel boundary conditions: periodic in x, psi=0 at y=0 and y=Ly.

    The Helmholtz solver uses a block-circulant structure in the zonal
    direction and a tridiagonal structure in the meridional direction.
    The wrap-around connections (i=n-1 to i=0) are included explicitly
    in the sparse matrix before LU factorisation.

    All differential operators use np.roll for periodic wrapping in x,
    and one-sided / zero-BC treatment at the north/south walls.

    Notes
    -----
    The grid has n points in x with spacing h. Periodicity means the
    point at i=n is identified with i=0, so the domain length in x is
    L_x = n*h (not (n-1)*h as in the Dirichlet case). The parameter
    passed to the model should be lx = n*h for consistency.

    In the channel geometry the zonal mean of q is generally non-zero
    and is advected correctly by the periodic operators.
    """

    kind = 'channel'

    def __init__(self, m: int, n: int, h: float, F: float):
        self.m = m
        self.n = n
        self._lu = factorized(_build_helmholtz_channel(m, n, h, F))

    # ── Helmholtz solver ─────────────────────────────────────────────────────

    def solve(self, q: np.ndarray) -> np.ndarray:
        """
        Invert (nabla^2 - F) psi = q with periodic x, psi=0 at y-walls.

        The RHS is zeroed on the north/south boundary rows before solving.
        """
        m, n = self.m, self.n
        mn   = m * n
        rhs  = q.ravel().copy()

        # only north/south walls have Dirichlet rows
        ns_border = list(range(n)) + list(range(mn - n, mn))
        rhs[ns_border] = 0.0
        return self._lu(rhs).reshape(m, n)

    # ── differential operators ───────────────────────────────────────────────

    @staticmethod
    def arakawa(h: float, psi: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Arakawa Jacobian with periodic-x, zero-flux north/south."""
        return _arakawa_channel(h, psi, q)

    @staticmethod
    def laplacian(h: float, A: np.ndarray) -> np.ndarray:
        """Centred Laplacian with periodic-x, Dirichlet north/south."""
        return _laplacian_channel(h, A)

    @staticmethod
    def dpsidy(h: float, psi: np.ndarray) -> np.ndarray:
        """d(psi)/dy with periodic-x stencil."""
        return _dpsidy_channel(h, psi)


# =============================================================================
# Helmholtz matrix builders
# =============================================================================

def _dirichlet_border_idx(m: int, n: int) -> list:
    mn = m * n
    return (
        list(range(n))           # south wall  (j=0)
      + list(range(mn-n, mn))    # north wall  (j=m-1)
      + list(range(0, mn, n))    # west wall   (i=0)
      + list(range(n-1, mn, n))  # east wall   (i=n-1)
    )


def _build_helmholtz_dirichlet(m: int, n: int, h: float, F: float):
    """
    Sparse (mn x mn) matrix for (nabla^2 - F) psi = q with Dirichlet BCs.
    Boundary rows are replaced by identity (psi_b = 0).
    """
    mn  = m * n
    h2  = 1.0 / (h * h)

    diag_vals = np.full(mn, -(4.0 * h2 + F))
    off_vals  = np.full(mn,  h2)

    # mark boundary nodes
    border = np.zeros(mn, dtype=bool)
    border[:n]      = True   # south
    border[-n:]     = True   # north
    border[::n]     = True   # west
    border[n-1::n]  = True   # east

    # identity on boundary rows
    diag_vals[border] = 1.0

    # zero off-diagonal couplings that cross east-west wrap (Dirichlet: no wrap)
    off_e = off_vals.copy();  off_e[border] = 0.0
    off_w = off_vals.copy();  off_w[border] = 0.0
    off_n = off_vals.copy();  off_n[border] = 0.0
    off_s = off_vals.copy();  off_s[border] = 0.0

    # kill east-west wrap artefacts at column boundaries
    for j in range(m):
        k = j * n + (n - 1)
        off_e[k] = 0.0
        off_w[k] = 0.0

    return diags(
        [off_s[:mn-n], off_w[:mn-1], diag_vals, off_e[:mn-1], off_n[:mn-n]],
        [-n, -1, 0, 1, n],
        shape=(mn, mn),
        format='csr',
    )


def _build_helmholtz_channel(m: int, n: int, h: float, F: float):
    """
    Sparse (mn x mn) matrix for (nabla^2 - F) psi = q with channel BCs:
    periodic in x (zonal), Dirichlet at j=0 and j=m-1 (north/south walls).

    The periodic wrap-around at i=0 and i=n-1 introduces off-diagonal
    entries at positions ±(n-1) relative to the main diagonal (within
    each meridional row). These are inserted explicitly using lil_matrix
    before converting to CSR for factorisation.

    Interior stencil for node (j, i):
        psi[j,i-1] + psi[j,i+1] + psi[j-1,i] + psi[j+1,i]
        - (4/h^2 + F) * psi[j,i] = q[j,i]

    with i-1 and i+1 computed mod n (periodic wrap).
    """
    mn  = m * n
    h2  = 1.0 / (h * h)

    # build in lil format for easy wrap-around insertion
    A = lil_matrix((mn, mn))

    for j in range(m):
        for i in range(n):
            row = j * n + i

            # north/south walls — Dirichlet identity
            if j == 0 or j == m - 1:
                A[row, row] = 1.0
                continue

            # interior node
            A[row, row] = -(4.0 * h2 + F)

            # meridional neighbours (always interior for 1 < j < m-1)
            A[row, (j - 1) * n + i] = h2   # south
            A[row, (j + 1) * n + i] = h2   # north

            # zonal neighbours — periodic wrap
            i_west = (i - 1) % n
            i_east = (i + 1) % n
            A[row, j * n + i_west] = h2
            A[row, j * n + i_east] = h2

    return A.tocsr()


# =============================================================================
# Differential operators — Dirichlet version
# =============================================================================

def _arakawa_dirichlet(h: float, psi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Arakawa (1966) energy- and enstrophy-conserving Jacobian.
    Interior stencil only; boundary values are zero (Dirichlet psi=0).
    Identical to the original QG-C implementation.
    """
    c = 12.0 * h * h
    J = np.zeros_like(psi)

    pjm = psi[:-2,  1:-1];  pjp = psi[2:,  1:-1]
    pim = psi[1:-1, :-2];   pip = psi[1:-1, 2:]
    pmm = psi[:-2,  :-2];   pmp = psi[:-2,  2:]
    ppm = psi[2:,   :-2];   ppp = psi[2:,   2:]

    qjm = q[:-2,  1:-1];    qjp = q[2:,  1:-1]
    qim = q[1:-1, :-2];     qip = q[1:-1, 2:]
    qmm = q[:-2,  :-2];     qmp = q[:-2,  2:]
    qpm = q[2:,   :-2];     qpp = q[2:,   2:]

    J[1:-1, 1:-1] = (
          (pjm - pim) * qmm
        + (pmm + pjm - ppm - pjp) * qim
        + (pim - pjp) * qpm
        + (pmp + pip - pmm - pim) * qjm
        + (pim + ppm - pip - ppp) * qjp
        + (pip - pjm) * qmp
        + (pjp + ppp - pjm - pmp) * qip
        + (pjp - pip) * qpp
    ) / c
    return J


def _laplacian_dirichlet(h: float, A: np.ndarray) -> np.ndarray:
    """Centred Laplacian, interior only, zero on boundary."""
    h2 = 1.0 / (h * h)
    L  = np.zeros_like(A)
    L[1:-1, 1:-1] = (
        A[:-2, 1:-1] + A[2:, 1:-1] +
        A[1:-1, :-2] + A[1:-1, 2:] -
        4.0 * A[1:-1, 1:-1]
    ) * h2
    return L


def _dpsidy_dirichlet(h: float, psi: np.ndarray) -> np.ndarray:
    """d(psi)/dy — centred differences, interior only."""
    dpdy = np.zeros_like(psi)
    dpdy[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)
    return dpdy


# =============================================================================
# Differential operators — Channel version (periodic in x)
# =============================================================================

def _arakawa_channel(h: float, psi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Arakawa (1966) Jacobian with periodic-x boundary conditions.

    Strategy: pad psi and q with one periodic column on each side,
    then apply the standard Arakawa stencil. The slices pp[:-2,:],
    pp[1:-1,:], pp[2:,:] select exactly the (m-2) interior rows,
    and the result is assigned directly to J[1:-1, :].

    This avoids explicit modular index arithmetic while keeping the
    fully vectorised stencil intact.
    """
    m, n = psi.shape
    c    = 12.0 * h * h
    J    = np.zeros_like(psi)

    # periodic padding in x: prepend last column, append first column
    # pp shape: (m, n+2)
    def _pad_x(A):
        return np.concatenate([A[:, -1:], A, A[:, :1]], axis=1)

    pp = _pad_x(psi)
    qp = _pad_x(q)

    # After padding, column index k in pp corresponds to:
    #   k=0       -> i = n-1  (west wrap)
    #   k=1..n    -> i = 0..n-1
    #   k=n+1     -> i = 0    (east wrap)
    #
    # Slicing pp[r1:r2, c1:c2] for the stencil:
    #   pp[:-2,  1:-1] -> rows 0..m-3,   cols 1..n   -> shape (m-2, n)  (j-1, i)
    #   pp[2:,   1:-1] -> rows 2..m-1,   cols 1..n   -> shape (m-2, n)  (j+1, i)
    #   pp[1:-1, :-2]  -> rows 1..m-2,   cols 0..n-1 -> shape (m-2, n)  (j,   i-1 mod n)
    #   pp[1:-1, 2:]   -> rows 1..m-2,   cols 2..n+1 -> shape (m-2, n)  (j,   i+1 mod n)
    # etc. — all shapes are (m-2, n). ✓

    pjm = pp[:-2,  1:-1];   pjp = pp[2:,   1:-1]
    pim = pp[1:-1, :-2];    pip = pp[1:-1, 2:]
    pmm = pp[:-2,  :-2];    pmp = pp[:-2,  2:]
    ppm = pp[2:,   :-2];    ppp = pp[2:,   2:]

    qjm = qp[:-2,  1:-1];   qjp = qp[2:,   1:-1]
    qim = qp[1:-1, :-2];    qip = qp[1:-1, 2:]
    qmm = qp[:-2,  :-2];    qmp = qp[:-2,  2:]
    qpm = qp[2:,   :-2];    qpp = qp[2:,   2:]

    # all terms shape (m-2, n) — assign directly to interior rows
    J[1:-1, :] = (
          (pjm - pim) * qmm
        + (pmm + pjm - ppm - pjp) * qim
        + (pim - pjp) * qpm
        + (pmp + pip - pmm - pim) * qjm
        + (pim + ppm - pip - ppp) * qjp
        + (pip - pjm) * qmp
        + (pjp + ppp - pjm - pmp) * qip
        + (pjp - pip) * qpp
    ) / c
    return J


def _laplacian_channel(h: float, A: np.ndarray) -> np.ndarray:
    """
    Centred Laplacian with periodic-x, Dirichlet north/south.

    Uses np.roll for the zonal wrap; meridional stencil is standard.
    """
    h2 = 1.0 / (h * h)
    L  = np.zeros_like(A)

    A_west = np.roll(A, +1, axis=1)   # A[:, i-1 mod n]
    A_east = np.roll(A, -1, axis=1)   # A[:, i+1 mod n]

    # interior rows only (meridional Dirichlet)
    L[1:-1, :] = (
        A[:-2, :] + A[2:, :] +
        A_west[1:-1, :] + A_east[1:-1, :] -
        4.0 * A[1:-1, :]
    ) * h2
    return L


def _dpsidy_channel(h: float, psi: np.ndarray) -> np.ndarray:
    """
    d(psi)/dy — centred differences, periodic in x, interior rows only.
    All columns are computed (x is periodic so no special treatment needed).
    """
    dpdy = np.zeros_like(psi)
    dpdy[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * h)
    return dpdy
