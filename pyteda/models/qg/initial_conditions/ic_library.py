"""
initial_conditions/ic_library.py
=================================
Initial condition library for the QG model.

All functions return (psi, q) as numpy arrays of shape (m, n),
satisfying the boundary conditions specified by bc_kind.

Available ICs
-------------
zero         — rest state (psi = q = 0)
fourier      — random Fourier superposition with spectral slope 1/(kx^2+ky^2)
vortex       — one or more Gaussian vortex patches
dipole       — counter-rotating vortex pair (clean advection test case)
rossby_wave  — small-amplitude linear Rossby wave (known exact evolution)
band_noise   — spectrally band-limited random noise
restart_npz  — load from .npz file (qg_model output)
restart_nc   — load from NetCDF file (QG-C compatible)

Boundary condition compatibility
---------------------------------
All ICs respect the requested bc_kind:
  'dirichlet' — psi=0 enforced on all four walls
  'channel'   — psi=0 enforced only on north/south walls;
                x-periodicity is implicit (no explicit enforcement needed
                because the IC functions use sinusoidal or Gaussian
                forms that are continuous across the periodic boundary,
                or a nearest-neighbour wrap is applied for the Fourier IC).
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# =============================================================================
# Internal helpers
# =============================================================================

def _enforce_bc(psi: np.ndarray, bc_kind: str) -> np.ndarray:
    """Apply boundary conditions to psi."""
    if bc_kind == 'dirichlet':
        psi[0,  :] = 0.0
        psi[-1, :] = 0.0
        psi[:,  0] = 0.0
        psi[:, -1] = 0.0
    elif bc_kind == 'channel':
        # periodic in x — no east/west zeroing
        psi[0,  :] = 0.0
        psi[-1, :] = 0.0
    else:
        raise ValueError(f"Unknown bc_kind '{bc_kind}'.")
    return psi


def _q_from_psi(psi: np.ndarray, h: float, F: float) -> np.ndarray:
    """q = laplacian(psi) - F*psi  (centred differences, zero on boundaries)."""
    h2  = 1.0 / (h * h)
    lap = np.zeros_like(psi)
    lap[1:-1, 1:-1] = (
        psi[:-2, 1:-1] + psi[2:, 1:-1] +
        psi[1:-1, :-2] + psi[1:-1, 2:] -
        4.0 * psi[1:-1, 1:-1]
    ) * h2
    return lap - F * psi


# =============================================================================
# Initial condition functions
# =============================================================================

def zero(m, n, h, F, bc_kind='dirichlet', **kwargs):
    """
    Rest state: psi = q = 0.

    The model develops turbulence spontaneously from the wind forcing.
    Spin-up takes ~50 000 time units for the closed basin.
    """
    return np.zeros((m, n)), np.zeros((m, n))


def fourier(
    m, n, h, F,
    bc_kind: str   = 'dirichlet',
    amplitude: float = 1.0,
    kmax:      int   = 5,
    seed:      int   = 42,
    lx:        float = 1.0,
    **kwargs,
):
    """
    Random Fourier modes with amplitude ~ 1/(kx^2 + ky^2).

    For 'dirichlet': sine modes in both x and y, zero on all walls.
    For 'channel':   sine modes in y, complex exponentials in x
                     (ensuring x-periodicity).

    Parameters
    ----------
    amplitude : overall amplitude scale
    kmax      : maximum wavenumber in each direction
    seed      : random seed for reproducibility
    lx        : domain size (used for channel x-wavenumbers)
    """
    rng   = np.random.default_rng(seed)
    j_idx = np.arange(m)
    i_idx = np.arange(n)
    psi   = np.zeros((m, n))

    if bc_kind == 'dirichlet':
        for kj in range(1, kmax + 1):
            for ki in range(1, kmax + 1):
                amp  = rng.uniform(-1, 1) * amplitude / (kj**2 + ki**2)
                psi += amp * np.outer(
                    np.sin(kj * np.pi * j_idx / (m - 1)),
                    np.sin(ki * np.pi * i_idx / (n - 1)),
                )

    elif bc_kind == 'channel':
        # Meridional: sine (zero at j=0 and j=m-1)
        # Zonal: cosine + sine superposition (periodic over n points)
        for kj in range(1, kmax + 1):
            for ki in range(1, kmax + 1):
                amp_c = rng.uniform(-1, 1) * amplitude / (kj**2 + ki**2)
                amp_s = rng.uniform(-1, 1) * amplitude / (kj**2 + ki**2)
                zy    = np.sin(kj * np.pi * j_idx / (m - 1))
                zx_c  = np.cos(2 * np.pi * ki * i_idx / n)
                zx_s  = np.sin(2 * np.pi * ki * i_idx / n)
                psi  += amp_c * np.outer(zy, zx_c)
                psi  += amp_s * np.outer(zy, zx_s)

    psi = _enforce_bc(psi, bc_kind)
    q   = _q_from_psi(psi, h, F)
    return psi, q


def vortex(
    m, n, h, F,
    bc_kind:  str            = 'dirichlet',
    vortices: Optional[list] = None,
    lx:       float          = 1.0,
    **kwargs,
):
    """
    One or more Gaussian vortex patches.

    Parameters
    ----------
    vortices : list of dicts with keys:
        x   : normalised x-position in [0,1]  (default 0.5)
        y   : normalised y-position in [0,1]  (default 0.5)
        r   : radius as fraction of lx        (default 0.1)
        amp : amplitude (positive=cyclone, negative=anticyclone)
    lx : domain size

    Notes
    -----
    For 'channel', vortex positions near x=0 or x=1 will partially
    overlap the periodic boundary; the Gaussian tails wrap correctly
    because the differential operators are periodic.
    """
    if vortices is None:
        vortices = [{'x': 0.5, 'y': 0.5, 'r': 0.1, 'amp': 1.0}]

    x_arr = np.linspace(0, lx, n)
    y_arr = np.linspace(0, lx, m)
    X, Y  = np.meshgrid(x_arr, y_arr)
    psi   = np.zeros((m, n))

    for v in vortices:
        x0  = v.get('x', 0.5) * lx
        y0  = v.get('y', 0.5) * lx
        r   = v.get('r', 0.1) * lx
        amp = v.get('amp', 1.0)

        if bc_kind == 'channel':
            # periodic wrap in x: use minimum image distance
            dx  = X - x0
            dx -= lx * np.round(dx / lx)
            dy  = Y - y0
            psi += amp * np.exp(-(dx**2 + dy**2) / (2.0 * r**2))
        else:
            psi += amp * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * r**2))

    psi = _enforce_bc(psi, bc_kind)
    q   = _q_from_psi(psi, h, F)
    return psi, q


def dipole(
    m, n, h, F,
    bc_kind:    str   = 'dirichlet',
    x0:         float = 0.5,
    y0:         float = 0.5,
    r:          float = 0.1,
    amp:        float = 1.0,
    separation: float = 0.12,
    lx:         float = 1.0,
    **kwargs,
):
    """
    Counter-rotating vortex dipole.

    A dipole self-advects in a straight line (in the unforced, unbounded case),
    making it an excellent test for measuring numerical dissipation and
    phase errors across integrators and BC types.

    Parameters
    ----------
    x0, y0     : dipole centre (normalised, in [0,1])
    r          : vortex radius as fraction of lx
    amp        : positive vortex amplitude
    separation : centre-to-centre distance as fraction of lx
    lx         : domain size

    Notes
    -----
    Expected self-advection speed: U ~ amp / (4*pi*separation*lx).
    Use short integration times (t << lx/U) for clean convergence tests.
    """
    vortices = [
        {'x': x0, 'y': y0 + 0.5 * separation, 'r': r, 'amp':  amp},
        {'x': x0, 'y': y0 - 0.5 * separation, 'r': r, 'amp': -amp},
    ]
    return vortex(m, n, h, F, bc_kind=bc_kind, vortices=vortices, lx=lx)


def rossby_wave(
    m, n, h, F,
    bc_kind:   str   = 'dirichlet',
    kx:        int   = 2,
    ky:        int   = 1,
    amplitude: float = 1e-3,
    lx:        float = 1.0,
    **kwargs,
):
    """
    Small-amplitude linear Rossby wave.

    In the linearised QG model this produces an analytically propagating
    solution, enabling exact error measurement for convergence analysis.

    For 'dirichlet': psi = A * sin(kx*pi*x/lx) * sin(ky*pi*y/lx)
    For 'channel':   psi = A * cos(2*pi*kx*x/lx) * sin(ky*pi*y/lx)
                     (cosine in x ensures periodicity at x=0 and x=lx)

    Keep amplitude << 1 for the linear approximation to hold.

    Parameters
    ----------
    kx, ky    : wavenumber integers (half-wavelengths in Dirichlet; full in channel)
    amplitude : wave amplitude (must be << 1 for linear regime)
    lx        : domain size
    """
    x_arr = np.linspace(0, lx, n)
    y_arr = np.linspace(0, lx, m)
    X, Y  = np.meshgrid(x_arr, y_arr)

    if bc_kind == 'dirichlet':
        psi = amplitude * (
            np.sin(kx * np.pi * X / lx) *
            np.sin(ky * np.pi * Y / lx)
        )
    elif bc_kind == 'channel':
        psi = amplitude * (
            np.cos(2 * np.pi * kx * X / lx) *
            np.sin(ky * np.pi * Y / lx)
        )

    psi = _enforce_bc(psi, bc_kind)
    q   = _q_from_psi(psi, h, F)
    return psi, q


def band_noise(
    m, n, h, F,
    bc_kind:   str   = 'dirichlet',
    k_low:     int   = 2,
    k_high:    int   = 8,
    amplitude: float = 0.5,
    slope:     float = -3.0,
    seed:      int   = 42,
    lx:        float = 1.0,
    **kwargs,
):
    """
    Spectrally band-limited random noise.

    Generates random streamfunction excitation in wavenumber band
    [k_low, k_high] with a prescribed spectral slope. Useful for
    ensemble studies of uncertainty propagation.

    Parameters
    ----------
    k_low, k_high : wavenumber band limits (inclusive)
    amplitude     : overall amplitude scale
    slope         : spectral slope in log-log (default -3, QG-turbulence-like)
    seed          : random seed
    """
    rng   = np.random.default_rng(seed)
    j_idx = np.arange(m)
    i_idx = np.arange(n)
    psi   = np.zeros((m, n))

    for kj in range(k_low, k_high + 1):
        for ki in range(k_low, k_high + 1):
            k_tot = np.sqrt(kj**2 + ki**2)
            if not (k_low <= k_tot <= k_high):
                continue
            amp_k = amplitude * k_tot**slope

            if bc_kind == 'dirichlet':
                phase = rng.uniform(0, 2 * np.pi)
                psi  += amp_k * np.cos(phase) * np.outer(
                    np.sin(kj * np.pi * j_idx / (m - 1)),
                    np.sin(ki * np.pi * i_idx / (n - 1)),
                )
            elif bc_kind == 'channel':
                pc = rng.uniform(0, 2 * np.pi)
                ps = rng.uniform(0, 2 * np.pi)
                zy = np.sin(kj * np.pi * j_idx / (m - 1))
                psi += amp_k * np.cos(pc) * np.outer(zy, np.cos(2*np.pi*ki*i_idx/n))
                psi += amp_k * np.cos(ps) * np.outer(zy, np.sin(2*np.pi*ki*i_idx/n))

    psi = _enforce_bc(psi, bc_kind)
    q   = _q_from_psi(psi, h, F)
    return psi, q


def restart_npz(m, n, h, F, bc_kind='dirichlet', fname=None, record=-1, **kwargs):
    """Load psi, q from a .npz file produced by qg_model.py."""
    if fname is None:
        raise ValueError("restart_npz requires fname=<path>")
    data = np.load(fname)
    psi  = data['psi'][record].astype(float)
    q    = data['q'][record].astype(float) if 'q' in data else _q_from_psi(psi, h, F)
    return psi, q


def restart_nc(m, n, h, F, bc_kind='dirichlet', fname=None, record=-1, **kwargs):
    """Load psi, q from a NetCDF file (QG-C or qg_model.py output)."""
    if fname is None:
        raise ValueError("restart_nc requires fname=<path>")
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required: pip install netCDF4")
    ds  = nc.Dataset(fname, 'r')
    psi = np.array(ds.variables['psi'][record], dtype=float)
    q   = (np.array(ds.variables['q'][record], dtype=float)
           if 'q' in ds.variables else _q_from_psi(psi, h, F))
    ds.close()
    return psi, q


# =============================================================================
# Registry and public API
# =============================================================================

_IC_REGISTRY = {
    'zero':        zero,
    'fourier':     fourier,
    'vortex':      vortex,
    'dipole':      dipole,
    'rossby_wave': rossby_wave,
    'band_noise':  band_noise,
    'restart_npz': restart_npz,
    'restart_nc':  restart_nc,
}


def get_ic(name: str, m: int, n: int, h: float, F: float,
           bc_kind: str = 'dirichlet', **kwargs):
    """
    Generate an initial condition by name.

    Parameters
    ----------
    name    : IC name (see list_ics())
    m, n    : grid dimensions
    h       : grid spacing
    F       : Froude number parameter
    bc_kind : 'dirichlet' or 'channel'
    **kwargs: passed to the IC function

    Returns
    -------
    psi : np.ndarray (m, n)
    q   : np.ndarray (m, n)
    """
    if name not in _IC_REGISTRY:
        raise ValueError(
            f"Unknown IC '{name}'. Available: {sorted(_IC_REGISTRY.keys())}"
        )
    return _IC_REGISTRY[name](m=m, n=n, h=h, F=F, bc_kind=bc_kind, **kwargs)


def list_ics() -> list[str]:
    """Return sorted list of all available IC names."""
    return sorted(_IC_REGISTRY.keys())
