"""
qg_model_v2.py
==============
1.5-layer Quasi-Geostrophic Model — extended for SoftwareX submission.

This module is a backward-compatible extension of qg_model.py (Sakov port).
New capabilities:

  1. Pluggable time integrators via the integrators/ library
     (Euler, Midpoint, RK4, DP5, SSPRK3, AB2, AB3, AB4,
      Leapfrog+RA, Leapfrog+RAW)

  2. Selectable boundary conditions via boundary_conditions.py
     - 'dirichlet' : psi=0 on all walls (closed basin, original behaviour)
     - 'channel'   : periodic in x, psi=0 at north/south (zonal channel)

  3. Extended initial conditions via initial_conditions/ library
     (zero, fourier, vortex, dipole, rossby_wave, band_noise,
      restart_npz, restart_nc)

  4. Unchanged physics, grid, output format and CLI relative to v1.

PHYSICS
-------
Potential vorticity equation:
    dq/dt = -r·J(psi,q) - rkb·zeta + rkh·nabla²zeta - rkh2·nabla⁴zeta
            + curl(tau) - dpsi/dy
    q = nabla²psi - F·psi

USAGE (Python API)
------------------
    from qg_model_v2 import QGModel, QGParams

    prm = QGParams(scheme='ab3', bc='channel', tend=5000)
    m   = QGModel(prm)
    psi, q = m.run('dipole', separation=0.12)

USAGE (command line)
--------------------
    python qg_model_v2.py --scheme ab3 --bc channel --tend 5000 \\
                          --ic dipole --out result.npz

REFERENCE MODEL
---------------
Sakov, P. (2024). QG-C: A quasi-geostrophic model in C.
    https://github.com/sakov/qg-c

DEPENDENCIES
------------
    numpy >= 1.24
    scipy >= 1.10
    matplotlib >= 3.7  (for analysis scripts only)
"""

import numpy as np
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# local modules (vendored as subpackages of pyteda.models.qg)
from .boundary_conditions import make_bc
from .integrators import get_integrator, list_integrators
from .initial_conditions import get_ic, list_ics

try:
    import netCDF4 as _nc
    HAS_NC = True
except ImportError:
    HAS_NC = False


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class QGParams:
    # ── Grid ──────────────────────────────────────────────────────────────────
    mrefin: int   = 7
    nx1:    int   = 3
    ny1:    int   = 3
    lx:     float = 1.0

    # ── Physics ───────────────────────────────────────────────────────────────
    rkb:    float = 3.0e-4
    rkh:    float = -5.0e-8
    rkh2:   float = 1.0e-12
    f:      float = 1600.0
    r:      float = 1.0e-5
    a:      float = 6.29
    k:      float = 1.0

    # ── Boundary conditions ───────────────────────────────────────────────────
    bc:     str   = 'dirichlet'   # 'dirichlet' or 'channel'

    # ── Time integration ──────────────────────────────────────────────────────
    scheme:     str   = 'rk4'
    dt:         float = 1.0
    tend:       float = 50000.0
    dtout:      float = 20.0

    # ── Leapfrog filter options ───────────────────────────────────────────────
    ra_alpha:   float = 0.1
    raw_filter: bool  = False
    raw_beta:   float = 0.5

    # ── I/O ───────────────────────────────────────────────────────────────────
    outfname: str  = 'qg.npz'
    save_q:   bool = False
    verbose:  bool = True

    # ── Derived (set in __post_init__) ────────────────────────────────────────
    m: int = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        ksq    = 2 ** (self.mrefin - 1)
        self.m = self.ny1 * ksq + 1
        self.n = self.nx1 * ksq + 1

    @classmethod
    def from_file(cls, fname: str) -> 'QGParams':
        """Load parameters from a .prm key=value file."""
        prm   = cls()
        _conv = {
            'mrefin':     ('mrefin',     int),
            'nx1':        ('nx1',        int),
            'ny1':        ('ny1',        int),
            'lx':         ('lx',         float),
            'rkb':        ('rkb',        float),
            'rkh':        ('rkh',        float),
            'rkh2':       ('rkh2',       float),
            'f':          ('f',          float),
            'r':          ('r',          float),
            'a':          ('a',          float),
            'k':          ('k',          float),
            'bc':         ('bc',         str),
            'scheme':     ('scheme',     str),
            'dt':         ('dt',         float),
            'tend':       ('tend',       float),
            'dtout':      ('dtout',      float),
            'ra_alpha':   ('ra_alpha',   float),
            'raw_filter': ('raw_filter', lambda x: x.strip().lower()
                           in ('1', 'yes', 'true')),
            'raw_beta':   ('raw_beta',   float),
            'outfname':   ('outfname',   str),
            'save_q':     ('save_q',     lambda x: x.strip().lower()
                           in ('1', 'yes', 'true')),
        }
        with open(fname) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip().lower()
                val = val.strip()
                if key in _conv:
                    attr, conv = _conv[key]
                    setattr(prm, attr, conv(val))
        prm.__post_init__()
        return prm

    def describe(self):
        print(f"  grid        : {self.m} x {self.n}  "
              f"(mrefin={self.mrefin}, nx1={self.nx1}, ny1={self.ny1})")
        print(f"  lx          : {self.lx}")
        print(f"  rkb/rkh/rkh2: {self.rkb} / {self.rkh} / {self.rkh2}")
        print(f"  F, r, A, k  : {self.f}, {self.r}, {self.a}, {self.k}")
        print(f"  bc          : {self.bc}")
        print(f"  scheme      : {self.scheme}")
        if 'leapfrog' in self.scheme:
            filt = 'RAW' if self.raw_filter else 'RA'
            print(f"  LF filter   : {filt}  alpha={self.ra_alpha}"
                  + (f"  beta={self.raw_beta}" if self.raw_filter else ""))
        print(f"  dt / tend   : {self.dt} / {self.tend}")
        print(f"  dtout       : {self.dtout}")
        print(f"  output      : {self.outfname}  (save_q={self.save_q})")


# =============================================================================
# QG Model
# =============================================================================

class QGModel:
    """
    1.5-layer QG model with pluggable integrators and boundary conditions.

    Parameters
    ----------
    prm : QGParams
        Model configuration.

    Examples
    --------
    Closed basin, RK4 (original behaviour):
        prm = QGParams(scheme='rk4', bc='dirichlet')
        m   = QGModel(prm)
        psi, q = m.run('zero')

    Zonal channel, Adams-Bashforth 3rd-order:
        prm = QGParams(scheme='ab3', bc='channel', tend=10000)
        m   = QGModel(prm)
        psi, q = m.run('fourier', amplitude=0.5)

    Channel with dipole IC for convergence test:
        prm = QGParams(scheme='rk4', bc='channel', tend=50, dt=0.5)
        m   = QGModel(prm)
        psi, q = m.run('dipole', separation=0.12)
    """

    def __init__(self, prm: QGParams):
        self.prm = prm
        self.m   = prm.m
        self.n   = prm.n
        self.h   = prm.lx / (prm.n - 1)
        self.t   = 0.0

        self.psi = np.zeros((prm.m, prm.n))
        self.q   = np.zeros((prm.m, prm.n))

        # Wind forcing (x-periodic sinusoid — same in both BC types)
        i_idx      = np.arange(prm.n)
        tmp        = np.sin(prm.k * 2.0 * np.pi * i_idx / prm.n)
        self.curlt = -prm.a * tmp * np.abs(tmp)

        # Boundary condition handler (Helmholtz + operators)
        if prm.verbose:
            print(f"  Building Helmholtz solver [{prm.bc}] "
                  f"({prm.m}x{prm.n})...", end=' ', flush=True)
        t0       = time.time()
        self._bc = make_bc(prm.bc, prm.m, prm.n, self.h, prm.f)
        if prm.verbose:
            print(f"done ({time.time()-t0:.2f}s)")

        # Time integrator
        self._integrator = self._build_integrator(prm)

        # Output buffers
        self._out_t   = []
        self._out_psi = []
        self._out_q   = []

    # ── integrator factory ───────────────────────────────────────────────────

    @staticmethod
    def _build_integrator(prm: QGParams):
        """Map QGParams scheme/filter settings to an integrator instance."""
        s = prm.scheme
        if s == 'leapfrog':
            # legacy name: honour raw_filter flag
            if prm.raw_filter:
                return get_integrator('leapfrog_raw',
                                      alpha=prm.ra_alpha, beta=prm.raw_beta)
            else:
                return get_integrator('leapfrog_ra', alpha=prm.ra_alpha)
        elif s in ('leapfrog_ra', 'leapfrog_raw'):
            if s == 'leapfrog_raw':
                return get_integrator(s, alpha=prm.ra_alpha, beta=prm.raw_beta)
            return get_integrator(s, alpha=prm.ra_alpha)
        elif s == 'order1':
            return get_integrator('euler')
        elif s == 'order2':
            return get_integrator('midpoint')
        else:
            return get_integrator(s)

    # ── physics ──────────────────────────────────────────────────────────────

    def _calc_psi(self, q: np.ndarray) -> np.ndarray:
        return self._bc.solve(q)

    def _rhs(self, q: np.ndarray) -> np.ndarray:
        """Compute dq/dt given q (solves for psi internally)."""
        prm  = self.prm
        h    = self.h
        bc   = self._bc
        psi  = self._calc_psi(q)

        J     = bc.arakawa(h, psi, q)
        zeta  = bc.laplacian(h, psi)
        zeta2 = bc.laplacian(h, zeta)
        zeta4 = bc.laplacian(h, zeta2)
        dpdy  = bc.dpsidy(h, psi)

        F       = np.zeros_like(q)

        if prm.bc == 'dirichlet':
            # interior only
            F[1:-1, 1:-1] = (
                - prm.r    * J[1:-1, 1:-1]
                - prm.rkb  * zeta[1:-1, 1:-1]
                + prm.rkh  * zeta2[1:-1, 1:-1]
                - prm.rkh2 * zeta4[1:-1, 1:-1]
                + self.curlt[1:-1]
                - dpdy[1:-1, 1:-1]
            )
        else:
            # channel: all columns are interior in x; only j=0,m-1 are walls
            F[1:-1, :] = (
                - prm.r    * J[1:-1, :]
                - prm.rkb  * zeta[1:-1, :]
                + prm.rkh  * zeta2[1:-1, :]
                - prm.rkh2 * zeta4[1:-1, :]
                + self.curlt
                - dpdy[1:-1, :]
            )
        return F

    # ── step ────────────────────────────────────────────────────────────────

    def _do_step(self):
        self.q   = self._integrator.step(self.q, self.t, self.prm.dt, self._rhs)
        self.psi = self._calc_psi(self.q)
        self.t  += self.prm.dt

        if not np.isfinite(self.psi).all():
            raise RuntimeError(
                f"\n\n  *** INSTABILITY (NaN/Inf) at t={self.t:.1f} ***\n"
                f"  scheme={self.prm.scheme}, dt={self.prm.dt}, bc={self.prm.bc}\n"
                f"  Suggestions:\n"
                f"    1. Reduce dt  -> try dt <= {self.prm.dt/4:.4g}\n"
                f"    2. Use leapfrog_ra with --ra-alpha 0.2\n"
                f"    3. Use leapfrog_raw instead\n"
                f"    4. Increase dissipation: --rkh2 1e-11\n"
            )

    # ── initial conditions ──────────────────────────────────────────────────

    def set_ic(self, kind: str = 'zero', **kwargs):
        """
        Set the initial condition.

        Parameters
        ----------
        kind : str
            One of: zero, fourier, vortex, dipole, rossby_wave,
                    band_noise, restart_npz, restart_nc
        **kwargs
            Passed to the IC function (see initial_conditions/ic_library.py).
        """
        self.t   = 0.0
        self._integrator.reset()

        psi, q = get_ic(
            kind,
            m=self.m, n=self.n, h=self.h, F=self.prm.f,
            bc_kind=self.prm.bc,
            lx=self.prm.lx,
            **kwargs,
        )
        self.psi = psi
        self.q   = q

    # ── output ──────────────────────────────────────────────────────────────

    def _save_record(self):
        self._out_t.append(float(self.t))
        self._out_psi.append(self.psi.copy().astype(np.float32))
        if self.prm.save_q:
            self._out_q.append(self.q.copy().astype(np.float32))

    def _write_output(self):
        prm   = self.prm
        fname = prm.outfname
        t_arr = np.array(self._out_t)
        p_arr = np.stack(self._out_psi)

        if HAS_NC and fname.endswith('.nc'):
            ds = _nc.Dataset(fname, 'w', format='NETCDF4')
            for k, v in [('dt', prm.dt), ('lx', prm.lx), ('rkb', prm.rkb),
                          ('rkh', prm.rkh), ('rkh2', prm.rkh2), ('F', prm.f),
                          ('r', prm.r), ('A', prm.a), ('k', prm.k),
                          ('scheme', prm.scheme), ('bc', prm.bc),
                          ('ra_alpha', prm.ra_alpha),
                          ('raw_filter', int(prm.raw_filter))]:
                ds.setncattr(k, v)
            ds.createDimension('record', None)
            ds.createDimension('j', prm.m)
            ds.createDimension('i', prm.n)
            ds.createVariable('t',   'f8', ('record',))[:] = t_arr
            ds.createVariable('psi', 'f4', ('record','j','i'))[:] = p_arr
            if prm.save_q:
                ds.createVariable('q', 'f4', ('record','j','i'))[:] = \
                    np.stack(self._out_q)
            ds.close()
        else:
            if not fname.endswith('.npz'):
                fname = os.path.splitext(fname)[0] + '.npz'
            kw = dict(t=t_arr, psi=p_arr,
                      lx=prm.lx, dt=prm.dt, rkb=prm.rkb,
                      rkh=prm.rkh, rkh2=prm.rkh2, F=prm.f,
                      r=prm.r, A=prm.a, k=prm.k,
                      scheme=prm.scheme, bc=prm.bc,
                      ra_alpha=prm.ra_alpha,
                      raw_filter=int(prm.raw_filter))
            if prm.save_q:
                kw['q'] = np.stack(self._out_q)
            np.savez_compressed(fname, **kw)

        if prm.verbose:
            print(f"\n  Output saved: {fname}  ({len(t_arr)} records)")

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self, ic: str = 'zero', **ic_kwargs):
        """
        Set IC and integrate to tend, saving output every dtout.

        Parameters
        ----------
        ic       : initial condition name (see list_ics())
        **ic_kwargs : passed to the IC function

        Returns
        -------
        psi, q : final state arrays
        """
        prm = self.prm
        self.set_ic(ic, **ic_kwargs)
        self._out_t   = []
        self._out_psi = []
        self._out_q   = []
        self._save_record()

        nstep = int((prm.tend - self.t) / prm.dt)
        dnout = max(1, int(prm.dtout / prm.dt))

        if prm.verbose:
            print(f"  nstep = {nstep}, nrecord ~= {nstep // dnout}")
            print(f"  integrating [{prm.scheme} | bc={prm.bc}]:",
                  end='', flush=True)

        t0 = time.time()
        for step in range(1, nstep + 1):
            self._do_step()
            if step % dnout == 0:
                self._save_record()
                if prm.verbose:
                    print(':', end='', flush=True)

        elapsed = time.time() - t0
        if prm.verbose:
            print(f"\n  done in {elapsed:.1f}s  ({nstep/elapsed:.0f} steps/s)")

        self._write_output()
        return self.psi, self.q

    # ── convenience: diagnostics ─────────────────────────────────────────────

    def kinetic_energy(self) -> float:
        """Domain-averaged kinetic energy: KE = 0.5 * <|nabla psi|^2>."""
        h  = self.h
        # centred differences for gradients
        dpsi_dx = np.zeros_like(self.psi)
        dpsi_dy = np.zeros_like(self.psi)
        dpsi_dx[1:-1, 1:-1] = (self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2*h)
        dpsi_dy[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2*h)
        return 0.5 * float(np.mean(dpsi_dx**2 + dpsi_dy**2))

    def enstrophy(self) -> float:
        """Domain-averaged enstrophy: Z = 0.5 * <q^2>."""
        return 0.5 * float(np.mean(self.q**2))


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == '__main__':
    import argparse

    _SCHEMES = list_integrators() + ['leapfrog', 'order1', 'order2']

    parser = argparse.ArgumentParser(
        description='QG-Python v2: 1.5-layer QG model with pluggable '
                    'integrators and boundary conditions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available schemes : {sorted(list_integrators())}
  Legacy aliases  : leapfrog (=leapfrog_ra|leapfrog_raw), order1 (=euler), order2 (=midpoint)

Available ICs     : {list_ics()}

Boundary conditions:
  dirichlet  — closed basin, psi=0 on all walls (default, QG-C compatible)
  channel    — zonal channel, periodic in x, psi=0 at north/south

Examples:
  # Closed basin, RK4 reference run (backward compatible with v1)
  python qg_model_v2.py --scheme rk4 --bc dirichlet --tend 110000 \\
      --ic restart --restart-file qg_spin.npz --out qg_rk4.npz

  # Zonal channel, Adams-Bashforth 3rd-order
  python qg_model_v2.py --scheme ab3 --bc channel --tend 10000 \\
      --ic fourier --out qg_ab3_channel.npz

  # Convergence test: dipole IC, channel, multiple schemes
  python qg_model_v2.py --scheme ssprk3 --bc channel --tend 20 \\
      --ic dipole --dt 0.1 --out qg_ssprk3_dipole.npz
""")

    # Time
    parser.add_argument('prmfile', nargs='?', help='parameter file (.prm)')
    parser.add_argument('--scheme', default=None)
    parser.add_argument('--bc',     default=None, choices=['dirichlet','channel'])
    parser.add_argument('--tend',   type=float)
    parser.add_argument('--dt',     type=float)
    parser.add_argument('--dtout',  type=float)
    parser.add_argument('--out',    metavar='FILE')
    parser.add_argument('--save-q', action='store_true')
    parser.add_argument('--quiet',  action='store_true')

    # Grid
    parser.add_argument('--mrefin', type=int)
    parser.add_argument('--nx1',    type=int)
    parser.add_argument('--ny1',    type=int)

    # Physics
    parser.add_argument('--rkb',  type=float)
    parser.add_argument('--rkh',  type=float)
    parser.add_argument('--rkh2', type=float)
    parser.add_argument('--a',    type=float)
    parser.add_argument('--f',    type=float)
    parser.add_argument('--r',    type=float)

    # Leapfrog
    parser.add_argument('--ra-alpha', type=float, default=None)
    parser.add_argument('--raw',      action='store_true')
    parser.add_argument('--raw-beta', type=float, default=None)

    # IC
    parser.add_argument('--ic', default='zero', choices=list_ics())
    parser.add_argument('--amplitude',      type=float, default=1.0)
    parser.add_argument('--kmax',           type=int,   default=5)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--kx',             type=int,   default=2)
    parser.add_argument('--ky',             type=int,   default=1)
    parser.add_argument('--separation',     type=float, default=0.12)
    parser.add_argument('--restart-file',   metavar='FILE')
    parser.add_argument('--restart-record', type=int, default=-1)

    args = parser.parse_args()

    prm = QGParams.from_file(args.prmfile) if args.prmfile else QGParams()

    if args.scheme:   prm.scheme   = args.scheme
    if args.bc:       prm.bc       = args.bc
    if args.tend:     prm.tend     = args.tend
    if args.dt:       prm.dt       = args.dt
    if args.dtout:    prm.dtout    = args.dtout
    if args.out:      prm.outfname = args.out
    if args.save_q:   prm.save_q   = True
    if args.quiet:    prm.verbose  = False
    if args.mrefin or args.nx1 or args.ny1:
        if args.mrefin: prm.mrefin = args.mrefin
        if args.nx1:    prm.nx1    = args.nx1
        if args.ny1:    prm.ny1    = args.ny1
        prm.__post_init__()
    if args.rkb  is not None: prm.rkb  = args.rkb
    if args.rkh  is not None: prm.rkh  = args.rkh
    if args.rkh2 is not None: prm.rkh2 = args.rkh2
    if args.a    is not None: prm.a    = args.a
    if args.f    is not None: prm.f    = args.f
    if args.r    is not None: prm.r    = args.r
    if args.ra_alpha is not None: prm.ra_alpha   = args.ra_alpha
    if args.raw:                  prm.raw_filter = True
    if args.raw_beta is not None: prm.raw_beta   = args.raw_beta

    print("  QG-Python v2")
    prm.describe()
    print(f"  netCDF4 available : {HAS_NC}")
    print(f"  initial condition : {args.ic}")

    ic_kwargs: dict = {}
    if args.ic == 'fourier':
        ic_kwargs = dict(amplitude=args.amplitude, kmax=args.kmax, seed=args.seed)
    elif args.ic == 'vortex':
        ic_kwargs = dict()   # use defaults; extend CLI if needed
    elif args.ic == 'dipole':
        ic_kwargs = dict(separation=args.separation)
    elif args.ic == 'rossby_wave':
        ic_kwargs = dict(kx=args.kx, ky=args.ky, amplitude=args.amplitude)
    elif args.ic == 'band_noise':
        ic_kwargs = dict(amplitude=args.amplitude, seed=args.seed)
    elif args.ic in ('restart_npz', 'restart_nc'):
        if not args.restart_file:
            parser.error(f"--ic {args.ic} requires --restart-file FILE")
        ic_kwargs = dict(fname=args.restart_file, record=args.restart_record)

    model = QGModel(prm)
    model.run(args.ic, **ic_kwargs)
