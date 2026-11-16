"""Nonperiodic, divergence-free 3D plane-wave MMS on a curved hexahedron.

Wavevectors are continuous random vectors, not a reciprocal lattice. All
components are physical Cartesian components; evaluate at physical coordinates.
The NumPy path is for validation; production generation uses CuPy on CUDA.
"""
from pathlib import Path
import numpy as np


class CurvedBox:
    """An exact Q2 map with six distinct curved physical boundary faces.

    X_i = s_i + a_i (1-2 s_i) prod_{j != i} 4 s_j (1-s_j).
    max|a_i|<0.1 makes ||D(X-s)||_infinity<1, ensuring an injective,
    orientation-preserving map on [0,1]^3. No faces are identified.
    """
    def __init__(self, amplitudes=(.08, -.06, .07)):
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        if self.amplitudes.shape != (3,) or not np.isfinite(self.amplitudes).all() or np.max(abs(self.amplitudes)) >= .1:
            raise ValueError('Three finite amplitudes with absolute value < 0.1 required')

    def map(self, reference, xp=np):
        s = xp.asarray(reference, dtype=xp.float64)
        if s.shape[-1] != 3:
            raise ValueError('Expected reference points (..., 3)')
        phi, dphi = 4*s*(1-s), 4*(1-2*s)
        x = s.copy(); jac = xp.zeros(s.shape[:-1]+(3, 3), dtype=xp.float64)
        for i in range(3):
            j, k = (i+1) % 3, (i+2) % 3
            a = float(self.amplitudes[i])
            x[..., i] += a*(1-2*s[..., i])*phi[..., j]*phi[..., k]
            jac[..., i, i] = 1-2*a*phi[..., j]*phi[..., k]
            jac[..., i, j] = a*(1-2*s[..., i])*dphi[..., j]*phi[..., k]
            jac[..., i, k] = a*(1-2*s[..., i])*phi[..., j]*dphi[..., k]
        det = xp.sum(jac[..., 0, :]*xp.cross(jac[..., 1, :], jac[..., 2, :]), axis=-1)
        return x, jac, det

    def face(self, axis, side, uv, xp=np):
        if axis not in (0, 1, 2) or side not in (0, 1):
            raise ValueError('Invalid face')
        uv = xp.asarray(uv)
        s = xp.empty(uv.shape[:-1]+(3,), dtype=xp.float64)
        j, k = (axis+1) % 3, (axis+2) % 3
        s[..., axis] = side; s[..., j] = uv[..., 0]; s[..., k] = uv[..., 1]
        x, jac, _ = self.map(s, xp)
        area_vector = (2*side-1)*xp.cross(jac[..., :, j], jac[..., :, k])
        return x, area_vector


class PlaneWaveMMS:
    """u=sum(a cos(theta)+b sin(theta))+u_offset, theta=k.(x-center).

    k.a=k.b=0. Pressure uses independent scalar cos/sin amplitudes.
    Equation convention: div u=0, f_NS=(u.grad)u+grad(p)-nu*Delta(u).
    No time dependence is assumed. Fourier shell energy is a prescribed modal
    quantity, not the FFT spectrum of samples on a nonperiodic curved domain.
    """
    def __init__(self, k, a, b, pc, ps, *, center=(.5, .5, .5),
                 u_offset=(0., 0., 0.), p_offset=0., viscosity=.01, xp=np):
        self.xp = xp
        self.k, self.a, self.b = (xp.asarray(v, dtype=xp.float64) for v in (k, a, b))
        self.pc, self.ps = (xp.asarray(v, dtype=xp.float64) for v in (pc, ps))
        m = len(self.k)
        if self.k.shape != (m, 3) or self.a.shape != (m, 3) or self.b.shape != (m, 3) or self.pc.shape != (m,) or self.ps.shape != (m,):
            raise ValueError('Invalid plane-wave coefficient shapes')
        self.center = xp.asarray(center, dtype=xp.float64)
        self.u_offset = xp.asarray(u_offset, dtype=xp.float64)
        self.p_offset = float(p_offset); self.viscosity = float(viscosity)
        if not np.isfinite(self.viscosity) or self.viscosity < 0:
            raise ValueError('Nonnegative finite viscosity required')
        self._features = None

    def coefficients(self):
        xp = self.xp
        if self._features is None:
            k2 = xp.sum(self.k*self.k, axis=1)
            # Columns: u(3), p(1), du_i/dx_j(9), -lap u(3), grad p(3), -lap p(1).
            cosine = xp.concatenate((self.a, self.pc[:, None],
                       (self.b[:, :, None]*self.k[:, None, :]).reshape(-1, 9),
                       k2[:, None]*self.a, self.ps[:, None]*self.k, (k2*self.pc)[:, None]), axis=1)
            sine = xp.concatenate((self.b, self.ps[:, None],
                       (-self.a[:, :, None]*self.k[:, None, :]).reshape(-1, 9),
                       k2[:, None]*self.b, -self.pc[:, None]*self.k, (k2*self.ps)[:, None]), axis=1)
            self._features = cosine, sine
        return self._features

    def evaluate(self, points, *, derivatives=True, chunk=2048):
        xp = self.xp
        points = xp.asarray(points, dtype=xp.float64)
        if points.shape[-1] != 3 or chunk < 1:
            raise ValueError('Expected physical points (...,3) and a positive chunk size')
        shape = points.shape[:-1]; flat = points.reshape(-1, 3)
        cosine, sine = self.coefficients()
        nc = 20 if derivatives else 4
        values = xp.empty((len(flat), nc), dtype=xp.float64)
        for start in range(0, len(flat), chunk):
            sl = slice(start, start+chunk)
            phase = (flat[sl]-self.center)@self.k.T
            values[sl] = xp.cos(phase)@cosine[:, :nc]+xp.sin(phase)@sine[:, :nc]
        values[:, :3] += self.u_offset; values[:, 3] += self.p_offset
        result = dict(velocity=values[:, :3].reshape(shape+(3,)), pressure=values[:, 3].reshape(shape))
        if derivatives:
            gradient = values[:, 4:13].reshape(-1, 3, 3)
            poisson = values[:, 13:16]; gradp = values[:, 16:19]
            stokes = self.viscosity*poisson+gradp
            advection = xp.einsum('nj,nij->ni', values[:, :3], gradient)
            vorticity = xp.stack((gradient[:, 2, 1]-gradient[:, 1, 2],
                                 gradient[:, 0, 2]-gradient[:, 2, 0],
                                 gradient[:, 1, 0]-gradient[:, 0, 1]), axis=1)
            result.update(velocity_gradient=gradient.reshape(shape+(3, 3)),
                          vorticity=vorticity.reshape(shape+(3,)),
                          vector_poisson_forcing=poisson.reshape(shape+(3,)),
                          pressure_gradient=gradp.reshape(shape+(3,)),
                          pressure_poisson_forcing=values[:, 19].reshape(shape),
                          stokes_forcing=stokes.reshape(shape+(3,)),
                          navier_stokes_forcing=(stokes+advection).reshape(shape+(3,)))
        return result

    def save(self, path, **metadata):
        xp = self.xp
        host = lambda a: np.asarray(a) if xp is np else xp.asnumpy(a)
        np.savez(path, k=host(self.k), a=host(self.a), b=host(self.b),
                 pc=host(self.pc), ps=host(self.ps), center=host(self.center),
                 u_offset=host(self.u_offset), p_offset=self.p_offset,
                 viscosity=self.viscosity, **metadata)

    @classmethod
    def load(cls, path, xp=np):
        with np.load(Path(path)) as data:
            return cls(**{k: data[k] for k in ('k','a','b','pc','ps','center','u_offset','p_offset','viscosity')}, xp=xp)


def random_modes(cp, *, seed=20261116, shell_min=4, shell_max=32, modes_per_shell=64,
                 k_unit=np.pi, viscosity=.01):
    """Generate continuous wavevectors and transverse polarizations on the GPU."""
    if not (1 <= shell_min < shell_max and modes_per_shell > 0 and k_unit > 0):
        raise ValueError('Invalid shell range, count, or wavenumber unit')
    rng = cp.random.default_rng(seed)
    shells = cp.repeat(cp.arange(shell_min, shell_max+1), modes_per_shell)
    m = len(shells)
    direction = rng.standard_normal((m, 3), dtype=cp.float64)
    direction /= cp.linalg.norm(direction, axis=1)[:, None]
    radius = k_unit*(shells+rng.random(m, dtype=cp.float64)-.5)
    k = radius[:, None]*direction
    polarizations = rng.standard_normal((m, 2, 3), dtype=cp.float64)
    for _ in range(2):
        polarizations -= cp.sum(polarizations*direction[:, None, :], axis=2)[:, :, None]*direction[:, None, :]
    target = cp.arange(shell_min, shell_max+1, dtype=cp.float64)**(-5/3)
    target *= .5/cp.sum(target)
    energy = .25*cp.sum(polarizations**2, axis=(1, 2))
    current = cp.bincount(shells, weights=energy, minlength=shell_max+1)
    factors = cp.zeros(shell_max+1, dtype=cp.float64)
    factors[shell_min:] = cp.sqrt(target/current[shell_min:])
    polarizations *= factors[shells, None, None]
    pressure = rng.standard_normal((m, 2), dtype=cp.float64)
    ep = .25*cp.sum(pressure**2, axis=1)
    pe = cp.bincount(shells, weights=ep, minlength=shell_max+1)
    pressure *= (.1*cp.sqrt(target/pe[shell_min:]))[(shells-shell_min), None]
    return PlaneWaveMMS(k, polarizations[:, 0], polarizations[:, 1], pressure[:, 0], pressure[:, 1],
                        viscosity=viscosity, xp=cp), shells, target
