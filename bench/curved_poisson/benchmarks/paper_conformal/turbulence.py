"""Seeded 2-D synthetic fields with a prescribed shell-energy spectrum.

The scalar is an analytic Poisson MMS; the companion velocity is solenoidal.
This creates fields only, without running a PDE solver or evolving turbulence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/bspf-turbulence-mpl')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class SyntheticField:
    def __init__(self, coefficients, length=2., origin=-.25):
        self.coefficients = coefficients
        self.length, self.origin = length, origin
        self.n = coefficients.shape[0]
        assert coefficients.shape == (self.n, self.n)
        self.modes = np.fft.fftfreq(self.n)*self.n
        self.k = 2*np.pi*self.modes/length
        self.radius = np.hypot(self.modes[:, None], self.modes[None, :])
        self.shell = np.floor(self.radius+.5).astype(int)
        self.k2 = self.k[:, None]**2+self.k[None, :]**2
        active = np.flatnonzero(np.any(abs(coefficients)>0, axis=0)
                                | np.any(abs(coefficients)>0, axis=1))
        self.active_k = self.k[active]
        self.active_c = coefficients[np.ix_(active, active)]

    def coefficient_fields(self):
        c, k = self.coefficients, self.k
        inv_k = np.zeros_like(self.k2)
        np.divide(1., np.sqrt(self.k2), out=inv_k, where=self.k2>0)
        psi = c*inv_k
        return dict(scalar=c, scalar_x=1j*k[:, None]*c,
                    scalar_y=1j*k[None, :]*c, forcing=self.k2*c,
                    velocity_x=1j*k[None, :]*psi,
                    velocity_y=-1j*k[:, None]*psi,
                    vorticity=self.k2*psi)

    def grid_fields(self):
        result = {}
        for name, coefficients in self.coefficient_fields().items():
            values = np.fft.ifft2(coefficients*self.n**2)
            assert np.max(abs(values.imag)) < 1e-10*max(1., np.max(abs(values.real)))
            result[name] = values.real
        return result

    def truth(self, x, y):
        """Return scalar, physical x/y derivatives and f=-Delta scalar.

        Evaluates the finite Fourier series at arbitrary physical points, with
        no interpolation or map assumption. Compatible with the Poisson MMS
        helper's return convention. Dirichlet data are the first return value.
        """
        x, y = np.broadcast_arrays(x, y)
        shape = x.shape
        xx, yy = x.ravel()-self.origin, y.ravel()-self.origin
        k, c = self.active_k, self.active_c
        cx = 1j*k[:, None]*c
        cf = (k[:, None]**2+k[None, :]**2)*c
        output = np.empty((4, len(xx)))
        for start in range(0, len(xx), 1024):
            sl = slice(start, start+1024)
            ex, ey = np.exp(1j*xx[sl, None]*k), np.exp(1j*yy[sl, None]*k)
            product = ex@c
            output[0, sl] = np.sum(product*ey, axis=1).real
            output[1, sl] = np.sum((ex@cx)*ey, axis=1).real
            output[2, sl] = np.sum(product*(1j*k*ey), axis=1).real
            output[3, sl] = np.sum((ex@cf)*ey, axis=1).real
        return tuple(v.reshape(shape) for v in output)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['coefficients'], float(data['length']), float(data['origin']))


def create(seed=20261115, n=512, shell_min=4, shell_max=64, rms=1.):
    if n % 2 or not 1 <= shell_min < shell_max < n//2-1 or rms <= 0:
        raise ValueError('Use an even grid, resolved positive shell range and positive RMS')
    # Real white noise supplies conjugate symmetry, random phases and magnitudes.
    rng = np.random.default_rng(seed)
    c = np.fft.fft2(rng.standard_normal((n, n)))/n**2
    field = SyntheticField(c)
    shells = np.arange(shell_min, shell_max+1)
    energy = shells.astype(float)**(-5/3)
    energy *= .5*rms**2/energy.sum()
    current = np.bincount(field.shell.ravel(), weights=(.5*abs(c)**2).ravel())
    factors = np.zeros_like(current)
    factors[shells] = np.sqrt(energy/current[shells])
    c *= factors[field.shell]
    return SyntheticField(c), shells, energy


def shell_energy(field, *values):
    # Measure from physical fields, not from the prescribed coefficients.
    power = sum(.5*abs(np.fft.fft2(v)/field.n**2)**2 for v in values)
    return np.bincount(field.shell.ravel(), weights=power.ravel())


def verify(field, fields, shells, target):
    scalar = shell_energy(field, fields['scalar'])
    velocity = shell_energy(field, fields['velocity_x'], fields['velocity_y'])
    slopes = [float(np.polyfit(np.log(shells), np.log(e[shells]), 1)[0])
              for e in (scalar, velocity)]
    errors = [float(np.max(abs(e[shells]/target-1))) for e in (scalar, velocity)]
    assert max(errors) < 2e-12 and max(abs(v+5/3) for v in slopes) < 2e-12
    q = field.coefficient_fields()
    div = 1j*field.k[:, None]*np.fft.fft2(fields['velocity_x'])/field.n**2
    div += 1j*field.k[None, :]*np.fft.fft2(fields['velocity_y'])/field.n**2
    divergence_rms = float(np.linalg.norm(div))
    derivative_scale = np.linalg.norm(np.sqrt(field.k2)*q['scalar'])
    assert divergence_rms/derivative_scale < 2e-13
    measured_c = np.fft.fft2(fields['scalar'])/field.n**2
    measured_f = np.fft.fft2(fields['forcing'])/field.n**2
    forcing_residual = float(np.linalg.norm(measured_f-field.k2*measured_c)
                             / np.linalg.norm(measured_f))
    assert forcing_residual < 2e-12
    physical_energy = .5*np.mean(fields['scalar']**2)
    assert abs(physical_energy-target.sum()) < 2e-13
    # Check the arbitrary-point evaluator against FFT grid samples and a direct
    # mode sum off the grid; these checks do not call a Poisson solver.
    rng = np.random.default_rng(9981)
    ix, iy = rng.integers(0, field.n, (2, 24))
    xy = field.origin+np.array([ix, iy])*field.length/field.n
    evaluated = field.truth(*xy)
    grid_errors = {}
    for name, v in zip(('scalar', 'scalar_x', 'scalar_y', 'forcing'), evaluated):
        relative = float(np.max(abs(v-fields[name][ix, iy]))/max(1., np.max(abs(fields[name]))))
        assert relative < 2e-12, (name, relative)
        grid_errors[name] = relative
    xy = field.origin+field.length*rng.random((2, 12))
    evaluated = field.truth(*xy)
    mx, my = np.nonzero(abs(field.coefficients)>0)
    phase = np.exp(1j*((xy[0, :, None]-field.origin)*field.k[mx]
                     +(xy[1, :, None]-field.origin)*field.k[my]))
    direct_errors = {}
    for name, v in zip(('scalar', 'scalar_x', 'scalar_y', 'forcing'), evaluated):
        direct = (phase@q[name][mx, my]).real
        relative = float(np.max(abs(v-direct))/max(1., np.max(abs(direct))))
        assert relative < 2e-12, (name, relative)
        direct_errors[name] = relative
    return dict(scalar_slope=slopes[0], kinetic_energy_slope=slopes[1],
                scalar_shell_max_relative_error=errors[0],
                kinetic_shell_max_relative_error=errors[1],
                scalar_rms=float(np.sqrt(np.mean(fields['scalar']**2))),
                velocity_rms=float(np.sqrt(np.mean(fields['velocity_x']**2+fields['velocity_y']**2))),
                scalar_mean=float(np.mean(fields['scalar'])),
                divergence_rms=divergence_rms,
                divergence_relative_to_gradient=float(divergence_rms/derivative_scale),
                poisson_forcing_relative_residual=forcing_residual,
                grid_evaluator_relative_errors=grid_errors,
                direct_mode_sum_relative_errors=direct_errors,
                pde_solve_calls=0), scalar, velocity


def plot(field, fields, shells, target, scalar_energy, velocity_energy, out, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coords = field.origin+field.length*np.arange(field.n)/field.n
    x, y = np.meshgrid(coords, coords, indexing='ij')
    w = (-1+np.sqrt(1+.6*(x+1j*y)))/.3
    mask = (w.real<0)|(w.real>1)|(w.imag<0)|(w.imag>1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    q = np.linspace(0, 1, 501)
    boundary = []
    for s, t in ((q, 0*q), (0*q+1, q), (q, 0*q+1), (0*q, q)):
        z = s+1j*t
        z = z+.15*z*z
        boundary.append((z.real, z.imag))
    def outline(ax):
        for bx, by in boundary:
            ax.plot(bx, by, color='#222222', lw=.8)
        ax.set(xlim=(-.17, 1.17), ylim=(-.02, 1.32), aspect='equal', xlabel='$x$', ylabel='$y$')
    scalar = np.ma.array(fields['scalar'], mask=mask)
    limit = float(np.max(abs(scalar)))
    im = axes[0, 0].pcolormesh(x, y, scalar, cmap='RdBu_r', vmin=-limit, vmax=limit,
                              shading='auto', rasterized=True)
    outline(axes[0, 0])
    axes[0, 0].set_title('Scalar Poisson MMS on the curved domain')
    fig.colorbar(im, ax=axes[0, 0], shrink=.85, label=r'$\phi(x,y)$')
    speed = np.ma.array(np.hypot(fields['velocity_x'], fields['velocity_y']), mask=mask)
    im = axes[0, 1].pcolormesh(x, y, speed, cmap='viridis', shading='auto', rasterized=True)
    vx = np.ma.array(fields['velocity_x'].T, mask=mask.T)
    vy = np.ma.array(fields['velocity_y'].T, mask=mask.T)
    axes[0, 1].streamplot(coords, coords, vx, vy, density=1.7, color='white', linewidth=.45, arrowsize=.55)
    outline(axes[0, 1])
    axes[0, 1].set_title('Divergence-free velocity: speed and streamlines')
    fig.colorbar(im, ax=axes[0, 1], shrink=.85, label=r'$|\mathbf{v}(x,y)|$')
    k = 2*np.pi*shells/field.length
    dk = 2*np.pi/field.length
    ax = axes[1, 0]
    ax.loglog(k, target/dk, color='black', lw=2, label=r'Target $E(k)\propto k^{-5/3}$')
    ax.loglog(k, scalar_energy[shells]/dk, 'o', ms=3.5, mfc='none', color='#087fa1', label='Scalar: measured FFT')
    ax.loglog(k, velocity_energy[shells]/dk, '.', ms=3, color='#853bac', label='Velocity: measured FFT')
    ax.set(title='Shell-integrated spectrum on the parent periodic box', xlabel='Angular wave number k', ylabel='Energy density E(k)')
    ax.grid(True, which='both', alpha=.2)
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    normalization = target[0]/dk*k[0]**(5/3)
    for e, label, color in ((scalar_energy, 'Scalar', '#087fa1'), (velocity_energy, 'Velocity', '#853bac')):
        ax.semilogx(k, (e[shells]/dk)*k**(5/3)/normalization, color=color, label=label)
    ax.axhline(1, color='black', linestyle='--', lw=.8)
    ax.set(ylim=(.95, 1.05), title='Compensated spectrum: flat across the prescribed band',
           xlabel='Angular wave number k', ylabel=r'$k^{5/3}E(k)$ / target constant')
    ax.grid(True, which='both', alpha=.2)
    ax.legend()
    fig.suptitle(f'Synthetic turbulence spectrum: random phases and magnitudes · seed {seed}', fontsize=15)
    fig.supxlabel('512 × 512 periodic parent box; RMS = 1. Spectrum imposed on shells 4–64. '
                  'Curved-domain restriction is not used for the FFT spectrum.', fontsize=9)
    fig.savefig(out/'synthetic_turbulence.png', dpi=180)
    fig.savefig(out/'synthetic_turbulence.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or ROOT/f'evidence/paper_conformal/turbulence_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists():
        raise RuntimeError('Choose a new output directory to preserve the existing realization')
    field, shells, target = create(args.seed)
    fields = field.grid_fields()
    validation, scalar, velocity = verify(field, fields, shells, target)
    np.savez_compressed(out/'field.npz', coefficients=field.coefficients,
                        length=field.length, origin=field.origin,
                        x=field.origin+field.length*np.arange(field.n)/field.n,
                        **fields)
    np.savez_compressed(out/'spectrum.npz', shell=shells, k=2*np.pi*shells/field.length,
                        target_shell_energy=target, scalar_shell_energy=scalar[shells],
                        kinetic_shell_energy=velocity[shells])
    metadata = dict(seed=args.seed, generator='NumPy PCG64 real Gaussian noise, Fourier shell normalization',
                    numpy_version=np.__version__, dimensions=2, grid=[field.n, field.n],
                    periodic_box=[[-.25, 1.75], [-.25, 1.75]],
                    shell_definition='s=floor(sqrt(mx^2+my^2)+0.5); both conjugate modes counted',
                    active_shells=[4, 64], target_spectrum='E_s = C*s^(-5/3); zero outside band',
                    energy_definition='sum E_s = 0.5*mean(phi^2) = 0.5*mean(vx^2+vy^2)',
                    active_fourier_modes=int(np.count_nonzero(field.coefficients)),
                    independent_conjugate_pairs=int(np.count_nonzero(field.coefficients)//2),
                    curved_map='z=w+0.15*w^2, w=s+i*t, (s,t) in [0,1]^2',
                    poisson_problem='-Delta phi=f, g=phi on curved boundary; exact Fourier derivatives, no damping envelope',
                    velocity_definition='psi_hat=phi_hat/|k|; vx=dy psi, vy=-dx psi',
                    scope='Synthetic instantaneous 2-D fields with imposed spectrum; not a Navier-Stokes simulation or evidence of a dynamical cascade. The prescribed spectrum applies to the full periodic parent box, not its curved subdomain.',
                    validation=validation,
                    reference='https://ntrs.nasa.gov/citations/19990004129')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    plot(field, fields, shells, target, scalar, velocity, out, args.seed)
    (out/'turbulence.py').write_bytes(Path(__file__).read_bytes())
    (out/'SHA256SUMS.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}, indent=2)+'\n')
    print(json.dumps(validation, indent=2))
    print(out/'synthetic_turbulence.png')


if __name__ == '__main__':
    main()
