# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental moving tanh enrichment of the FFT BSPF Navier--Stokes space.

No artificial viscosity. Basis updates use conservative L2 projection. The
sensor sees the computed solution only; it has no prescribed front trajectory.
The basis is frozen during each RK4 step. This is an experiment, not an
entropy-stable or positivity-preserving shock-capturing method.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse.linalg import LinearOperator, eigsh

from shock_tube_1d import BSPFNavierStokes, NonphysicalState, fluxes, initial_state, primitives


class EnrichedSpace:
    def __init__(self, base, centers, widths, orthogonalize=False, retained=None):
        self.base = base
        self.centers, self.widths = np.asarray(centers), np.asarray(widths)
        self.n, self.k = base.n, len(centers)
        # Subtract the interior BSPF projection. Both the raw functions and
        # interior basis vanish at the walls; momentum remains exactly zero.
        raw, raw_x = self.raw(base.q)
        self.correction = np.zeros((self.n, self.k))
        fresh = np.ones(self.k, dtype=bool)
        seed = getattr(self, 'correction_seed', None)
        if seed is not None:
            indices, old_correction = seed
            self.correction[:, indices] = old_correction
            fresh[indices] = False
        if np.any(fresh):
            self.correction[1:-1, fresh] = base.solve_mass(base.project(raw[:, fresh])[1:-1], interior=True)
        h = raw-base.evaluate(self.correction)
        self.scale = np.sqrt(np.sum(base.weights[:, None]*h*h, axis=0))
        if np.any(self.scale < 1e-9):
            raise ValueError('enrichment is numerically redundant')
        self.transform = np.diag(1/self.scale)
        if orthogonalize:
            directions = np.eye(len(centers))
            if retained is not None:
                # Keep the old space exactly. Orthogonalize only newly added
                # raw functions, rather than truncating a rotated old space.
                directions = directions[:, np.all(retained == 0, axis=1)]
                old = h@retained
                gram_old = old.T@(base.weights[:, None]*old)
                coupling = old.T@(base.weights[:, None]*(h@directions))
                directions = directions-retained@np.linalg.solve(gram_old, coupling)
            residual = h@directions
            gram = residual.T@(base.weights[:, None]*residual)
            eigenvalues, eigenvectors = np.linalg.eigh((gram+gram.T)/2)
            cutoff = max(1e-15, 1e-12*eigenvalues[-1]) if len(eigenvalues) else 1e-15
            keep = eigenvalues > cutoff
            added = directions@(eigenvectors[:, keep]/np.sqrt(eigenvalues[keep]))
            self.transform = added if retained is None else np.column_stack((retained, added))
            self.k = self.transform.shape[1]
        self.H = h@self.transform
        self.Hx = (raw_x-base.evaluate(self.correction, 1))@self.transform
        self.cross = base.project(self.H)
        self.small = self.H.T@(base.weights[:, None]*self.H)
        self.factors = {}
        for interior in (False, True):
            cross = self.cross[1:-1] if interior else self.cross
            extension = base.solve_mass(cross, interior)
            schur = self.small-cross.T@extension
            self.factors[interior] = (extension, cho_factor((schur+schur.T)/2))
        self.integral = np.r_[base.integral, base.weights@self.H]
        # Generalized eigenvalue estimate includes all enrichment modes.
        size = self.n+self.k
        def op(action):
            return LinearOperator((size, size), matvec=lambda x: action(x[:, None]).ravel(), dtype=float)
        seed = self.radius_start(size) if hasattr(self, 'radius_start') else np.random.default_rng(0).normal(size=size)
        eigenvalues, eigenvectors = eigsh(op(lambda x: self.project(self.evaluate(x, 1), 1)),
            M=op(lambda x: self.project(self.evaluate(x))), Minv=op(self.mass),
            k=1, which='LA', tol=1e-8, v0=seed, return_eigenvectors=True)
        self.radius = float(eigenvalues[0])
        self.radius_vector = eigenvectors[:, 0]

    def raw(self, x):
        x = np.asarray(x)[:, None]
        t = np.tanh((x-self.centers)/self.widths)
        left = np.tanh(-self.centers/self.widths)
        right = np.tanh((1-self.centers)/self.widths)
        return t-left-x*(right-left), (1-t*t)/self.widths-(right-left)

    def evaluate(self, state, derivative=0):
        return self.base.evaluate(state[:self.n], derivative)+(self.Hx if derivative else self.H)@state[self.n:]

    def at(self, x, state):
        evaluation = self.base.line.evaluate(x).matrix
        h = (self.raw(x)[0]-evaluation@self.correction)@self.transform
        return evaluation@state[:self.n]+h@state[self.n:]

    def project(self, values, derivative=0):
        h = self.Hx if derivative else self.H
        return np.vstack((self.base.project(values, derivative), h.T@(self.base.weights[:, None]*values)))

    def mass(self, rhs, interior=False):
        nbase = self.n-2 if interior else self.n
        cross = self.cross[1:-1] if interior else self.cross
        extension, factor = self.factors[interior]
        z = self.base.solve_mass(rhs[:nbase], interior)
        a = cho_solve(factor, rhs[nbase:]-cross.T@z)
        return np.vstack((z-extension@a, a))

    def fit(self, samples):
        load = self.project(samples)
        state = self.mass(load)
        free = np.r_[np.arange(1, self.n-1), np.arange(self.n, self.n+self.k)]
        state[free, 1:2] = self.mass(load[free, 1:2], interior=True)
        state[[0, self.n-1], 1] = 0
        return state

    def evaluate_with_gradient(self, state):
        return self.evaluate(state), self.evaluate(state, 1)

    def rhs(self, state):
        b = self.base
        values, gradient = self.evaluate_with_gradient(state)
        convective, viscous = fluxes(values, gradient, b.mu, b.kappa, b.gamma)
        load = self.project(convective-viscous, 1)
        rate = np.zeros_like(load)
        rate[:, (0, 2)] = self.mass(load[:, (0, 2)])
        free = np.r_[np.arange(1, self.n-1), np.arange(self.n, self.n+self.k)]
        rate[free, 1:2] = self.mass(load[free, 1:2], interior=True)
        rate[[0, self.n-1], 1] = 0
        return rate


def detect(x, samples, h, count=2):
    """Locate separated density/pressure-gradient peaks, estimate tanh widths."""
    rho, _, pressure, _ = primitives(samples)
    fields = [gaussian_filter1d(v, max(.5, .25*h/(x[1]-x[0]))) for v in (rho, pressure)]
    gradients = [np.abs(np.gradient(v, x)) for v in fields]
    sensor = sum(g/max(np.ptp(v), 1e-10) for g, v in zip(gradients, fields))
    sensor[(x < 4*h) | (x > 1-4*h)] = 0
    peaks, _ = find_peaks(sensor, height=.05*sensor.max(), distance=max(1, int(4*h/(x[1]-x[0]))))
    chosen = sorted(peaks, key=lambda i: sensor[i], reverse=True)[:count]
    centers, widths = [], []
    span = max(2, int(3*h/(x[1]-x[0])))
    for i in chosen:
        field = int(np.argmax([g[i]/max(np.ptp(v), 1e-10) for g, v in zip(gradients, fields)]))
        v, g = fields[field], gradients[field]
        jump = abs(v[min(i+span, len(x)-1)]-v[max(i-span, 0)])
        centers.append(float(x[i]))
        widths.append(float(np.clip(jump/(2*max(g[i], 1e-12)), .25*h, 1.5*h)))
    return centers, widths


def run_enriched(n=65, mu=.0001, final_time=.1, width=.01, interval=.002, quadrature=12, cfl=.35, scales=(1.,), space_factory=None):
    if final_time <= 0 or width <= 0 or interval <= 0 or not 0 < cfl <= 1:
        raise ValueError("positive time, width, interval and 0 < cfl <= 1 required")
    if not scales or any(s <= 0 for s in scales):
        raise ValueError("positive enrichment width scales required")
    def make_space(base, centers, widths):
        if space_factory is not None:
            return space_factory(base, centers, widths)
        return EnrichedSpace(base, np.repeat(centers, len(scales)),
            (np.asarray(widths)[:, None]*np.asarray(scales)).ravel())
    start = perf_counter()
    base = BSPFNavierStokes(n, mu=mu, quadrature=quadrature)
    probes = np.linspace(0, 1, 2001)
    h = 1/(n-1)
    centers, widths = detect(probes, initial_state(probes, width), h)
    space = make_space(base, centers, widths)
    state = space.fit(initial_state(base.q, width))
    totals = space.integral@state
    t, steps, rejected, rejected_remaps = 0., 0, 0, 0
    next_update = interval
    updates = [dict(time=t, centers=centers, widths=widths, projection_l2=0., modes=space.k)]
    history = []
    def record():
        samples = space.evaluate(state)
        rho, _, p, _ = primitives(samples)
        current = space.integral@state
        history.append(dict(time=t, mass=float(current[0]), energy=float(current[2]),
            min_density=float(rho.min()), min_pressure=float(p.min()),
            entropy=float(base.weights@(rho*np.log(p/rho**base.gamma)/(base.gamma-1)))))
    record()
    while t < final_time-1e-14:
        rho, u, p, _ = primitives(space.evaluate(state))
        speed = np.max(abs(u)+np.sqrt(base.gamma*p/rho))
        diffusion = max(4*mu/3, base.kappa*(base.gamma-1))/rho.min()
        dt = min(cfl/(speed*np.sqrt(space.radius)/2.83+diffusion*space.radius/2.785), final_time-t, next_update-t)
        for attempt in range(13):
            try:
                k1 = space.rhs(state)
                k2 = space.rhs(state+dt*k1/2)
                k3 = space.rhs(state+dt*k2/2)
                k4 = space.rhs(state+dt*k3)
                trial = state+dt*(k1+2*k2+2*k3+k4)/6
                primitives(space.evaluate(trial))
                break
            except NonphysicalState:
                rejected += 1
                dt *= .5
        else:
            raise RuntimeError(f'nonphysical enriched stages at t={t}')
        if dt < 1e-14:
            raise RuntimeError(f'time step underflow at t={t}')
        state, t, steps = trial, t+dt, steps+1
        if t >= next_update-1e-13 and t < final_time-1e-13:
            centers, widths = detect(probes, space.at(probes, state), h)
            candidate = make_space(base, centers, widths)
            samples = space.evaluate(state)
            remapped = candidate.transfer_from(space, state) if hasattr(candidate, 'transfer_from') else candidate.fit(samples)
            try:
                primitives(candidate.evaluate(remapped))
                primitives(candidate.at(probes, remapped))
            except NonphysicalState:
                rejected_remaps += 1
            else:
                error = np.sqrt(np.sum(base.weights[:, None]*(candidate.evaluate(remapped)-samples)**2))
                updates.append(dict(time=t, centers=centers, widths=widths, projection_l2=float(error), modes=candidate.k))
                space, state = candidate, remapped
            next_update += interval
            record()
    record()
    output = space.at(probes, state)
    primitives(output)
    return dict(probes=probes, output=output, history=history, updates=updates,
        summary=dict(n=n, mu=mu, final_time=t, initial_width=width, quadrature=quadrature,
            cfl=cfl, update_interval=interval, width_scales=list(scales), enrichment_count=space.k, steps=steps,
            rejected_stages=rejected, rejected_remaps=rejected_remaps, seconds=perf_counter()-start,
            max_relative_mass_drift=max(abs(v['mass']/totals[0]-1) for v in history),
            max_relative_energy_drift=max(abs(v['energy']/totals[2]-1) for v in history)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=65)
    parser.add_argument('--reference-n', type=int, default=257)
    parser.add_argument('--mu', type=float, default=.0001)
    parser.add_argument('--time', type=float, default=.1)
    parser.add_argument('--width', type=float, default=.01)
    parser.add_argument('--interval', type=float, default=.002)
    parser.add_argument('--quadrature', type=int, default=12)
    parser.add_argument('--scales', type=float, nargs='+', default=[1.])
    parser.add_argument('--prefix', default='benchmarks/shock_tube/enrichment')
    args = parser.parse_args()
    results = {}
    for label, n in [('base', args.n), ('enriched', args.n), ('reference', args.reference_n)]:
        print(f'Running {label}, n={n}', flush=True)
        if label == 'enriched':
            result = run_enriched(n, args.mu, args.time, args.width, args.interval, args.quadrature, scales=args.scales)
        else:
            result = BSPFNavierStokes(n, mu=args.mu, quadrature=args.quadrature).solve(args.time, args.width, .35, progress=True)
        results[label] = result
        print(json.dumps(result['summary']), flush=True)
    reference = primitives(results['reference']['output'])
    metrics = {}
    for label in ('base', 'enriched'):
        values = primitives(results[label]['output'])
        metrics[label] = {name: dict(l1=float(np.trapezoid(abs(v-r), results[label]['probes'])),
            linf=float(np.max(abs(v-r))), total_variation=float(np.sum(abs(np.diff(v)))))
            for name, v, r in zip(('rho', 'u', 'p', 'T'), values, reference)}
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = dict(parameters=vars(args), metrics=metrics, results={k: {s: v[s] for s in ('summary', 'history')} for k, v in results.items()}, updates=results['enriched']['updates'])
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
    np.savez(prefix.with_suffix('.npz'), x=results['base']['probes'], **{k:v['output'] for k,v in results.items()})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label, result in results.items():
        for ax, name, values in zip(axes.flat, ('Density', 'Velocity', 'Pressure', 'Temperature'), primitives(result['output'])):
            ax.plot(result['probes'], values, label=label, lw=1.2)
            ax.set(xlabel='x', title=name)
            ax.legend()
    fig.suptitle(f'Adaptive tanh + BSPF, mu={args.mu:g}, t={args.time:g}, n={args.n}; reference n={args.reference_n}')
    fig.savefig(prefix.with_suffix('.png'), dpi=160)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
