# SPDX-License-Identifier: GPL-3.0-or-later
"""Local quintic B-spline enrichment, with retained refinement behind fronts.

Front locations come from the evolving density/pressure sensor. The local
basis centers lie on a fixed fine lattice. Previously selected centers are
retained: adaptation adds resolution instead of intentionally discarding
previously represented structures. Nearly dependent residual modes are removed
from newly added directions by a mass-Gram eigendecomposition; previously
accepted directions are preserved. FFT BSPF remains the global base space.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from shock_tube_enrichment import EnrichedSpace, run_enriched, primitives


class SplineSpace(EnrichedSpace):
    half_support = 3
    shape = BSpline.basis_element(np.arange(-3., 4.), extrapolate=False)
    shape_x = shape.derivative()

    def __init__(self, base, centers, spacing, previous=None):
        self.spacing = spacing
        retained = None
        if previous is not None:
            retained = np.zeros((len(centers), previous.k))
            retained[np.searchsorted(centers, previous.centers)] = previous.transform
        super().__init__(base, centers, np.full(len(centers), spacing),
                         orthogonalize=True, retained=retained,
                         retained_gram=None if previous is None else previous.small)

    def raw(self, x):
        z = (np.asarray(x)[:, None]-self.centers)/self.spacing
        return (np.nan_to_num(self.shape(z)), np.nan_to_num(self.shape_x(z))/self.spacing)


class SparseSplineSpace(SplineSpace):
    """Exact factored actions H=(S-Phi C)T using compact-support sparse S.

    The accepted trial space and mass Schur complement are unchanged. Dense H
    arrays are used during setup only and discarded after spectral estimation.
    """
    def __init__(self, base, centers, spacing, previous=None):
        self.previous_radius_vector=None if previous is None else previous.radius_vector
        self.correction_seed=None if previous is None else (np.searchsorted(centers,previous.centers),previous.correction)
        self.S, self.Sx = self.sparse_samples(base.q, np.asarray(centers), spacing)
        self.ST, self.SxT = self.S.T, self.Sx.T
        self._probe_cache = None
        super().__init__(base,centers,spacing,previous)
        walls = np.array([0, self.n-1])
        loads = np.zeros((self.n+self.k, 2))
        loads[walls] = np.eye(2)
        response = self.mass(loads)
        self.wall_lift = np.linalg.solve(response[walls].T, response.T).T
        del self.H, self.Hx, self.correction_seed

    def solve_rates(self, load):
        """One three-field mass solve, then a rank-two wall constraint."""
        rate = self.mass(load)
        rate[:, 1] -= self.wall_lift@rate[[0, self.n-1], 1]
        rate[[0, self.n-1], 1] = 0.
        return rate

    def _radius_operators(self):
        """Build temporary block products for the setup-only Lanczos solve."""
        if self.base.backend != 'fft':
            return super()._radius_operators()
        # For small 1D bases, cache just the two base blocks. Keep the
        # matrix-free path for larger bases; never assemble the full enriched
        # operator or a quadrature-by-enrichment matrix for time stepping.
        mass_action, stiffness_action = self.base.galerkin.mass, self.base.galerkin.stiffness
        if self.n <= 128:
            blocks = getattr(self.base, '_radius_base_blocks', None)
            if blocks is None:
                identity = np.eye(self.n)
                blocks = (mass_action(identity), stiffness_action(identity))
                self.base._radius_base_blocks = blocks
            mass_action, stiffness_action = blocks[0].__matmul__, blocks[1].__matmul__
        cross_stiffness = self.base.project(self.Hx, 1)
        small_stiffness = self.Hx.T@(self.base.weights[:, None]*self.Hx)
        def product(x, action, cross, small):
            b, a = x[:self.n], x[self.n:]
            return np.vstack((action(b)+cross@a, cross.T@b+small@a))
        return (lambda x: product(x, mass_action, self.cross, self.small),
                lambda x: product(x, stiffness_action, cross_stiffness, small_stiffness))

    @classmethod
    def sparse_samples(cls, x, centers, spacing):
        """Evaluate only entries inside each compact spline's support."""
        x = np.asarray(x)
        if np.any(np.diff(x) < 0):
            order = np.argsort(x)
            values, gradients = cls.sparse_samples(x[order], centers, spacing)
            inverse = np.argsort(order)
            return values[inverse], gradients[inverse]
        lower = np.searchsorted(x, centers-cls.half_support*spacing, side='left')
        upper = np.searchsorted(x, centers+cls.half_support*spacing, side='right')
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in zip(lower, upper)]) if len(centers) else np.array([], dtype=int)
        columns = np.repeat(np.arange(len(centers)), upper-lower)
        z = (x[rows]-centers[columns])/spacing
        shape = (len(x), len(centers))
        return (csr_matrix((np.nan_to_num(cls.shape(z)), (rows, columns)), shape=shape),
                csr_matrix((np.nan_to_num(cls.shape_x(z))/spacing, (rows, columns)), shape=shape))

    def raw(self, x):
        if x is self.base.q:
            return self.S.toarray(), self.Sx.toarray()
        return super().raw(x)

    def transfer_from(self, previous, state):
        """Inject coefficients into the retained basis, without an L2 solve."""
        if self.base is not previous.base or self.k < previous.k or self.spacing != previous.spacing:
            raise ValueError('transfer requires the same base and a retained spline space')
        indices=np.searchsorted(self.centers,previous.centers)
        if np.any(indices>=len(self.centers)) or not np.array_equal(self.centers[indices],previous.centers):
            raise ValueError('previous centers were not retained')
        other=np.ones(len(self.centers),dtype=bool);other[indices]=False
        if (not np.array_equal(self.correction[:,indices],previous.correction)
                or not np.array_equal(self.transform[indices,:previous.k],previous.transform)
                or np.any(self.transform[other,:previous.k])):
            raise ValueError('previous basis directions were not retained exactly')
        result=np.zeros((self.n+self.k,state.shape[1]))
        result[:self.n+previous.k]=state
        return result

    def evaluate_with_gradient(self,state):
        a=self.transform@state[self.n:]
        coefficients=state[:self.n]-self.correction@a
        if hasattr(self.base,'evaluate_with_gradient'):
            values,gradient=self.base.evaluate_with_gradient(coefficients)
            return values+self.S@a,gradient+self.Sx@a
        return (self.base.evaluate(coefficients)+self.S@a,
                self.base.evaluate(coefficients,1)+self.Sx@a)

    def radius_start(self,size):
        if self.previous_radius_vector is None:
            return np.random.default_rng(0).normal(size=size)
        seed=np.zeros(size)
        seed[:len(self.previous_radius_vector)]=self.previous_radius_vector
        seed+=1e-3*np.linalg.norm(seed)*np.random.default_rng(0).normal(size=size)/np.sqrt(size)
        return seed

    def evaluate(self,state,derivative=0):
        a=self.transform@state[self.n:]
        coefficients=state[:self.n]-self.correction@a
        return self.base.evaluate(coefficients,derivative)+(self.Sx if derivative else self.S)@a

    def project(self,values,derivative=0):
        b=self.base.project(values,derivative)
        local=(self.SxT if derivative else self.ST)@(self.base.weights[:,None]*values)
        return np.vstack((b,self.transform.T@(local-self.correction.T@b)))

    def at(self,x,state):
        # The detector and remap checks repeatedly use the same probe grid.
        # Share its base matrix across successive spaces; cache only one grid.
        x = np.asarray(x)
        if hasattr(self.base,'at'):
            if self._probe_cache is None or not np.array_equal(self._probe_cache[0],x):
                self._probe_cache=(x.copy(),self.sparse_samples(x,self.centers,self.spacing)[0])
            a=self.transform@state[self.n:]
            return self.base.at(x,state[:self.n]-self.correction@a)+self._probe_cache[1]@a
        cached = getattr(self.base, '_enrichment_probe_cache', None)
        if cached is None or not np.array_equal(cached[0], x):
            cached = (x.copy(), self.base.line.evaluate(x).matrix)
            self.base._enrichment_probe_cache = cached
        if self._probe_cache is None or not np.array_equal(self._probe_cache[0], x):
            self._probe_cache = (x.copy(), self.sparse_samples(x, self.centers, self.spacing)[0])
        a = self.transform@state[self.n:]
        return cached[1]@(state[:self.n]-self.correction@a)+self._probe_cache[1]@a


class LocalRefinement:
    def __init__(self, refinement=8, halfwidth=2., space_type=SplineSpace, retain_directions=True):
        self.refinement, self.halfwidth = refinement, halfwidth
        if refinement < 1 or halfwidth <= 0:
            raise ValueError("positive refinement and patch halfwidth required")
        self.space_type=space_type
        self.retain_directions=retain_directions
        self.indices = set()
        self.previous = None

    def __call__(self, base, centers, widths):
        h = 1/(base.n-1)
        spacing = h/self.refinement
        support = self.space_type.half_support
        for center in centers:
            low = max(support, int(np.floor((center-self.halfwidth*h)/spacing)))
            high = min(int(round(1/spacing))-support, int(np.ceil((center+self.halfwidth*h)/spacing)))
            self.indices.update(range(low, high+1))
        centers=np.array(sorted(self.indices))*spacing
        if self.previous is not None and self.previous.base is base and np.array_equal(centers,self.previous.centers):
            return self.previous
        self.previous = self.space_type(base, centers, spacing, previous=self.previous if self.retain_directions else None)
        return self.previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    parser.add_argument('--refinement', type=int, default=8)
    parser.add_argument('--halfwidth', type=float, default=2.)
    parser.add_argument('--quadrature', type=int, default=64)
    parser.add_argument('--interval', type=float, default=.005)
    parser.add_argument('--cfl', type=float, default=.35)
    parser.add_argument('--backend',choices=['dense','sparse'],default='sparse')
    parser.add_argument('--prefix', default='benchmarks/shock_tube/spline_enrichment')
    args = parser.parse_args()
    params = json.loads(Path(args.reference).with_suffix('.json').read_text())['parameters']
    result = run_enriched(n=params['n'], mu=params['mu'], final_time=params['time'],
        width=params['width'], interval=args.interval, quadrature=args.quadrature, cfl=args.cfl,
        space_factory=LocalRefinement(args.refinement, args.halfwidth,
            space_type=SparseSplineSpace if args.backend=='sparse' else SplineSpace))
    reference = np.load(args.reference)
    exact = primitives(reference['reference'])
    metrics = {name:dict(l1=float(np.trapezoid(abs(v-r),result['probes'])), linf=float(abs(v-r).max()),
        total_variation=float(np.sum(abs(np.diff(v)))))
        for name,v,r in zip(('rho','u','p','T'),primitives(result['output']),exact)}
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = dict(parameters=vars(args), metrics=metrics, **{s:result[s] for s in ('summary','history','updates')})
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
    np.savez(prefix.with_suffix('.npz'),x=result['probes'],output=result['output'])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for label, output in [('BSPF',reference['base']),('Local spline enrichment',result['output']),('Reference',reference['reference'])]:
        for ax,name,v in zip(axes.flat,('Density','Velocity','Pressure','Temperature'),primitives(output)):
            ax.plot(result['probes'],v,label=label,lw=1.1)
            ax.set(xlabel='x',title=name,xlim=(.35,.85))
            ax.legend(fontsize=8)
    fig.suptitle(f"Local spline enrichment: {params['n']} base + {result['summary']['enrichment_count']} modes, mu={params['mu']:g}")
    fig.savefig(prefix.with_suffix('.png'),dpi=160)
    print(json.dumps(dict(summary=result['summary'],metrics=metrics),indent=2),flush=True)


if __name__ == '__main__':
    main()
