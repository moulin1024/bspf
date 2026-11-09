# SPDX-License-Identifier: GPL-3.0-or-later
"""Separate enriched-space approximation, coordinates, truncation and selection.

Projection diagnostics use dense weighted SVD independently of the production
Schur solver. Evolution uses the existing structured FFT Galerkin solver.
"""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.linalg import lstsq, solve as dense_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/pde'))
from viscous_traveling_shock import BeckerShock, BSPFNavierStokes, spline_space_type, solve, primitives
from shock_tube_spline_enrichment import LocalRefinement


def peak(values, target):
    return float(max(np.max(abs(a-b)) for a,b in zip(primitives(values), primitives(target))))


def projection(n, quadrature=64):
    shock = BeckerShock()
    base = BSPFNavierStokes(n, mu=shock.mu, prandtl=.75, quadrature=quadrature,
                           line_options=dict(n_basis=(n-1)//16+9))
    factory = LocalRefinement(4, 12, spline_space_type(13), retain_directions=False)
    space = factory(base, [.4], [shock.w])
    P = base.evaluate(np.eye(n))
    S = space.S.toarray()
    C = space.correction
    R = S-P@C
    eig, V = np.linalg.eigh(R.T@(base.weights[:,None]*R))
    x = np.linspace(0, 1, 16001)
    Px = base.line.evaluate(x).matrix
    Sx = space.sparse_samples(x, space.centers, space.spacing)[0].toarray()
    Rx = Sx-Px@C
    # Prescribed boundary lifting is identical for all spaces and coordinates.
    boundary = shock.state(np.array([0., 1.]))
    lift = P[:, [0,-1]]@boundary
    lift_x = Px[:, [0,-1]]@boundary
    y = shock.state(base.q)-lift
    exact = shock.state(x)
    sqrtw = np.sqrt(base.weights)[:,None]
    rows, predictions = [], {}

    def fit(name, E, Ex):
        A = np.column_stack((P[:,1:-1], E))
        Ax = np.column_stack((Px[:,1:-1], Ex))
        Aw = sqrtw*A
        z, _, rank, singular = lstsq(Aw, sqrtw*y, cond=1e-14, lapack_driver='gelsd')
        value = lift_x+Ax@z
        cond = float(singular[0]/singular[-1])
        row = dict(name=name, total_modes=A.shape[1]+2, rank=int(rank),
                   design_condition=cond, mass_condition_estimate=cond**2,
                   primitive_peak=peak(value,exact), coefficient_norm=float(np.linalg.norm(z)),
                   conservative_l2=float(np.linalg.norm(sqrtw*(A@z-y))),
                   normal_equation_warning=False)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                zn = dense_solve(Aw.T@Aw, Aw.T@(sqrtw*y), assume_a='sym')
            row['normal_equation_warning'] = bool(caught)
            row['normal_equation_prediction_delta'] = float(np.max(abs(Ax@(zn-z))))
        except np.linalg.LinAlgError:
            row['normal_equation_prediction_delta'] = None
            row['normal_equation_warning'] = True
        rows.append(row)
        predictions[name] = value

    fit('base', np.empty((len(P),0)), np.empty((len(x),0)))
    fit('raw_all', S, Sx)
    fit('residual_all', R, Rx)
    cutoff = max(1e-15, 1e-12*eig[-1])
    keep = eig > cutoff
    # Same retained subspace in three coordinate systems: no rank confound.
    Vk = V[:,keep]
    T = Vk/np.sqrt(eig[keep])
    fit('raw_retained', S@Vk, Sx@Vk)
    fit('residual_retained', R@Vk, Rx@Vk)
    fit('whitened_retained', R@T, Rx@T)
    for relative in (1e-10, 1e-8):
        chosen = eig > max(1e-15, relative*eig[-1])
        transform = V[:,chosen]/np.sqrt(eig[chosen])
        fit(f'whitened_cutoff_{relative:g}', R@transform, Rx@transform)
    reference = predictions['whitened_retained']
    for row in rows:
        row['delta_from_whitened'] = float(np.max(abs(predictions[row['name']]-reference)))
    matched_n = n+int(sum(keep))
    matched_n += 1-matched_n%2  # odd cardinal grid, at most one extra coefficient
    matched = BSPFNavierStokes(matched_n,mu=shock.mu,prandtl=.75,quadrature=quadrature,
                              line_options=dict(n_basis=(matched_n-1)//16+9))
    MP = matched.evaluate(np.eye(matched_n))
    my = shock.state(matched.q)-MP[:,[0,-1]]@boundary
    mz = lstsq(np.sqrt(matched.weights)[:,None]*MP[:,1:-1],
               np.sqrt(matched.weights)[:,None]*my,cond=1e-14)[0]
    MX = matched.line.evaluate(x).matrix
    matched_peak = peak(MX[:,[0,-1]]@boundary+MX[:,1:-1]@mz,exact)
    report = dict(n=n, quadrature=quadrature, dictionary_size=len(space.centers),
                  retained=int(sum(keep)), gram_cutoff=float(cutoff),
                  interior_orthogonality=float(np.max(abs(P[:,1:-1].T@(base.weights[:,None]*R)))),
                  whitening_defect=float(np.max(abs(T.T@(R.T@(base.weights[:,None]*R))@T-np.eye(sum(keep))))),
                  matched_ordinary=dict(n=matched_n,primitive_peak=matched_peak),rows=rows)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='benchmarks/traveling_shock/ablations')
    parser.add_argument('--projection-only', action='store_true')
    parser.add_argument('--resume', action='store_true',help='Reuse existing case files for this configuration')
    args=parser.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for n,q in ((65,64),(129,64),(193,64),(129,128)):
        report=projection(n,q); reports.append(report)
        (out/'projection.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
        print('projection',n,q,[(r['name'],r['primitive_peak']) for r in report['rows']],flush=True)
    if args.projection_only:
        return
    common=dict(n=129,mu=.001,final_time=.1,quadrature=48,cfl=.7,interval=.005,
                refinement=4,halfwidth=4,spline_degree=9,recondition=True,
                line_options=dict(n_basis=17),output_points=16001,progress=True)
    results={}
    def run(name, **options):
        if args.resume and (out/f'{name}.npz').exists():
            with np.load(out/f'{name}.npz') as saved:
                r={key:saved[key] for key in saved.files}
            r['summary']=json.loads((out/f'{name}.json').read_text())
        else:
            r=solve(**(common|options))
        results[name]=r
        (out/f'{name}.json').write_text(json.dumps(r['summary'],indent=2))
        np.savez(out/f'{name}.npz',**{k:r[k] for k in ('x','output','exact','centers') if k in r})
        print(name,r['summary']['final_modes'],max(v['linf'] for v in r['summary']['metrics'].values()),flush=True)
    run('ordinary',enriched=False)
    run('fixed_initial',enriched=True,selection='fixed')
    run('detected',enriched=True)
    run('oracle',enriched=True,selection='oracle')
    run('detected_retained',enriched=True,recondition=False)
    path=sorted({c for u in results['detected']['summary']['updates'] for c in u['centers']})
    run('fixed_union',enriched=True,selection='fixed',fixed_centers=path)
    matched_n=results['detected']['summary']['final_modes']
    matched_n += 1-matched_n%2
    run('ordinary_matched',enriched=False,n=matched_n,line_options=dict(n_basis=(matched_n-1)//16+9))
    # Same dictionary controls at half time step and doubled quadrature.
    run('detected_half_cfl',enriched=True,cfl=.35)
    run('detected_double_quadrature',enriched=True,quadrature=96)
    summary={name:r['summary'] for name,r in results.items()}
    for name in ('detected_half_cfl','detected_double_quadrature'):
        summary[name]['primitive_delta_from_detected']=peak(results[name]['output'],results['detected']['output'])
    (out/'evolution.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for name in ('ordinary','ordinary_matched','fixed_initial','detected','oracle','fixed_union','detected_retained'):
        r=results[name]; axes[0].plot(r['x'],r['output'][:,0],label=name)
        axes[1].semilogy(r['x'],np.maximum(abs(r['output'][:,0]-r['exact'][:,0]),1e-16),label=name)
        u=r['summary']['updates']
        axes[2].step([v['time'] for v in u]+[.1],[v['modes'] for v in u]+[u[-1]['modes']],where='post',label=name)
    axes[0].plot(r['x'],r['exact'][:,0],'k--',label='exact')
    axes[0].set(xlim=(.37,.49),xlabel='x',ylabel='Density')
    axes[1].set(xlim=(.3,.55),xlabel='x',ylabel='Absolute density error')
    axes[2].set(xlabel='Time',ylabel='Coefficients per field')
    axes[0].legend(fontsize=8)
    fig.savefig(out/'selection.png',dpi=170);fig.savefig(out/'selection.pdf')


if __name__=='__main__':
    main()
