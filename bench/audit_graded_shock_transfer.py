"""Test the single-front graded space on an existing shock-tube snapshot.

This is a projection audit, not a time evolution or an exact-error benchmark.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from shock_tube_enrichment import detect
from shock_tube_1d import BSPFNavierStokes,primitives
from graded_spline_shock import graded_factory
from viscous_traveling_shock import ShockSpace


def main():
    source=ROOT/'benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz'
    data=np.load(source);x=data['x'];reference=data['reference']
    interpolate=CubicSpline(x,reference)
    probes=np.linspace(0,1,4001)
    centers,widths=detect(probes,interpolate(probes),1/48,count=3)
    factory=graded_factory(degree=13,intervals=38,core_scale=.25,
        support_width=8.,core_padding=.5,quadrature=8)
    rows=[];outputs={}
    for n in (25,65,129):
        base=BSPFNavierStokes(n,mu=.0001,prandtl=.72,quadrature=8,
                             line_options=dict(n_basis=(n-1)//16+9))
        enrichment=factory(base,centers[:1],widths[:1])
        space=ShockSpace(enrichment.base,enrichment)
        state=space.fit_samples(interpolate(space.base.q),interpolate(np.array([0.,1.])))
        result=space.at(x,state);outputs[f'fit_{n}']=result
        metrics={}
        for field,a,b in zip(('rho','u','p','T'),primitives(result),primitives(reference)):
            i=int(np.argmax(abs(a-b)))
            metrics[field]=dict(linf=float(abs(a[i]-b[i])),x=float(x[i]))
        rows.append(dict(n=n,total_modes=n+enrichment.k,metrics=metrics))
    report=dict(reference=str(source.relative_to(ROOT)),
        reference_kind='Stored 1025-node numerical solution; not a certified 1e-8 reference',
        mu=.0001,prandtl=.72,time=.1,initial_width=.01,
        centers=centers,widths=widths,selected_center=centers[:1],runs=rows,
        note='Projection of a stored shock-tube snapshot only; no new time integration or runtime comparison. '
             'Cubic interpolation supplies quadrature data. Viscosity differs from the Becker benchmark.')
    out=ROOT/'benchmarks/shock_tube/graded_transfer_audit'
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez(out.with_suffix('.npz'),x=x,reference=reference,**outputs)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
