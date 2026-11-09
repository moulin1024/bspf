# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent dense-output and local step/quadrature checks of saved coefficients.

This does not claim a full time-step-halved evolution; it compares one RK4 step
with two half steps starting from the saved final state, and changes quadrature
while holding the trial functions and coefficients fixed.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import cho_factor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from shock_tube_1d import OPTIONS
from viscous_traveling_shock import BeckerShock,BSPFNavierStokes,ShockSpace,spline_space_type,primitives


def restore(report,data,quadrature):
    base=BSPFNavierStokes(report['n'],mu=report['mu'],prandtl=.75,quadrature=quadrature,
        line_options=report.get('line_options',OPTIONS))
    if not report['enriched']:
        return ShockSpace(base)
    kind=spline_space_type(report['spline_degree'])
    e=kind.__new__(kind)
    e.base,e.n=base,base.n
    e.centers,e.correction,e.transform=data['centers'],data['correction'],data['transform']
    e.k=e.transform.shape[1];e.spacing=1/((base.n-1)*report['refinement'])
    e.S,e.Sx=e.sparse_samples(base.q,e.centers,e.spacing)
    e.ST,e.SxT=e.S.T,e.Sx.T;e._probe_cache=None
    H=(e.S.toarray()-base.evaluate(e.correction))@e.transform
    e.cross=base.project(H);e.small=H.T@(base.weights[:,None]*H)
    e.factors={}
    for interior in (False,True):
        cross=e.cross[1:-1] if interior else e.cross
        extension=base.solve_mass(cross,interior)
        schur=e.small-cross.T@extension
        e.factors[interior]=(extension,cho_factor((schur+schur.T)/2))
    loads=np.zeros((e.n+e.k,2));loads[[0,e.n-1]]=np.eye(2)
    response=e.mass(loads)
    e.wall_lift=np.linalg.solve(response[[0,e.n-1]].T,response.T).T
    e.radius=0. # Not used for this local step check.
    return ShockSpace(base,e)


def sample(space,state,x):
    return np.vstack([space.at(x[i:i+2048],state) for i in range(0,len(x),2048)])


def rk4(space,state,t,h,shock):
    k1=space.rhs(state,t,shock)
    k2=space.rhs(state+h*k1/2,t+h/2,shock)
    k3=space.rhs(state+h*k2/2,t+h/2,shock)
    k4=space.rhs(state+h*k3,t+h,shock)
    out=state+h*(k1+2*k2+2*k3+k4)/6
    out[space.walls]=shock.state(np.array([0.,1.]),t+h)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix')
    p.add_argument('--quadrature',type=int,required=True)
    a=p.parse_args();prefix=ROOT/a.prefix
    r=json.loads(prefix.with_suffix('.json').read_text());data=np.load(prefix.with_suffix('.npz'))
    state=data['coefficients'];shock=BeckerShock(mu=r['mu'],speed=r.get('speed',.5),center=r.get('center',.4))
    space=restore(r,data,r['quadrature']);x=np.linspace(0,1,64001)
    output=sample(space,state,x);target=shock.state(x,r['time'])
    metrics={key:dict(linf=float(abs(v-z).max()),l2=float(np.sqrt(np.trapezoid((v-z)**2,x))))
        for key,v,z in zip(('rho','u','p','T'),primitives(output),primitives(target))}
    reproduction=float(abs(sample(space,state,data['x'])-data['output']).max())
    h=r['time']/r['steps'];t=r['time']
    whole=rk4(space,state,t,h,shock)
    half=rk4(space,rk4(space,state,t,h/2,shock),t+h/2,h/2,shock)
    probes=np.linspace(0,1,16001)
    local_step_difference=float(abs(sample(space,whole,probes)-sample(space,half,probes)).max())
    refined=restore(r,data,a.quadrature)
    fine_step=rk4(refined,state,t,h,shock)
    quadrature_step_difference=float(abs(sample(space,whole,probes)-sample(refined,fine_step,probes)).max())
    validation=dict(metrics=metrics,reproduction_error=reproduction,local_step=h,
        step_halving_conservative_difference=local_step_difference,
        refined_quadrature=a.quadrature,quadrature_one_step_conservative_difference=quadrature_step_difference,
        note='Local time and quadrature checks at final time; not a full re-integration with refined parameters.')
    Path(str(prefix)+'_check.json').write_text(json.dumps(validation,indent=2))
    np.savez(str(prefix)+'_dense.npz',x=x,output=output,exact=target)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
