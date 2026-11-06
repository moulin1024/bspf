# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent dense-grid audit and broadening measurement for constrained BSPF."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'/'pde'))
from shock_tube_monotone import BSPFNavierStokes, MonotoneLimiter, diagnostics, primitives


def crossing(x,p,value):
    mask=(x>.6)&(x<.85)
    x,p=x[mask],p[mask]
    ids=np.flatnonzero((p[:-1]>=value)&(p[1:]<=value))
    if not len(ids):
        return None
    i=ids[0]
    return float(x[i]+(x[i+1]-x[i])*(p[i]-value)/(p[i]-p[i+1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='benchmarks/shock_tube/monotone_evolution65')
    args=parser.parse_args()
    prefix=Path(args.input)
    saved=np.load(prefix.with_suffix('.npz'))
    summary=json.loads(prefix.with_suffix('.json').read_text())
    ref=np.load('benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    base=BSPFNavierStokes(summary['n'],mu=.0001,quadrature=12)
    limiter=MonotoneLimiter(base)
    output=base.line.evaluate(ref['x']).matrix@saved['state']
    target=primitives(ref['reference'])
    metrics={}
    left=float(np.median(target[2][(ref['x']>.62)&(ref['x']<.65)]))
    right=float(np.median(target[2][(ref['x']>.72)&(ref['x']<.75)]))
    for name,U in [('original',ref['base']),('limited',output),('reference',ref['reference'])]:
        values=primitives(U)
        row={field:dict(l1=float(np.trapezoid(abs(v-r),ref['x'])),linf=float(abs(v-r).max()),
            min=float(v.min()),max=float(v.max()),total_variation=float(abs(np.diff(v)).sum()))
            for field,v,r in zip(('rho','u','p','T'),values,target)}
        xs=[crossing(ref['x'],values[2],right+fraction*(left-right)) for fraction in (.9,.5,.1)]
        row['shock_pressure_90_to_10_width']=None if None in xs else xs[2]-xs[0]
        row['shock_pressure_midpoint']=xs[1]
        row['temperature_extrema_prominence_1e4']=int(len(find_peaks(values[3],prominence=1e-4)[0])+len(find_peaks(-values[3],prominence=1e-4)[0]))
        metrics[name]=row
    audit=diagnostics(base,limiter,saved['state'],np.linspace(0,1,32769))
    report=dict(evolution=summary,independent_audit_32769=audit,metrics=metrics,
        notes=['The limiter constrains samples; the independent audit is not a proof of continuum monotonicity.',
               'Pressure shock width uses fixed plateau levels measured from the numerical reference.',
               'Limiting imposes pre-reflection Sod shape information and introduces numerical dissipation.'])
    Path(str(prefix)+'_audit.json').write_text(json.dumps(report,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,U in [('Original BSPF 65',ref['base']),('Constrained BSPF 65',output),('Reference 1025',ref['reference'])]:
        for ax,title,v in zip(axes.flat,('Density','Velocity','Pressure','Temperature'),primitives(U)):
            ax.plot(ref['x'],v,label=name,lw=1.1)
            ax.set(title=title,xlabel='x',xlim=(.3,.85));ax.legend(fontsize=8)
    fig.suptitle(f"Sod tube, mu=1e-4, t=0.1: shape constraints + artificial diffusion {summary.get('artificial_diffusion',0):g}")
    fig.savefig(str(prefix)+'_comparison.png',dpi=160)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
