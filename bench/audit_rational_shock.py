"""Snapshot-only AAA compression audit; this does not evolve Navier--Stokes.

Only uniform real samples are passed to AAA. Dense exact data is used after
fitting to check function and first-derivative errors. Pole/residue conversion
is independently checked against the barycentric evaluation.
"""
import json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.interpolate import AAA

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/pde'))
from viscous_traveling_shock import BeckerShock,primitives


def barycentric_gradient(fit,x):
    """Differentiate after analytically removing the nearest support singularity.

    r=f_k+t*A/(w_k+t*B), t=x-z_k. Unlike converting to residues, this keeps
    the original fitted representation and is also valid exactly at z_k.
    """
    result=[];nodes=fit.support_points;f=fit.support_values;w=fit.weights
    for start in range(0,len(x),4096):
        d=x[start:start+4096,None]-nodes
        nearest=np.argmin(abs(d),axis=1);rows=np.arange(len(d));t=d[rows,nearest].copy()
        d[rows,nearest]=np.inf;inverse=1/d;delta=f[None,:]-f[nearest,None]
        B=inverse@w;A=(inverse*delta)@w
        Bp=-(inverse**2)@w;Ap=-(inverse**2*delta)@w
        denominator=w[nearest]+t*B;offset=t*A/denominator
        result.append((A+t*Ap-offset*(B+t*Bp))/denominator)
    return np.concatenate(result)


def main():
    test=np.linspace(0,1,64001)
    rows=[]
    cases=[(mu,center,8193) for mu in (.001,.0001) for center in (.4,.417)]
    cases += [(.0001,center,32769) for center in (.4,.417)]
    for mu,center,count in cases:
        train=np.linspace(0,1,count)
        shock=BeckerShock(mu=mu,center=center)
        samples=shock.state(train);exact,gradient=shock.state_gradient(test)
        fits=[];values=[];derivatives=[];conversion=[];minimum_pole_distance=[];terms=[];pole_gradient_errors=[]
        start=perf_counter()
        for field in range(3):
            fit=AAA(train,samples[:,field],rtol=1e-12,max_terms=80)
            fits.append(fit)
        fitting=perf_counter()-start
        for field,fit in enumerate(fits):
            value=fit(test);poles=fit.poles();residues=fit.residues()
            distance=test[:,None]-poles
            derivative=-np.sum(residues/distance**2,axis=1)
            # Equal-degree barycentric numerator/denominator has a constant
            # polynomial part. Anchor that constant at the left endpoint.
            constant=fit(0.)-np.sum(residues/(-poles))
            converted=constant+np.sum(residues/distance,axis=1)
            values.append(value);derivatives.append(barycentric_gradient(fit,test).real)
            pole_gradient_errors.append(float(np.max(abs(derivative.real-gradient[:,field]))))
            conversion.append(float(np.max(abs(converted-value))))
            pole_distance=np.hypot(poles.imag,np.maximum(0,np.maximum(-poles.real,poles.real-1)))
            minimum_pole_distance.append(float(np.min(pole_distance)))
            terms.append(len(fit.support_points))
        output=np.column_stack(values);g=np.column_stack(derivatives)
        errors={name:float(np.max(abs(v-e))) for name,v,e in zip(('rho','u','p','T'),primitives(output),primitives(exact))}
        row=dict(mu=mu,center=center,training_samples=len(train),validation_samples=len(test),
            relative_fit_tolerance=1e-12,support_terms_per_conservative_field=terms,
            max_primitive_error=max(errors.values()),primitive_errors=errors,
            max_absolute_conservative_derivative_error=float(np.max(abs(g-gradient))),
            relative_conservative_derivative_errors=(np.max(abs(g-gradient),axis=0)/np.max(abs(gradient),axis=0)).tolist(),
            partial_fraction_conversion_errors=conversion,minimum_distance_of_poles_to_domain=minimum_pole_distance,
            partial_fraction_gradient_errors=pole_gradient_errors,
            fit_seconds=fitting)
        rows.append(row);print(json.dumps(row),flush=True)
    folder=ROOT/'benchmarks/traveling_shock/rational_audit';folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(dict(rows=rows,
        note='Independent snapshot fits, not PDE evolution, not a matched-DOF comparison. Support locations, values and weights are all part of each fitted representation. Dense training samples are included in the audit description.'),indent=2))


if __name__=='__main__':main()
