"""Independent checks of the serialized patch geometry and connectivity."""
import json
from pathlib import Path
import numpy as np
from generate_patches import metrics, basis


def validate(folder):
    with np.load(folder/'patches.npz') as data:
        geometry=data['geometry']; ids=data['connectivity']; interfaces=data['interfaces']
    boundary=json.loads((folder/'boundary.json').read_text())
    quality=json.loads((folder/'quality.json').read_text())
    assert np.isfinite(geometry).all()
    def face(a,i):
        return [a[0,:],a[-1,:],a[:,0],a[:,-1]][i]
    used=set(); max_gap=0.
    B=basis(np.linspace(0,1,101))
    for p,f,r,g,reverse in interfaces:
        assert (p,f) not in used and (r,g) not in used
        used.update([(p,f),(r,g)])
        left=face(geometry[p],f);right=face(geometry[r],g)
        left_ids=face(ids[p],f);right_ids=face(ids[r],g)
        if reverse: right=right[::-1];right_ids=right_ids[::-1]
        assert np.array_equal(left_ids,right_ids)
        max_gap=max(max_gap,float(np.max(np.linalg.norm(B@left-B@right,axis=1))))
    for b in boundary:
        key=(b['patch'],b['face'])
        assert key not in used
        used.add(key)
    assert len(used)==4*len(geometry)
    corners=ids[:,[0,-1,-1,0],[0,0,-1,-1]]
    V=len(np.unique(corners));E=len(interfaces)+len(boundary);F=len(geometry)
    assert V-E+F==quality['domain_components']-quality['holes']
    # A different sampling set from the generator: 51-point Gauss nodes + endpoints.
    q=np.r_[0.,(np.polynomial.legendre.leggauss(51)[0]+1)/2,1.]
    minimum=np.inf;condition=0.;scaled=1.
    for chunk in np.array_split(geometry,max(1,len(geometry)//40)):
        d,s,k=metrics(chunk,q)
        minimum=min(minimum,float(d.min()));condition=max(condition,float(k.max()))
        scaled=min(scaled,float(s.min()))
    assert minimum>0 and max_gap<1e-12
    assert quality['certified_patches']==len(geometry)
    assert quality['status']=='geometry_checks_passed'
    result=dict(passed=True,patches=len(geometry),max_interface_gap=max_gap,
                euler_characteristic=V-E+F,min_jacobian=minimum,
                min_scaled_jacobian=scaled,max_map_condition=condition,
                note='Geometry/topology validation; not a Poisson MMS solve or direct Fortran classifier comparison.')
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    for case in ['tcv','iter']:
        print(case,validate(Path(__file__).resolve().parent/'meshes'/case),flush=True)
