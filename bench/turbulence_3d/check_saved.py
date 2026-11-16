"""CPU read-back validation of a saved curved nonperiodic 3D MMS dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
from mms import CurvedBox, PlaneWaveMMS
from generate import cpu_reference


def check(folder):
    report=json.loads((folder/'report.json').read_text())
    if report['status']!='validated' or report['periodic']:
        raise ValueError('Expected a validated nonperiodic dataset')
    digest=hashlib.sha256((folder/'modes.npz').read_bytes()).hexdigest()
    assert digest==report['modes_file_sha256']
    field=PlaneWaveMMS.load(folder/'modes.npz')
    domain=CurvedBox(report['map_amplitudes'])
    errors={}
    with h5py.File(folder/'fields.h5') as f:
        assert not f.attrs['periodic'] and f.attrs['modes_file_sha256']==digest
        n=len(f['reference_axis']);rng=np.random.default_rng(4321)
        indices=np.vstack((rng.integers(0,n,(32,3)),[[0,0,0],[n-1,n-1,n-1],[0,n-1,0]]))
        x=np.array([f['coordinates'][tuple(i)] for i in indices])
        mapped,_,det=domain.map(indices/(n-1))
        errors['coordinates']=float(np.max(abs(x-mapped)))
        errors['jacobian']=float(np.max(abs(np.array([f['jacobian'][tuple(i)] for i in indices])-det)))
        reference=cpu_reference(field,x)
        for name,expected in reference.items():
            if name not in f:continue
            stored=np.array([f[name][tuple(i)] for i in indices])
            errors[name]=float(np.max(abs(stored-expected))/max(1.,np.max(abs(expected))))
        g=reference['velocity_gradient']
        vort=np.stack((g[:,2,1]-g[:,1,2],g[:,0,2]-g[:,2,0],g[:,1,0]-g[:,0,1]),axis=1)
        errors['vorticity']=float(np.max(abs(np.array([f['vorticity'][tuple(i)] for i in indices])-vort))/max(1.,np.max(abs(vort))))
        for axis in range(3):
            for side in (0,1):
                key=f'{"xyz"[axis]}{side}';group=f['boundary'][key];ij=rng.integers(0,n,(12,2))
                bx=np.array([group['coordinates'][tuple(i)] for i in ij]);bu=np.array([group['velocity'][tuple(i)] for i in ij]);bp=np.array([group['pressure'][tuple(i)] for i in ij])
                expected,area=domain.face(axis,side,ij/(n-1));normal=area/np.linalg.norm(area,axis=1)[:,None]
                truth=cpu_reference(field,bx)
                errors[key+'_coordinates']=float(np.max(abs(bx-expected)))
                errors[key+'_normal']=float(np.max(abs(normal-np.array([group['outward_normal'][tuple(i)] for i in ij]))))
                errors[key+'_velocity']=float(np.max(abs(bu-truth['velocity']))/max(1.,np.max(abs(bu))))
                errors[key+'_pressure']=float(np.max(abs(bp-truth['pressure'])))
    assert max(errors.values())<2e-11,errors
    file_hash=hashlib.sha256()
    with (folder/'fields.h5').open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):file_hash.update(block)
    result=dict(status='passed',volume_probe_count=len(indices),boundary_probe_count=72,
                relative_or_absolute_max_errors=errors,fields_file_sha256=file_hash.hexdigest())
    (folder/'storage_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: saved coordinates, Jacobians, fields, forcing and six boundary traces; max error',max(errors.values()))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    check(parser.parse_args().folder)
