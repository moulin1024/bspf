"""Exercise the restart reader, shared nodes, source sign and field derivatives."""
from pathlib import Path
import tempfile
import h5py
import numpy as np
from campaign import Restart,run


def check():
    # Four sheared cubic elements, exact psi=R^3+Z^2, j=Delta*psi=3R+2.
    s,t = np.meshgrid(np.arange(3),np.arange(3),indexing='ij')
    dr = np.array([.25,.05])
    dz = np.array([-.03,.3])
    R = (1+s*dr[0]+t*dr[1]).ravel()
    Z = (-.4+s*dz[0]+t*dz[1]).ravel()
    x = np.zeros((2,4,1,9))
    x[:,0,0] = [R,Z]
    x[0,1:3,0] = dr[:,None]/3
    x[1,1:3,0] = dz[:,None]/3
    values = np.zeros((3,4,1,9))
    values[0,0,0] = R**3+Z**2
    values[0,1:3,0] = (3*dr[:,None]*R**2+2*dz[:,None]*Z)/3
    values[0,3,0] = (6*R*np.prod(dr)+2*np.prod(dz))/9
    values[2,0,0] = 3*R+2
    values[2,1:3,0] = dr[:,None]
    corner = np.array([[0,1,3,4]])+np.array([[0],[3],[4],[1]])
    signs = np.array([[1,1,1,1],[1,-1,-1,1],[1,1,-1,-1],[1,-1,1,-1]])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'synthetic.h5'
        with h5py.File(path,'w') as f:
            f['vertex'] = corner+1
            f['size'] = np.broadcast_to(signs[:,:,None],(4,4,4))
            f['x'],f['values'] = x,values
        ref = Restart(path)
        row = run(ref,4)
        assert row['free_dofs']==121 and max(row['flux_relative_L2'],row['Bp_relative_L2'])<1e-9,row
        print(row)

if __name__=='__main__':
    check()
