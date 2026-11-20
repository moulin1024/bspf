"""Replay the stored current in the restart's own bicubic test space, varying quadrature."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from campaign import Restart, rule, gradient, run


class CubicAxis:
    """Cubic Lagrange basis on four Lobatto nodes for the C0 control."""
    x = np.array([0., (1 - 1 / np.sqrt(5)) / 2, (1 + 1 / np.sqrt(5)) / 2, 1.])
    n = len(x)

    def evaluate(self, points, order):
        basis = []
        for i, node in enumerate(self.x):
            poly = np.polynomial.Polynomial.fromroots(np.delete(self.x, i))
            basis.append(poly / poly(node))
        return [np.column_stack([p.deriv(k)(points) for p in basis])
                for k in range(order + 1)]


def native_check(path, orders=(4,8,16)):
    ref = Restart(path)
    with h5py.File(path) as f:
        vertex = f['vertex'][:]-1
        ix = f['index'][:]-1
        scales = f['size'][:].transpose(2,0,1).reshape(-1,16)
        bc = f['boundary'][:]
    g = ix[:,vertex].transpose(2,0,1).reshape(-1,16)
    fixed = np.unique(np.r_[ix[0,bc!=0],
        ix[1,np.isin(bc,[1,3,4,9,11,12,19,20,21])],
        ix[2,np.isin(bc,[2,3,5,9,15,19,20,21])]])
    out = []
    for order in orders:
        z,w = np.polynomial.legendre.leggauss(order)
        bf = rule((z+1)/2)
        weight = np.outer(w,w).ravel()/4
        K, rhs = [],np.zeros(g.max()+1)
        for p,idx in enumerate(g):
            R,_,_,j = ref.fields(p,bf[0])
            a,c,ps,_ = ref.fields(p,bf[1])
            b,d,pt,_ = ref.fields(p,bf[2])
            det = a*d-b*c
            gr,gz = gradient(ps,pt,a,b,c,d,det)
            dr,dz = gradient(bf[1],bf[2],a,b,c,d,det)
            scale = scales[p,:,None]
            dr,dz = dr*scale,dz*scale
            wt = weight*abs(det)/R
            K.append((dr*wt)@dr.T+(dz*wt)@dz.T)
            np.add.at(rhs,idx,-(bf[0]*scale)@(wt*j)-dr@(wt*gr)-dz@(wt*gz))
        rows = np.broadcast_to(g[:,:,None],(len(g),16,16))
        cols = np.broadcast_to(g[:,None,:],rows.shape)
        A = coo_matrix((np.ravel(K),(rows.ravel(),cols.ravel())),shape=(len(rhs),len(rhs))).tocsr()
        free = np.setdiff1d(np.flatnonzero(A.diagonal()>0),fixed)
        delta = np.zeros_like(rhs)
        delta[free] = spsolve(A[free][:,free],rhs[free])
        q,w = np.polynomial.legendre.leggauss(20)
        tab = rule((q+1)/2)
        sums=np.zeros(4)
        for p,idx in enumerate(g):
            R,_,v,_ = ref.fields(p,tab[0])
            a,c,ps,_ = ref.fields(p,tab[1])
            b,d,pt,_ = ref.fields(p,tab[2])
            det = a*d-b*c
            diff = delta[idx]*scales[p]
            dv,ds,dt = [diff@basis for basis in tab]
            dr,dz = gradient(ds,dt,a,b,c,d,det)
            gr,gz = gradient(ps,pt,a,b,c,d,det)
            wt = np.outer(w,w).ravel()/4*abs(det)
            sums += [wt@dv**2,wt@v**2,wt@((dr**2+dz**2)/R**2),wt@((gr**2+gz**2)/R**2)]
        out.append(dict(assembly_order=order,flux_relative_change=np.sqrt(sums[0]/sums[1]),
                        field_relative_change=np.sqrt(sums[2]/sums[3])))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--c0-output', type=Path, help='Also run the value-continuous bicubic control')
    args=parser.parse_args()
    if args.c0_output:
        control = run(Restart(args.restart), 3, axis=CubicAxis(), assembly_order=16)
        args.c0_output.write_text(json.dumps(control, indent=2, allow_nan=False) + '\n')
    result=native_check(args.restart)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result)
