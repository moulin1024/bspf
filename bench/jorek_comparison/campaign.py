"""Compare a model-303 axisymmetric bicubic JOREK restart with BSPF."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bench/curved_poisson/src'))
from optimized_bvp import HybridAxis


class Restart:
    def __init__(self, path):
        with h5py.File(path) as f:
            vertex = f['vertex'][:].astype(int)-1
            size = f['size'][:]
            x = f['x'][:]
            values = f['values']
            if x.ndim == 4 and x.shape[2] == 1:
                x = x[:,:,0,:]
            if (vertex.shape[0] != 4 or size.shape != (4,*vertex.shape)
                    or x.ndim != 3 or x.shape[:2] != (2,4)
                    or values.shape[1:] != (4,1,x.shape[-1])):
                raise ValueError('expected bicubic axisymmetric model-303 restart layout')
            if vertex.min()<0 or vertex.max()>=x.shape[-1]:
                raise ValueError('invalid restart connectivity')
            psi_j = values[[0,2],:,0,:]
            self.coeff = np.concatenate([x,psi_j])[:,:,vertex]*size
            self.count = vertex.shape[1]
            if not np.isfinite(self.coeff).all():
                raise ValueError('nonfinite geometry or fields')

    def fields(self,p,b):
        return self.coeff[:,:,:,p].reshape(4,16) @ b


def rule(nodes, t_nodes=None, *, mixed=False):
    s,t = np.meshgrid(nodes,nodes if t_nodes is None else t_nodes,indexing='ij')
    # Tensor Bernstein/Hermite convention of JOREK's jorek_read_h5.py:
    # inward endpoint tangents have factor 3; element.size carries their signs.
    def basis(x, end, tangent, derivative):
        y = 1-x if end else x
        if tangent:
            value = 3*(1-y)*(1-3*y) if derivative else 3*y*(1-y)**2
        else:
            value = -6*y*(1-y) if derivative else (1-y)**2*(1+2*y)
        return -value if derivative and end else value
    return [np.array([basis(s.ravel(),i,hs,ds)*basis(t.ravel(),j,ht,dt)
                      for hs,ht in [(0,0),(1,0),(0,1),(1,1)]
                      for i,j in [(0,0),(1,0),(1,1),(0,1)]])
            for ds,dt in ([(0,0),(1,0),(0,1),(1,1)] if mixed else [(0,0),(1,0),(0,1)])]


def gradient(ds,dt,a,b,c,d,det):
    return (d*ds-c*dt)/det, (a*dt-b*ds)/det


def run(ref, degree, check_order=18, *, diagnose=None, assembly_order=None, axis=None, exact=None, observe=None):
    start = time.perf_counter()
    if axis is None:
        axis = HybridAxis(1,degree=degree,nodes='symmetric_qr')
    n = axis.n
    # Merge physical traces, including coincident axis vertices of collapsed cells.
    nodal = rule(axis.x)[0]
    coords = np.stack([ref.fields(p,nodal)[:2].T for p in range(ref.count)])
    _,ids = np.unique(np.round(coords.reshape(-1,2),11),axis=0,return_inverse=True)
    ids = ids.reshape(ref.count,n,n)
    edges = {}
    for grid in ids:
        for side in [grid[:,0],grid[-1],grid[:,-1],grid[0]]:
            if side.min()==side.max(): continue
            key = tuple(min(tuple(side),tuple(side[::-1])))
            edges[key] = edges.get(key,0)+1
    if any(count>2 for count in edges.values()): raise ValueError('nonmanifold edge')
    wall = np.unique([edge for edge,count in edges.items() if count==1])
    if not len(wall): raise ValueError('missing exterior edges')
    q,w = np.polynomial.legendre.leggauss(assembly_order or 2*n+2)
    def tables(q,w):
        q,w = (q+1)/2,w/2
        b,d = axis.evaluate(q,1)
        return rule(q),np.kron(b,b),np.kron(d,b),np.kron(b,d),np.outer(w,w).ravel()
    geo,B,Ds,Dt,weights = tables(q,w)
    ndof = ids.max()+1
    prescribed = np.zeros(ndof)
    rhs = np.zeros(ndof)
    local = []
    def metrics(p,geo):
        R,Z,psi,j = ref.fields(p,geo[0])
        a,c,ps,unused = ref.fields(p,geo[1])
        b,d,pt,unused = ref.fields(p,geo[2])
        det = a*d-b*c
        if R.min()<=0 or np.any(det==0) or np.any(np.sign(det)!=np.sign(det[0])):
            raise ValueError(f'invalid interior geometry in element {p}')
        if exact is not None:
            psi, gr, gz, j = exact(R, Z)
            ps, pt = gr*a + gz*c, gr*b + gz*d
        return R,psi,j,a,b,c,d,det,ps,pt

    for p,grid in enumerate(ids):
        g = grid.ravel()
        R,psi,j,a,b,c,d,det,_,_ = metrics(p,geo)
        dr,dz = gradient(Ds,Dt,*[v[:,None] for v in (a,b,c,d,det)])
        wt = weights*abs(det)/R
        local.append(dr.T@(wt[:,None]*dr)+dz.T@(wt[:,None]*dz))
        np.add.at(rhs,g,B.T@(-wt*j))
        prescribed[g] = (ref.fields(p,nodal)[2] if exact is None
                         else exact(*ref.fields(p,nodal)[:2])[0])
    g = ids.reshape(ref.count,-1)
    rows = np.broadcast_to(g[:,:,None],(ref.count,n*n,n*n)).ravel()
    cols = np.broadcast_to(g[:,None,:],(ref.count,n*n,n*n)).ravel()
    A = coo_matrix((np.ravel(local),(rows,cols)),shape=(ndof,ndof)).tocsr()
    free = np.setdiff1d(np.arange(ndof),wall)
    psi_b = np.zeros(ndof)
    psi_b[wall] = prescribed[wall]
    load = (rhs-A@psi_b)[free]
    matrix = A[free][:,free].tocsc()
    setup = time.perf_counter()-start
    start = time.perf_counter()
    psi_b[free] = spsolve(matrix,load)
    solve = time.perf_counter()-start
    residual = np.linalg.norm(matrix@psi_b[free]-load)/max(np.linalg.norm(load),1e-300)
    geo,B,Ds,Dt,w = tables(*np.polynomial.legendre.leggauss(check_order))
    sums = np.zeros(4)
    for p,gp in enumerate(g):
        R,psi,j,a,b,c,d,det,ps,pt = metrics(p,geo)
        u = psi_b[gp]
        fr,fz = gradient(Ds@u-ps,Dt@u-pt,a,b,c,d,det)
        rr,rz = gradient(ps,pt,a,b,c,d,det)
        wt = w*abs(det)
        if observe is not None:
            Z = ref.fields(p,geo[0])[1]
            observe(R,Z,wt,fr+rr,fz+rz,rr,rz)
        sums += [wt@(B@u-psi)**2,wt@psi**2,wt@((fr**2+fz**2)/R**2),
                 wt@((rr**2+rz**2)/R**2)]
    if not np.isfinite(sums).all() or min(sums[[1,3]])<=0 or residual>1e-8:
        raise ValueError('degenerate reference or failed discrete solve')
    if diagnose is not None:
        diagnose(ref,axis,ids,free,wall,A,rhs,prescribed,psi_b,tables,assembly_order or 2*n+2)
    return dict(nodes_per_axis=n,free_dofs=len(free),elements=ref.count,setup_s=setup,
                factor_and_solve_s=solve,relative_linear_residual=residual,
                flux_relative_L2=np.sqrt(sums[0]/sums[1]),Bp_relative_L2=np.sqrt(sums[2]/sums[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--degrees',type=int,nargs='+',default=[2,4,6])
    args = parser.parse_args()
    ref = Restart(args.restart)
    with args.restart.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    report = dict(protocol='fixed-current model-303 replay',reference=str(args.restart),
                  sha256=digest,quadrature_check_order=18,runs=[run(ref,d) for d in args.degrees])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)

if __name__=='__main__':
    main()
