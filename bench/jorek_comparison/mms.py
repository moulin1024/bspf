"""Compare the compiled JOREK GS element and BSPF on a manufactured equilibrium."""
import argparse
import os
import ctypes
import json
import subprocess
import tempfile
from time import perf_counter
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from campaign import ROOT, Restart, gradient, rule, run

HERE = Path(__file__).resolve().parent
JOREK = Path(os.environ.get('JOREK_SOURCE', ROOT / 'third_party/JOREK')).resolve()
ASSEMBLY_ORDER = 16
SCORE_ORDER = 24


def manufactured(wave):
    def exact(R, Z):
        c = np.cos(wave * Z)
        psi = (1 + R**2) * c
        return psi, 2 * R * c, -(1 + R**2) * wave * np.sin(wave * Z), -wave**2 * psi
    return exact


def compile_kernel(build, order=ASSEMBLY_ORDER):
    lib = build / f'mms{order}.so'
    subprocess.run(['gfortran', '-cpp', f'-DMMS_GAUSS_ORDER={order}', '-O2', '-shared', '-fPIC', '-ffree-line-length-none',
                    '-o', lib, HERE/'mms_context.f90', JOREK/'elements/mod_basisfunctions.f90',
                    JOREK/'models/element_matrix_GS.f90', HERE/'mms_kernel.f90'],
                   cwd=build, check=True)
    kernel = ctypes.CDLL(str(lib))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='F_CONTIGUOUS')
    kernel.init_mms.argtypes = [array, array, ctypes.c_double]
    kernel.mms_element.argtypes = [array] * 4
    kernel.init_mms.restype = kernel.mms_element.restype = None
    kernel.set_tilted_mms.argtypes = [ctypes.c_int]
    kernel.set_tilted_mms.restype = None
    kernel.gauss_order = order
    return kernel


def native(path, ref, wave, kernel, *, timings=False, tilted=False, observe=None, capture=None):
    start = perf_counter()
    with h5py.File(path) as f:
        x = f['x'][:].reshape(2, 4, -1)
        ix = f['index'][:] - 1
        vertex = f['vertex'][:] - 1
        sizes = f['size'][:]
        bc = f['boundary'][:]
    exact = manufactured(wave)
    kernel.set_tilted_mms(int(tilted))
    if tilted:
        from tilted_mms import manufactured_tilted
        exact = manufactured_tilted(wave)
    psi, gr, gz, _ = exact(*x[:, 0])
    # Follow the restart's boundary tags, including shared derivative indices.
    tags = [bc != 0, np.isin(bc, [1,3,4,9,11,12,19,20,21]),
            np.isin(bc, [2,3,5,9,15,19,20,21])]
    fixed = np.concatenate([ix[i, mask] for i, mask in enumerate(tags)])
    prescribed = np.concatenate([v[mask] for mask, v in zip(tags,
        [psi, gr*x[0,1]+gz*x[1,1], gr*x[0,2]+gz*x[1,2]])])
    u = np.zeros(ix.max() + 1)
    u[fixed] = prescribed
    if not np.allclose(u[fixed], prescribed, rtol=1e-10, atol=1e-12):
        raise ValueError('inconsistent analytic values on a shared boundary index')
    g = ix[:,vertex].transpose(2,1,0).reshape(-1,16)
    blocks = np.empty((ref.count,16,16))
    rhs = np.zeros_like(u)
    q,w = np.polynomial.legendre.leggauss(kernel.gauss_order)
    kernel.init_mms((q+1)/2, w/2, wave)
    block = np.empty((16,16), order='F')
    load = np.empty(16)
    for p, idx in enumerate(g):
        kernel.mms_element(np.asfortranarray(x[:,:,vertex[:,p]]),
                           np.asfortranarray(sizes[:,:,p],dtype=float), block, load)
        blocks[p] = block
        np.add.at(rhs,idx,load)
    rows = np.broadcast_to(g[:,:,None],blocks.shape)
    cols = np.broadcast_to(g[:,None,:],blocks.shape)
    A = coo_matrix((blocks.ravel(),(rows.ravel(),cols.ravel())),shape=(len(u),len(u))).tocsr()
    free = np.setdiff1d(np.flatnonzero(A.diagonal()),fixed)
    load = (rhs-A@u)[free]
    assembled = perf_counter()
    u[free] = spsolve(A[free][:,free],load)
    solved = perf_counter()
    if capture is not None:
        capture(u[g].reshape(-1,4,4).transpose(0,2,1)*sizes.transpose(2,0,1))
    residual = np.linalg.norm((rhs-A@u)[free])/np.linalg.norm(load)
    q,w = np.polynomial.legendre.leggauss(SCORE_ORDER)
    tab = rule((q+1)/2)
    weights = np.outer(w,w).ravel()/4
    sums = np.zeros(4)
    for p, idx in enumerate(g):
        coef = (u[idx].reshape(4,4).T*sizes[:,:,p]).ravel()
        val, ds, dt = [coef@table for table in tab]
        R,Z = ref.fields(p,tab[0])[:2]
        a,c = ref.fields(p,tab[1])[:2]
        b,d = ref.fields(p,tab[2])[:2]
        det = a*d-b*c
        if np.any(det <= 0):
            raise ValueError('native GS kernel needs positively oriented elements')
        dr,dz = gradient(ds,dt,a,b,c,d,det)
        val0,dr0,dz0,_ = exact(R,Z)
        wt = weights*det
        if observe is not None:
            observe(R,Z,wt,dr,dz,dr0,dz0)
        sums += [wt@(val-val0)**2, wt@val0**2, (wt/R**2)@((dr-dr0)**2+(dz-dz0)**2),
                 (wt/R**2)@(dr0**2+dz0**2)]
    if not np.isfinite(residual) or residual > 1e-8:
        raise ValueError(f'native free residual {residual}')
    times = dict(setup_s=assembled-start, factor_and_solve_s=solved-assembled) if timings else {}
    return dict(**times, free_dofs=len(free), relative_linear_residual=residual,
                flux_relative_L2=np.sqrt(sums[0]/sums[1]), Bp_relative_L2=np.sqrt(sums[2]/sums[3]))


def compare(path, kernel, wave, degrees):
    ref = Restart(path)
    rows = {'JOREK bicubic': native(path, ref, wave, kernel)}
    for degree in degrees:
        row = run(ref,degree,exact=manufactured(wave),assembly_order=ASSEMBLY_ORDER,
                  check_order=SCORE_ORDER)
        rows[f'BSPF {row["nodes_per_axis"]} nodes/axis'] = {
            k:v for k,v in row.items() if k in rows['JOREK bicubic']}
    return rows


def rectangular_mesh(path, n):
    """Uniform affine cells on the fixed physical rectangle [1,2] x [-1,1]."""
    i,j = np.indices((n+1,n+1))
    x = np.zeros((2,4,1,i.size))
    x[:,0,0] = [1+i.ravel()/n, 2*j.ravel()/n-1]
    x[0,1,0] = 1/(3*n)
    x[1,2,0] = 2/(3*n)
    corners = (i[:-1,:-1]*(n+1)+j[:-1,:-1]).ravel()
    vertex = corners + np.array([[0],[n+1],[n+2],[1]])
    signs = np.array([[1,1,1,1],[1,-1,-1,1],[1,1,-1,-1],[1,-1,1,-1]])
    bc = ((j==0)|(j==n)).astype(int) + 2*((i==0)|(i==n))
    with h5py.File(path,'w') as f:
        f['vertex'] = vertex + 1
        f['size'] = np.broadcast_to(signs[:,:,None],(4,4,n*n))
        f['x'] = x
        f['values'] = np.zeros((3,4,1,i.size))
        f['boundary'] = bc.ravel()
        f['index'] = np.arange(1,4*i.size+1).reshape(i.size,4).T


def resolution_study(build, kernel, wave):
    rows = []
    for factor in (2,4,8):
        # A bicubic Hermite mesh has about 4n² DOFs, a four-node BSPF mesh about 9n².
        mesh = build/'rectangle.h5'
        n = 3*factor
        rectangular_mesh(mesh,n)
        row = native(mesh,Restart(mesh),wave,kernel)
        rows.append(dict(method='JOREK bicubic', cells_per_axis=n, **row))
        for degree,n in [(1,2*factor),(4,factor)]:
            rectangular_mesh(mesh,n)
            row = run(Restart(mesh),degree,exact=manufactured(wave),assembly_order=ASSEMBLY_ORDER,
                      check_order=SCORE_ORDER)
            rows.append(dict(method=f'BSPF {row["nodes_per_axis"]} nodes/axis', cells_per_axis=n,
                             **{key: row[key] for key in ('free_dofs','flux_relative_L2',
                                                          'Bp_relative_L2','relative_linear_residual')}))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        kernel = compile_kernel(Path(tmp))
        report = dict(assembly_order=ASSEMBLY_ORDER, score_order=SCORE_ORDER, wave=3.0,
                      mastu=compare(args.restart,kernel,3.0,[1,2,4]),
                      rectangle=resolution_study(Path(tmp),kernel,3.0))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
