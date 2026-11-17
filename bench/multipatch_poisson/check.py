"""Two curved patches, shared trace DOFs, variable-coefficient Poisson MMS.

Uses the same HybridAxis as the 3D BSPF benchmark. This is a small dense
assembly check, not a TCV mesh or a scalable solver.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve, eigvalsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'curved_poisson/src'))
from optimized_bvp import HybridAxis


def geometry(s, t):
    return 1+s+.07*np.sin(np.pi*t), t+.04*np.sin(np.pi*s)


def exact(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


def run(n, q):
    axis = HybridAxis((n-9)//2, degree=8, nodes='symmetric_qr')
    z, w = np.polynomial.legendre.leggauss(q)
    z, w = (z+1)/2, w/2
    B, D = axis.evaluate(z, 1)
    P = np.kron(B, B)
    Ps, Pt = 2*np.kron(D, B), np.kron(B, D)
    size = (2*n-1)*n
    A, rhs = np.zeros((size, size)), np.zeros(size)
    S = np.r_[axis.x/2, .5+axis.x[1:]/2]
    X, Y = geometry(*np.meshgrid(S, axis.x, indexing='ij'))
    expected = exact(X, Y).ravel()
    min_det = 1.
    for patch in range(2):
        s, t = np.meshgrid((patch+z)/2, z, indexing='ij')
        s, t = s.ravel(), t.ravel()
        x, y = geometry(s, t)
        xt, ys = .07*np.pi*np.cos(np.pi*t), .04*np.pi*np.cos(np.pi*s)
        det = 1-xt*ys
        min_det = min(min_det, float(det.min()/2))
        Px = (Ps-ys[:, None]*Pt)/det[:, None]
        Py = (Pt-xt[:, None]*Ps)/det[:, None]
        weight = np.outer(w, w).ravel()*det/2
        c = 1+.2*x+.1*y
        u = exact(x, y)
        ux = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        uy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        f = 2*np.pi**2*c*u-.2*ux-.1*uy
        local = Px.T@((weight*c)[:, None]*Px)+Py.T@((weight*c)[:, None]*Py)
        ids = np.arange(size).reshape(2*n-1, n)[patch*(n-1):patch*(n-1)+n].ravel()
        A[np.ix_(ids, ids)] += local
        rhs[ids] += P.T@(weight*f)
    wall = np.zeros((2*n-1, n), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    free, fixed = np.flatnonzero(~wall.ravel()), np.flatnonzero(wall.ravel())
    Af = A[np.ix_(free, free)]
    b = rhs[free]-A[np.ix_(free, fixed)]@expected[fixed]
    sol = expected.copy()
    sol[free] = solve(Af, b, assume_a='pos')
    return dict(n=n, q=q, dofs=size, max_error=float(abs(sol-expected).max()),
                relative_residual=float(np.linalg.norm(Af@sol[free]-b)/np.linalg.norm(b)),
                symmetry_error=float(np.linalg.norm(Af-Af.T)/np.linalg.norm(Af)),
                smallest_eigenvalue=float(eigvalsh(Af, subset_by_index=[0, 0])[0]),
                min_patch_jacobian=min_det)


if __name__ == '__main__':
    results = [run(n, q) for n in (11, 17, 25) for q in (2*n+8, 3*n+8)]
    print(json.dumps(results, indent=2))
    assert all(r['relative_residual'] < 1e-10 and r['smallest_eigenvalue'] > 0 for r in results)
    assert results[-1]['max_error'] < 1e-8
