"""Compact M_J f for the current curved box, matching the selected Gauss rule.

The determinant is an even quartic in each centered reference coordinate.
Three weighted 1D mass matrices therefore suffice per axis. This is a load
operator only; no stiffness approximation or global dense matrix is formed.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


class NodalMass:
    def __init__(self, basis, points, weights, domain, xp=np):
        b, x, w = np.asarray(basis), np.asarray(points), np.asarray(weights)
        if (b.ndim != 2 or x.shape != (len(b),) or w.shape != x.shape or np.any(w <= 0)
                or not all(np.isfinite(v).all() for v in (b, x, w))):
            raise ValueError('Expected basis(q,n), points(q), and positive weights(q)')
        self.n, self.xp = b.shape[1], xp
        # Five-point Gauss exactly integrates det(J)*P_l for l=0,2,4.
        t, wg = leggauss(5)
        s = (t+1)/2
        reference = np.stack(np.meshgrid(s, s, s, indexing='ij'), axis=-1)
        det = domain.map(reference)[2]
        p = legvander(t, 4)[:, [0, 2, 4]]
        projection = (p * (wg/2)[:, None] * np.array([1, 5, 9])).T
        coefficient = np.einsum('ai,bj,ck,ijk->abc', projection, projection, projection, det, optimize=True)
        # Reject geometry outside this exact even-quartic representation.
        probe = np.random.default_rng(41).random((128, 3))
        v = [legvander(2*probe[:, j]-1, 4)[:, [0, 2, 4]] for j in range(3)]
        fitted = np.einsum('abc,pa,pb,pc->p', coefficient, *v, optimize=True)
        self.determinant_error = float(np.max(np.abs(fitted-domain.map(probe)[2])))
        if self.determinant_error > 1e-12:
            raise ValueError('Geometry determinant is not an even tensor quartic')
        p = legvander(2*x-1, 4)[:, [0, 2, 4]]
        mass = np.stack([b.T @ ((w*p[:, j])[:, None]*b) for j in range(3)])
        mass = .5*(mass+mass.transpose(0, 2, 1))
        middle = np.einsum('rms,mij->rsij', coefficient, mass, optimize=True)
        n = self.n
        # Apply in x, y, z order to a z,y,x-ordered nodal field.
        self.x_matrix = xp.asarray(np.ascontiguousarray(mass.reshape(3*n, n)))
        self.y_matrix = xp.asarray(np.ascontiguousarray(middle.transpose(1, 2, 0, 3).reshape(3*n, 3*n)))
        self.z_matrix = xp.asarray(np.ascontiguousarray(mass.transpose(1, 0, 2).reshape(n, 3*n)))
        self.bytes = sum(a.nbytes for a in (self.x_matrix, self.y_matrix, self.z_matrix))

    def __call__(self, nodal):
        n = self.n
        f = self.xp.asarray(nodal, dtype=self.xp.float64)
        if f.shape != (n, n, n):
            raise ValueError('Expected nodal forcing with shape (n,n,n)')
        v = self.x_matrix @ f.reshape(n*n, n).T
        v = v.reshape(3, n, n, n).transpose(0, 3, 2, 1).reshape(3*n, n*n)
        v = self.y_matrix @ v
        v = v.reshape(3, n, n, n).transpose(0, 2, 1, 3).reshape(3*n, n*n)
        return (self.z_matrix @ v).reshape(n, n, n)
