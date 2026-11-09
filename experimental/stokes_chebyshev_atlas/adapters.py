"""Exact Goursat derivatives for the existing AAA/lightning Arnoldi solver.

No geometry is imported and no Stokes solve is launched. This adapter consumes
an already fitted solution (groups and f coefficients), including log modes.
"""
from math import comb, factorial
import numpy as np
from .atlas import StokesGuide


def from_arnoldi_solver(solver, scale=1., log_terms=None, chunk_size=1024):
    """Adapt RationalStokes/ObstacleStokes, or an equivalent Arnoldi solution.

    Each group exposes polynomial, scale/poles, and Arnoldi recurrence H.
    Coefficient order is the first complete group followed by other groups
    without their repeated constant. log_terms is [(center, complex_a), ...]
    for f(z) += a*log(z-center); supply [] to explicitly omit log contributions.
    The existing single-hole solver's log_coeff/ laurent_center are recognized.
    """
    groups = solver.groups
    f = np.array(solver.f, dtype=complex, copy=True)
    expected = 1 + sum(len(g.H) for g in groups)
    if not groups or f.shape != (expected,):
        raise ValueError("Goursat coefficient count does not match Arnoldi groups")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if log_terms is None:
        log_terms = []
        if hasattr(solver, 'log_coeff'):
            lc = np.asarray(solver.log_coeff)
            if lc.shape != (3,) or not hasattr(solver, 'laurent_center'):
                raise ValueError("Supply explicit log_terms for this solver's logarithmic basis")
            log_terms = [(solver.laurent_center, lc[0]+1j*lc[1])]
    log_terms = tuple(log_terms)

    def jet(z):
        shape = np.shape(z)
        points = np.asarray(z, dtype=complex).ravel()
        out = np.zeros((2, len(points)), dtype=complex)
        for start in range(0, len(points), chunk_size):
            zz = points[start:start+chunk_size]
            offset = 0
            for index, group in enumerate(groups):
                count = len(group.H)
                jets = np.zeros((4, len(zz), count+1), dtype=complex)
                jets[0, :, 0] = 1
                for k, h in enumerate(group.H):
                    if group.polynomial:
                        factor = [zz/group.scale, np.full_like(zz, 1/group.scale),
                                  np.zeros_like(zz), np.zeros_like(zz)]
                    else:
                        r = 1/(zz-group.poles[k])
                        factor = [(-1)**d * factorial(d) * r**(d+1) for d in range(4)]
                    for d in range(4):
                        value = sum(comb(d, j)*factor[j]*jets[d-j, :, k] for j in range(d+1))
                        jets[d, :, k+1] = (value-jets[d, :, :k+1]@h[:-1])/h[-1]
                first = int(index > 0)
                width = count+1-first
                out[:, start:start+len(zz)] += np.einsum('dpk,k->dp', jets[2:, :, first:], f[offset:offset+width])
                offset += width
            for center, coefficient in log_terms:
                r = 1/(zz-center)
                out[0, start:start+len(zz)] -= coefficient*r*r
                out[1, start:start+len(zz)] += 2*coefficient*r**3
        return (4*out).reshape((2,)+shape)

    # Keep one bounded cache entry so dq does not repeat the recurrence. Compare
    # values too: callers may reuse and mutate a physical-coordinate array.
    cache = {}
    def cached(z, component):
        if 'input' not in cache or not np.array_equal(cache['input'], np.asarray(z)):
            cache['input'] = np.array(z, copy=True)
            cache['jet'] = jet(z)
        return cache['jet'][component]
    return StokesGuide(lambda z: cached(z, 0), lambda z: cached(z, 1), scale)
