"""Interpolate physical forcing samples on reference nodes to Gauss points."""
import numpy as np
from scipy.interpolate import make_interp_spline


def spline_matrix(nodes, points, degree=5):
    """Not-a-knot interpolating B-spline evaluation matrix on one axis.

    Interpolate f(X(s)), not the Jacobian-weighted load. Geometry and volume
    weights are still evaluated directly at the integration points.
    """
    nodes, points = np.asarray(nodes, dtype=float), np.asarray(points, dtype=float)
    if (nodes.ndim != 1 or points.ndim != 1 or not np.isfinite(nodes).all()
            or not np.isfinite(points).all() or np.any(np.diff(nodes) <= 0)):
        raise ValueError('Finite ordered 1D nodes and finite 1D evaluation points required')
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1 or degree % 2 != 1 or degree >= len(nodes):
        raise ValueError('Spline degree must be a positive odd integer smaller than node count')
    if np.any(points < nodes[0]) or np.any(points > nodes[-1]):
        raise ValueError('Forcing interpolation does not extrapolate')
    return make_interp_spline(nodes, np.eye(len(nodes)), k=degree, axis=0)(points)


def interpolate_forcing(values, interpolation, xp=np):
    """Tensor interpolation; input and output are z,y,x ordered."""
    return xp.einsum('ai,bj,ck,kji->cba', interpolation, interpolation,
                     interpolation, values, optimize=True)
