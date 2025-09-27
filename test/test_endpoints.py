import math
import numpy as np
import numpy.testing as npt
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from bspf.bspf1d import Grid1D, BSplineBasis1D, EndpointOps1D  # ← replace with your actual import path


def _open_uniform_knots(p: int, n_basis: int, a: float, b: float) -> np.ndarray:
    """Clamped/open-uniform knot vector on [a,b] with n_basis basis functions of degree p."""
    if n_basis <= p:
        raise ValueError("n_basis must exceed degree.")
    n_knots = n_basis + p + 1
    n_interior = n_knots - 2 * (p + 1)
    if n_interior <= 0:
        return np.array([a] * (p + 1) + [b] * (p + 1), dtype=np.float64)
    interior = np.linspace(a, b, n_interior + 2)[1:-1]
    return np.array([a] * (p + 1) + list(interior) + [b] * (p + 1), dtype=np.float64)


def test_endpointops1d_correctness():
    # --- basis & grid setup ---
    p = 3
    a, b = -1.0, 2.0
    n_basis = 8
    knots = _open_uniform_knots(p, n_basis, a, b)
    x = np.linspace(a, b, 401)
    grid = Grid1D(x)
    basis = BSplineBasis1D(p, knots, grid)

    # Choose endpoint operator settings
    order = 3        # we want up to 2nd derivative constraints (0,1,2)
    num_bd = 5       # number of boundary samples used to reconstruct endpoint derivs
    end = EndpointOps1D(basis, order=order, num_bd=num_bd)

    # ---------- 1) C maps spline coefficients to exact endpoint derivatives ----------
    # Random coefficients define a spline f(x) = sum_i c_i N_i(x)
    rng = np.random.default_rng(2024)
    c = rng.standard_normal(n_basis)

    S = SciPyBSpline(knots, c, p)
    f_left = np.array([S.derivative(k)(a) for k in range(order)], dtype=np.float64)
    f_right = np.array([S.derivative(k)(b) for k in range(order)], dtype=np.float64)
    CR = end.C @ c  # shape (2*order,)

    npt.assert_allclose(CR[:order], f_left, atol=5e-12, rtol=0)
    npt.assert_allclose(CR[order:], f_right, atol=5e-12, rtol=0)

    # ---------- 2) BND maps *samples* to endpoint derivatives exactly for polynomials ----------
    # For any polynomial of degree < num_bd, BND @ samples must yield the exact
    # left/right derivatives (0..order-1) at a and b.
    # We'll test several random polynomials of degree num_bd-1.
    for _ in range(4):
        # Random polynomial of degree <= num_bd-1 in the global coordinate x
        coeffs = rng.standard_normal(num_bd)
        # numpy.polyval expects highest degree first
        poly = np.poly1d(coeffs)

        # Samples on the grid
        f_samples = poly(x)

        # Apply BND to samples → endpoint derivative vector
        y = end.BND @ f_samples  # shape (2*order,)

        # Ground-truth derivatives at endpoints
        truths_left = []
        truths_right = []
        P = poly
        for k in range(order):
            truths_left.append(P(a))
            truths_right.append(P(b))
            P = P.deriv()  # next derivative
        truths_left = np.array(truths_left, dtype=np.float64)
        truths_right = np.array(truths_right, dtype=np.float64)

        # Exact for polynomials up to degree num_bd-1
        npt.assert_allclose(y[:order], truths_left, atol=1e-8, rtol=0)
        npt.assert_allclose(y[order:], truths_right, atol=1e-8, rtol=0)

    # ---------- 3) Algebraic sanity of X_left / X_right (solve correctness) ----------
    # A_left * X_left^T == E_left and A_right * X_right^T == E_right by construction
    i, j = np.meshgrid(np.arange(num_bd), np.arange(num_bd), indexing="ij")
    fact = np.array([math.factorial(k) for k in range(num_bd)], dtype=np.float64)
    A_left = (j**i) / fact[:, None]
    A_right = np.flip(A_left * ((-1.0) ** i), axis=(0, 1))
    E_left = np.eye(num_bd, dtype=np.float64)[:order, :].T
    idx = np.arange(num_bd - 1, num_bd - order - 1, -1)
    E_right = np.eye(num_bd, dtype=np.float64)[idx, :].T

    # Check residuals are near machine precision
    npt.assert_allclose(A_left @ end.X_left.T, E_left, atol=1e-8, rtol=0)
    npt.assert_allclose(A_right @ end.X_right.T, E_right, atol=1e-8, rtol=0)

    # ---------- 4) Dimensional checks (lightweight) ----------
    n_basis_B0, n_pts = basis.B0.shape
    assert end.C.shape == (2 * order, n_basis_B0)
    assert end.BND.shape == (2 * order, n_pts)
    assert end.X_left.shape == (order, num_bd)
    assert end.X_right.shape == (order, num_bd)
