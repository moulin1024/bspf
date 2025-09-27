import numpy as np
import numpy.testing as npt
import pytest

from bspf.bspf1d import BSplineBasis1D, Grid1D  # ← replace with your actual import
from scipy.interpolate import BSpline as SciPyBSpline


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


def test_bsplines_basis_correctness():
    # Problem setup
    p = 3
    a, b = 0.0, 1.0
    n_basis = 7  # interior knots exist
    knots = _open_uniform_knots(p, n_basis, a, b)

    # A moderately fine uniform grid to evaluate/collocate
    x = np.linspace(a, b, 401)
    grid = Grid1D(x)

    basis = BSplineBasis1D(p, knots, grid)

    # 1) Partition of unity (strict)
    #    Sum_i N_i(x) == 1 on [a,b]
    sumN = basis.B0.sum(axis=0)
    npt.assert_allclose(sumN, 1.0, atol=5e-13, rtol=0)

    # 2) Local support check:
    #    N_i(x) == 0 outside (t_i, t_{i+p+1})
    t = knots
    for i, s in enumerate(basis._splines):
        left, right = t[i], t[i + p + 1]
        mask_outside = (x < left) | (x > right)
        npt.assert_array_less(np.abs(s(x[mask_outside])), 1e-12 + 0*x[mask_outside])

    # 3) Cross-check against SciPy composition for *random coefficients*
    #    f(x) = sum_i c_i N_i(x) must equal SciPy BSpline(knots, c, p)(x)
    rng = np.random.default_rng(12345)
    c = rng.standard_normal(n_basis)
    f_mat = basis.BT0 @ c
    f_scipy = SciPyBSpline(knots, c, p)(x)
    npt.assert_allclose(f_mat, f_scipy, atol=5e-13, rtol=0)

    # 4) First derivative correctness:
    #    f'(x) from basis.BkT(1) vs SciPy derivative
    df_mat = basis.BkT(1) @ c
    df_scipy = SciPyBSpline(knots, c, p).derivative(1)(x)
    npt.assert_allclose(df_mat, df_scipy, atol=2e-11, rtol=0)

    # 5) Second derivative correctness:
    d2f_mat = basis.BkT(2) @ c
    d2f_scipy = SciPyBSpline(knots, c, p).derivative(2)(x)
    npt.assert_allclose(d2f_mat, d2f_scipy, atol=8e-10, rtol=0)

    # 6) Fundamental theorem (exact): ∫ f'(x) dx = f(b) - f(a)

    # Exact via basis endpoint values
    NB_a = basis.B0[:, 0]     # N_i(a)
    NB_b = basis.B0[:, -1]    # N_i(b)
    int_df_exact = float(np.dot(c, NB_b - NB_a))
    jump_f = float(f_mat[-1] - f_mat[0])
    assert int_df_exact == pytest.approx(jump_f, abs=1e-13)

    # Cross-check with SciPy using derivative THEN antiderivative
    S = SciPyBSpline(knots, c, p)
    Fprime = S.derivative(1).antiderivative(1)
    int_df_scipy = float(Fprime(b) - Fprime(a))
    assert int_df_scipy == pytest.approx(jump_f, abs=1e-13)

    # 7) Basis integrals exactly assemble spline integral:
    #    ∫ f = c · [∫ N_i]
    basis_int = basis.integrate_basis(a, b)
    exact_int = float(np.dot(c, basis_int))

    # Cross-check with SciPy antiderivative of f (exact up to FP)
    S = SciPyBSpline(knots, c, p)
    F = S.antiderivative(1)
    int_f_scipy = float(F(b) - F(a))
    assert exact_int == pytest.approx(int_f_scipy, abs=1e-13)

    # 8) Derivative of partition of unity is zero:
    #    sum_i N_i'(x) == 0
    sumN1 = (basis.BkT(1).T).sum(axis=0)  # (n_basis, n_pts) -> sum over i, remain (n_pts,)
    npt.assert_allclose(sumN1, 0.0, atol=2e-11, rtol=0)

    # 9) Cache correctness:
    #    Re-calling BkT(k) returns the *same* object (cached), and matches vectorized evaluator.
    B1T = basis.BkT(1)
    assert basis.BkT(1) is B1T
    vec1 = basis._evaluate_splines_vectorized(x, deriv_order=1)  # (n_basis, n_pts)
    npt.assert_allclose(B1T, vec1.T, atol=0, rtol=0)

    # 10) Endpoint behavior for clamped splines:
    #     f(a) equals value from SciPy; likewise at b.
    assert f_mat[0] == pytest.approx(f_scipy[0], abs=1e-13)
    assert f_mat[-1] == pytest.approx(f_scipy[-1], abs=1e-13)
