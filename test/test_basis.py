import unittest
import numpy as np

from bspf.grid import Grid1D
from bspf.knots import KnotGenerator
from bspf.basis import BSplineBasis1D
from scipy.interpolate import BSpline


class TestBSplineBasis1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Uniform grid on [0, 1]
        cls.x = np.linspace(0.0, 1.0, 129, dtype=np.float64)
        cls.grid = Grid1D(cls.x)
        # Clamped uniform knots with interior points
        cls.degree = 3
        cls.n_basis = 12
        cls.knots = KnotGenerator.generate(
            degree=cls.degree, domain=(cls.grid.a, cls.grid.b),
            n_basis=cls.n_basis, use_clustering=False, clustering_factor=2.0
        )
        cls.basis = BSplineBasis1D(degree=cls.degree, knots=cls.knots, grid=cls.grid)

    def test_shapes_and_transposes(self):
        B0 = self.basis.B0          # (n_basis, n_points)
        BT0 = self.basis.BT0        # (n_points, n_basis)
        self.assertEqual(B0.shape, (self.n_basis, self.grid.n))
        self.assertEqual(BT0.shape, (self.grid.n, self.n_basis))
        # BT0 is the transpose of B0
        np.testing.assert_allclose(B0.T, BT0)

        # BkT(0) should be exactly BT0 (and ideally the same object)
        BkT0 = self.basis.BkT(0)
        np.testing.assert_allclose(BkT0, BT0)

    def test_partition_of_unity(self):
        # Sum of basis functions at each x equals 1 on a clamped open knot vector
        sums = np.sum(self.basis.B0, axis=0)  # over basis
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-12, rtol=0)

    def test_derivative_matrices_match_scipy(self):
        # Random coefficients -> build a spline and compare its derivative
        rng = np.random.default_rng(42)
        c = rng.standard_normal(self.n_basis).astype(np.float64)

        # Our matrix-based derivative
        f_from_matrix = self.basis.BT0 @ c
        df_from_matrix = self.basis.BkT(1) @ c
        d2f_from_matrix = self.basis.BkT(2) @ c

        # SciPy evaluation of the same spline and its derivatives
        sp = BSpline(self.knots, c, self.degree)
        f_sp = sp(self.x)
        df_sp = sp.derivative(1)(self.x)
        d2f_sp = sp.derivative(2)(self.x)

        np.testing.assert_allclose(f_from_matrix, f_sp, atol=1e-12, rtol=0)
        np.testing.assert_allclose(df_from_matrix, df_sp, atol=5e-11, rtol=0)
        np.testing.assert_allclose(d2f_from_matrix, d2f_sp, atol=5e-10, rtol=0)

    def test_integrate_basis_linear_functionality(self):
        # For any coefficients c, integral of sum_i c_i * N_i == c · integrate_basis
        rng = np.random.default_rng(7)
        c = rng.standard_normal(self.n_basis).astype(np.float64)
        sp = BSpline(self.knots, c, self.degree)

        left, right = float(self.grid.a), float(self.grid.b)

        # Using helper
        ints_per_basis = self.basis.integrate_basis(left, right)
        integral_from_basis = float(np.dot(c, ints_per_basis))

        # Ground truth from SciPy BSpline
        integral_sp = float(sp.integrate(left, right))

        np.testing.assert_allclose(integral_from_basis, integral_sp, atol=1e-12, rtol=0)

    def test_derivative_cache_identity(self):
        # Repeated calls to BkT(k) should return the same cached ndarray object
        B1a = self.basis.BkT(1)
        B1b = self.basis.BkT(1)
        self.assertIs(B1a, B1b)

        B2a = self.basis.BkT(2)
        B2b = self.basis.BkT(2)
        self.assertIs(B2a, B2b)


if __name__ == "__main__":
    unittest.main()
