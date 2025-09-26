import unittest
import numpy as np

from bspf.grid import Grid1D
from bspf.correction import ResidualCorrection


class TestResidualCorrection(unittest.TestCase):
    def setUp(self):
        # Uniform grid on [0, 1]
        self.x = np.linspace(0.0, 1.0, 257, dtype=np.float64)
        self.grid = Grid1D(self.x)

    def test_none_returns_zeros(self):
        r = np.random.default_rng(0).standard_normal(self.grid.n).astype(np.float64)
        out = ResidualCorrection.none(r, self.grid.omega, kind="diff", order=1, n=self.grid.n, x=self.x)
        self.assertEqual(out.shape, (self.grid.n,))
        np.testing.assert_allclose(out, 0.0)

    def test_spectral_diff_matches_known_derivative(self):
        # Use a pure FFT mode: r[j] = sin(omega_m * (x[j] - x0))
        # Here omega_m = 2π m / (n*dx) which matches Grid1D.omega
        m = 5
        x0 = float(self.x[0])
        omega_m = self.grid.omega[m]

        r = np.sin(omega_m * (self.x - x0))
        expected = omega_m * np.cos(omega_m * (self.x - x0))

        out = ResidualCorrection.spectral(
            residual=r, omega=self.grid.omega, kind="diff", order=1, n=self.grid.n, x=self.x
        )
        np.testing.assert_allclose(out, expected, atol=1e-10, rtol=0)

        # Second derivative should be -omega_m^2 * sin(...)
        expected2 = -(omega_m**2) * r
        out2 = ResidualCorrection.spectral(
            residual=r, omega=self.grid.omega, kind="diff", order=2, n=self.grid.n, x=self.x
        )
        np.testing.assert_allclose(out2, expected2, atol=1e-9, rtol=0)


    def test_spectral_int_order1_constant(self):
        # Integrate a constant residual c: result should be c * (x - x0) with left anchor at 0
        c = 3.25
        r = np.full(self.grid.n, c, dtype=np.float64)
        out = ResidualCorrection.spectral(
            residual=r, omega=self.grid.omega, kind="int", order=1, n=self.grid.n, x=self.x
        )
        expected = c * (self.x - self.x[0])
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)

    def test_spectral_int_order2_constant(self):
        # Twice-integrate a constant residual c: q(x) = 0.5*c*(x-x0)*(x-x1) (zero at both ends)
        c = -1.7
        r = np.full(self.grid.n, c, dtype=np.float64)
        out = ResidualCorrection.spectral(
            residual=r, omega=self.grid.omega, kind="int", order=2, n=self.grid.n, x=self.x
        )
        x0, x1 = float(self.x[0]), float(self.x[-1])
        expected = 0.5 * c * (self.x - x0) * (self.x - x1)
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)

    def test_invalid_kind_and_order_raise(self):
        r = np.zeros(self.grid.n, dtype=np.float64)
        with self.assertRaises(ValueError):
            ResidualCorrection.spectral(r, self.grid.omega, kind="oops", order=1, n=self.grid.n, x=self.x)
        with self.assertRaises(ValueError):
            ResidualCorrection.spectral(r, self.grid.omega, kind="int", order=3, n=self.grid.n, x=self.x)


if __name__ == "__main__":
    unittest.main()
