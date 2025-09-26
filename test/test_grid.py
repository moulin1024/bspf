import unittest
import numpy as np
from bspf.grid import Grid1D

class TestGrid1D(unittest.TestCase):
    def test_uniform_grid_core_properties(self):
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        g = Grid1D(x)

        # basic geometry
        self.assertEqual(g.n, 5)
        self.assertAlmostEqual(g.a, 0.0)
        self.assertAlmostEqual(g.b, 1.0)
        self.assertAlmostEqual(g.dx, 0.25)

        # trapezoid weights
        expected_trap = np.full_like(x, g.dx)
        expected_trap[0] = expected_trap[-1] = g.dx / 2.0
        np.testing.assert_allclose(g.trap, expected_trap)

        # rFFT angular frequencies
        expected_omega = 2.0 * np.pi * np.fft.rfftfreq(g.n, d=g.dx)
        np.testing.assert_allclose(g.omega, expected_omega)
        self.assertEqual(len(g.omega), g.n // 2 + 1)

if __name__ == "__main__":
    unittest.main()
