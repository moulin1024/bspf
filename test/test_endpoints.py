import unittest
import numpy as np

from bspf.grid import Grid1D
from bspf.endpoints import EndpointOps1D

# Minimal dummy basis so we can test EndpointOps1D in isolation.
class _DummyBasis:
    def __init__(self, grid, n_basis: int = 1):
        self.grid = grid
        # shape: (n_basis, n_points)
        self.B0 = np.zeros((n_basis, grid.n), dtype=np.float64)

    # EndpointOps1D calls BkT(k).T to get (n_basis, n_points),
    # so we return (n_points, n_basis) filled with zeros.
    def BkT(self, k: int):
        return np.zeros((self.grid.n, self.B0.shape[0]), dtype=np.float64)


class TestEndpointOps1D(unittest.TestCase):
    def test_bnd_weights_reproduce_endpoint_derivatives_for_polynomials(self):
        """
        The finite-difference operator BND is constructed from a Vandermonde system
        and (by design) reproduces endpoint derivatives up to order-1 exactly for
        any polynomial of degree <= num_bd-1 on a uniform grid. This verifies that.
        """
        # Grid and polynomial setup
        n = 21
        x = np.linspace(0.0, 1.0, n, dtype=np.float64)
        grid = Grid1D(x)

        # Choose num_bd and order (order <= num_bd). We'll test 0th, 1st, 2nd derivatives.
        num_bd = 4
        order = 3

        # Build EndpointOps1D with a dummy basis (so C is irrelevant; we test BND)
        basis = _DummyBasis(grid)
        ep = EndpointOps1D(basis, order=order, num_bd=num_bd)

        # Polynomial of degree <= num_bd-1 (here 3). p(x) = ax^3 + bx^2 + cx + d
        a, b, c, d = 1.3, -0.7, 0.5, 2.0
        p = a * x**3 + b * x**2 + c * x + d

        # True endpoint derivatives at left (x=a0) and right (x=b0)
        a0, b0 = grid.a, grid.b

        def deriv_val(k, t):
            if k == 0:
                return a * t**3 + b * t**2 + c * t + d
            if k == 1:
                return 3 * a * t**2 + 2 * b * t + c
            if k == 2:
                return 6 * a * t + 2 * b
            raise ValueError("k>2 not supported in this test")

        true_left = np.array([deriv_val(k, a0) for k in range(order)], dtype=np.float64)
        true_right = np.array([deriv_val(k, b0) for k in range(order)], dtype=np.float64)

        # Apply BND to samples
        dY = ep.BND @ p  # shape (2*order,)

        # Compare left rows (0..order-1) and right rows (order..2*order-1)
        np.testing.assert_allclose(dY[:order], true_left, rtol=0, atol=1e-12)
        np.testing.assert_allclose(dY[order:], true_right, rtol=0, atol=1e-12)

        # Sanity: BND has expected shape and uses only num_bd samples per side
        self.assertEqual(ep.BND.shape, (2 * order, n))
        # nonzero structure check (left uses first num_bd, right uses last num_bd)
        self.assertTrue(np.all(ep.BND[:order, num_bd:] == 0.0))
        self.assertTrue(np.all(ep.BND[order:, : n - num_bd] == 0.0))


if __name__ == "__main__":
    unittest.main()
