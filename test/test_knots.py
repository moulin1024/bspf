import unittest
import numpy as np

from bspf.knots import KnotGenerator
from bspf.grid import Grid1D


class TestKnotGenerator(unittest.TestCase):
    def test_generate_basic_properties(self):
        degree = 3
        a, b = 2.0, 5.0
        n_basis = 10  # must be > degree
        k = KnotGenerator.generate(
            degree=degree, domain=(a, b), n_basis=n_basis,
            use_clustering=False, clustering_factor=2.0
        )

        n_knots = n_basis + degree + 1
        self.assertEqual(k.shape, (n_knots,))
        # clamped ends repeated degree+1 times
        self.assertTrue(np.allclose(k[: degree + 1], a))
        self.assertTrue(np.allclose(k[-(degree + 1):], b))
        # non-decreasing
        self.assertTrue(np.all(np.diff(k) >= 0))

        # correct number of interior knots
        n_interior = n_knots - 2 * (degree + 1)
        interior = k[degree + 1 : n_knots - (degree + 1)]
        self.assertEqual(interior.size, n_interior)
        # interior are strictly within (a, b)
        self.assertTrue(np.all(interior > a))
        self.assertTrue(np.all(interior < b))

    def test_generate_no_interior_knots(self):
        degree = 2
        a, b = -1.0, 3.0
        n_basis = degree + 1  # yields zero interior knots
        k = KnotGenerator.generate(
            degree=degree, domain=(a, b), n_basis=n_basis,
            use_clustering=False, clustering_factor=1.0
        )
        # Expect exactly degree+1 copies of a, then degree+1 copies of b
        expected = np.concatenate([np.full(degree + 1, a), np.full(degree + 1, b)])
        np.testing.assert_allclose(k, expected)

    def test_clustering_brings_knots_toward_ends(self):
        degree = 3
        a, b = 0.0, 1.0
        n_basis = 12

        k_uniform = KnotGenerator.generate(
            degree=degree, domain=(a, b), n_basis=n_basis,
            use_clustering=False, clustering_factor=2.0
        )
        k_cluster = KnotGenerator.generate(
            degree=degree, domain=(a, b), n_basis=n_basis,
            use_clustering=True, clustering_factor=3.0
        )

        # First interior and last interior indices
        first_int = degree + 1
        last_int = k_uniform.size - degree - 2

        # Distance of first/last interior knots from the ends
        dL_uniform = k_uniform[first_int] - a
        dR_uniform = b - k_uniform[last_int]
        dL_cluster = k_cluster[first_int] - a
        dR_cluster = b - k_cluster[last_int]

        # Clustering should pull interior knots closer to the boundaries
        self.assertLess(dL_cluster, dL_uniform + 1e-15)
        self.assertLess(dR_cluster, dR_uniform + 1e-15)

    def test_resolve_explicit_knots_precedence_and_validation(self):
        degree = 3
        grid = Grid1D(np.linspace(0.0, 1.0, 9))
        explicit = np.array([0.0, 0.0, 0.0, 0.0,
                             0.3, 0.6,
                             1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        out = KnotGenerator.resolve(
            degree=degree, grid=grid,
            knots=explicit, n_basis=None, domain=None,
            use_clustering=False, clustering_factor=2.0
        )
        np.testing.assert_array_equal(out, explicit)

        # Non-1D knots should raise
        with self.assertRaises(ValueError):
            KnotGenerator.resolve(
                degree=degree, grid=grid,
                knots=np.zeros((2, 5), dtype=np.float64),  # not 1D
                n_basis=None, domain=None,
                use_clustering=False, clustering_factor=2.0
            )

    def test_resolve_defaults_for_n_basis_and_domain(self):
        degree = 2
        x = np.linspace(-2.0, 3.0, 17)
        grid = Grid1D(x)

        # n_basis defaults to 2*(degree+1)*2 = 4*(degree+1)
        expected_n_basis = 4 * (degree + 1)
        k = KnotGenerator.resolve(
            degree=degree, grid=grid,
            knots=None, n_basis=None, domain=None,
            use_clustering=False, clustering_factor=1.0
        )
        self.assertEqual(k.size, expected_n_basis + degree + 1)
        # Ends should match grid domain by default
        self.assertAlmostEqual(k[0], grid.a)
        self.assertAlmostEqual(k[-1], grid.b)

    def test_generate_rejects_too_few_basis(self):
        degree = 3
        with self.assertRaises(ValueError):
            KnotGenerator.generate(
                degree=degree, domain=(0.0, 1.0), n_basis=degree,  # n_basis <= degree
                use_clustering=False, clustering_factor=1.0
            )


if __name__ == "__main__":
    unittest.main()
