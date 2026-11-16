"""CPU checks for forcing interpolation, independent of CUDA bindings."""
import importlib.util
from pathlib import Path
import unittest
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('nodal_forcing', ROOT/'bench/turbulence_3d/bspf_poisson/nodal_forcing.py')
forcing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forcing)


class NodalForcingTests(unittest.TestCase):
    def test_full_bspf_transfer_reproduces_enriched_trial_field(self):
        sys.path.insert(0, str(ROOT/'bench/curved_poisson/src'))
        from optimized_bvp import HybridAxis
        axis = HybridAxis(2, degree=8, nodes='symmetric_qr')
        q = (np.polynomial.legendre.leggauss(23)[0]+1)/2
        b = axis.evaluate(q, 0)[0]
        def field(nodes):
            z, y, x = np.meshgrid(nodes, nodes, nodes, indexing='ij')
            return np.sin(2*np.pi*x/1.5)*np.cos(4*np.pi*y/1.5)+z**8
        np.testing.assert_allclose(forcing.interpolate_forcing(field(axis.x), b), field(q), rtol=2e-12, atol=2e-12)

    def test_nonuniform_polynomial_and_axis_order(self):
        x = (1-np.cos(np.linspace(0, np.pi, 13)))/2
        q = (np.polynomial.legendre.leggauss(20)[0]+1)/2
        a = forcing.spline_matrix(x, q)
        z, y, xx = np.meshgrid(x, x, x, indexing='ij')
        values = 2+xx**5+3*y**4-7*z**3+xx*y*z
        z, y, xx = np.meshgrid(q, q, q, indexing='ij')
        exact = 2+xx**5+3*y**4-7*z**3+xx*y*z
        np.testing.assert_allclose(forcing.interpolate_forcing(values, a), exact, rtol=2e-13, atol=2e-13)

    def test_gpu_slab_order_matches_tensor_interpolation(self):
        x = np.linspace(0, 1, 11)
        q = (np.polynomial.legendre.leggauss(17)[0]+1)/2
        a = forcing.spline_matrix(x, q)
        values = np.random.default_rng(19).normal(size=(11,)*3)
        xy = np.einsum('ai,bj,kji->kba', a, a, values, optimize=True)
        slabs = np.stack([np.einsum('k,kji->ji', row, xy, optimize=True) for row in a])
        np.testing.assert_allclose(slabs, forcing.interpolate_forcing(values, a), rtol=2e-13, atol=2e-13)

    def test_nodal_interpolation_and_validation(self):
        x = np.linspace(0, 1, 11)
        np.testing.assert_allclose(forcing.spline_matrix(x, x), np.eye(11), atol=1e-14)
        for degree in [0, 2, 11]:
            with self.assertRaises(ValueError):
                forcing.spline_matrix(x, x, degree)
        with self.assertRaises(ValueError):
            forcing.spline_matrix(x, [-.1, .5])


if __name__ == '__main__':
    unittest.main()
