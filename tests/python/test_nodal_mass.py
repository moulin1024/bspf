"""Independent quadrature checks for the compact curved mass operator."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'bench/turbulence_3d/bspf_poisson'), str(ROOT/'bench/curved_poisson/src'), str(ROOT/'bench/turbulence_3d')]
from nodal_mass import NodalMass
from nodal_forcing import interpolate_forcing
from optimized_bvp import HybridAxis
from mms import CurvedBox


class NodalMassTests(unittest.TestCase):
    def test_full_random_load_matches_interpolate_then_integrate(self):
        for amplitudes in [(0, 0, 0), (.08, -.06, .07), (-.03, .04, .02)]:
            axis = HybridAxis(4, degree=8, nodes='symmetric_qr')
            x, w = np.polynomial.legendre.leggauss(29); x = (x+1)/2; w = w/2
            b = axis.evaluate(x, 0)[0]
            domain = CurvedBox(amplitudes)
            op = NodalMass(b, x, w, domain)
            z, y, xx = np.meshgrid(x, x, x, indexing='ij')
            det = domain.map(np.stack([xx, y, z], axis=-1))[2]
            weight = w[:, None, None]*w[None, :, None]*w[None, None, :]*det
            rng = np.random.default_rng(31)
            f = rng.normal(size=(axis.n,)*3)
            reference = interpolate_forcing(weight*interpolate_forcing(f, b), b.T)
            np.testing.assert_allclose(op(f), reference, rtol=1e-11, atol=1e-16)
            self.assertLess(np.linalg.norm(op(f)-reference)/np.linalg.norm(reference), 1e-12)
            g = rng.normal(size=f.shape)
            self.assertLess(abs(np.vdot(g, op(f))-np.vdot(op(g), f)), 1e-13)
            self.assertGreater(np.vdot(f, op(f)), 0)

    def test_constant_forcing_and_invalid_shape(self):
        axis = HybridAxis(1, degree=8, nodes='symmetric_qr')
        x, w = np.polynomial.legendre.leggauss(20); x=(x+1)/2; w=w/2
        b = axis.evaluate(x, 0)[0]
        op = NodalMass(b, x, w, CurvedBox())
        self.assertAlmostEqual(float(op(np.ones((axis.n,)*3)).sum()), .92, places=12)
        with self.assertRaises(ValueError):op(np.zeros((2, 2, 2)))


if __name__ == '__main__':
    unittest.main()
