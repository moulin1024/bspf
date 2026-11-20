"""Check composite integration and an exactly representable GS solution on a rectangle."""
import tempfile
from pathlib import Path
import numpy as np
from mms import Restart,rectangular_mesh
from refined_global import RefinedGlobalBSPF, prolong_cardinal
from scipy.interpolate import BSpline


def check_prolongation():
    degree=5
    coarse=np.random.default_rng(42).normal(size=(11,14))
    fine=prolong_cardinal(coarse,(17,22),degree)
    kernel=BSpline.basis_element(np.arange(degree+2))
    def sample(t,count,spacing,derivative):
        u=t[:,None]/spacing - (np.arange(count)-degree)
        return np.where((u>=0)&(u<degree+1),kernel(u,nu=derivative)/spacing**derivative,0.)
    r=np.linspace(0,5.9,37)
    z=np.linspace(0,8.4,41)
    for dr,dz in ((0,0),(1,0),(0,1),(2,0),(0,2)):
        def field(c,h):
            return sample(r,c.shape[0],h,dr)@c@sample(z,c.shape[1],h,dz).T
        np.testing.assert_allclose(field(coarse,1.),field(fine,.5),atol=4e-14)


def check():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'rectangle.h5'
        rectangular_mesh(path,3)
        solver=RefinedGlobalBSPF(Restart(path),degree=(3,4),order=4)
        def flux(r,z):
            return r*r*z*z+r**3*z
        def source(r,z):
            return 2*r*r+3*r*z
        solver.pilot(source,flux)
        solver.refine(0.1)
        # An affine domain supplies independent exact volume, perimeter and moments.
        for wall in (False,True):
            measure=moment=oscillatory=0.
            frequency=55.
            for boundary,(r,z,w) in solver.quadrature(6):
                if boundary==wall:
                    measure+=w.sum()
                    moment+=w@r
                    oscillatory+=w@np.cos(frequency*r)
            if not wall:
                integral=2*(np.sin(2*frequency)-np.sin(frequency))/frequency
                np.testing.assert_allclose(oscillatory,integral,atol=1e-8)
            np.testing.assert_allclose([measure,moment],[6,9] if wall else [2,3],
                                       rtol=2e-12,atol=2e-12)
        r,z,w=solver.volume
        for mode in ('domain','local'):
            if mode=='local':
                solver.refine(0.1,support_padding=0.6)
            u,ur,uz=solver.evaluate(r,z)
            np.testing.assert_allclose(u,flux(r,z),rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(ur,2*r*z*z+3*r*r*z,rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(uz,2*r*r*z+r**3,rtol=1e-7,atol=1e-7)



if __name__=='__main__':
    check_prolongation()
    check()
