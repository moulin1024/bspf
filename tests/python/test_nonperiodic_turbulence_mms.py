"""Analytic checks for the 3D curved-domain MMS; no CUDA dependency."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('turbulence3d_mms',Path(__file__).resolve().parents[2]/'bench/turbulence_3d/mms.py')
mms=importlib.util.module_from_spec(spec);spec.loader.exec_module(mms)


def test_exact_transverse_wave_and_manufactured_forcing(tmp_path):
    k=np.sqrt(2.)
    f=mms.PlaneWaveMMS([[k,0,0]],[[0,1,0]],[[0,0,1]],[.3],[.2],
                      center=(0,0,0),u_offset=(1,0,0),viscosity=.03)
    x=np.random.default_rng(42).random((17,3));q=k*x[:,0];v=f.evaluate(x,chunk=3)
    expected=np.column_stack((np.ones(len(x)),np.cos(q),np.sin(q)))
    np.testing.assert_allclose(v['velocity'],expected,atol=1e-15)
    grad=np.zeros((len(x),3,3));grad[:,1,0]=-k*np.sin(q);grad[:,2,0]=k*np.cos(q)
    np.testing.assert_allclose(v['velocity_gradient'],grad,atol=1e-15)
    np.testing.assert_allclose(np.trace(v['velocity_gradient'],axis1=1,axis2=2),0,atol=1e-15)
    poisson=k*k*(expected-[1,0,0]);gp=np.zeros((len(x),3));gp[:,0]=k*(-.3*np.sin(q)+.2*np.cos(q))
    np.testing.assert_allclose(v['vector_poisson_forcing'],poisson,atol=1e-15)
    np.testing.assert_allclose(v['navier_stokes_forcing'],grad[:,:,0]+gp+.03*poisson,atol=1e-15)
    f.save(tmp_path/'modes.npz')
    np.testing.assert_array_equal(mms.PlaneWaveMMS.load(tmp_path/'modes.npz').evaluate(x)['velocity'],v['velocity'])
    assert np.linalg.norm(f.evaluate([[0,0,0],[1,0,0]])['velocity'][1]-f.evaluate([[0,0,0],[1,0,0]])['velocity'][0])>.5


def test_curved_map_jacobian_and_six_physical_normals():
    domain=mms.CurvedBox();s=np.random.default_rng(8).random((40,3));x,jac,det=domain.map(s)
    assert det.min()>0
    for axis in range(3):
        step=np.eye(3)[axis]*1e-6
        fd=(domain.map(s+step)[0]-domain.map(s-step)[0])/(2e-6)
        np.testing.assert_allclose(fd,jac[:,:,axis],atol=2e-10,rtol=2e-10)
        for side in (0,1):
            points,normal=domain.face(axis,side,np.array([[.5,.5],[0,0],[1,1]]))
            assert (2*side-1)*normal[0,axis]>0
            assert abs(points[0,axis]-side)>.01
            assert points[1,axis]==side and points[2,axis]==side
    with pytest.raises(ValueError):mms.CurvedBox((.1,0,0))


def test_curved_boundary_flux_for_divergence_free_wave():
    domain=mms.CurvedBox()
    k=np.array([1.2,2.3,-3.4]);a=np.cross(k,[1.,0.,0.]);b=np.cross(k,[0.,1.,0.])
    f=mms.PlaneWaveMMS(k[None],a[None],b[None],[0.],[0.])
    z,w=np.polynomial.legendre.leggauss(24);z=(z+1)/2;w=w/2
    uv=np.stack(np.meshgrid(z,z,indexing='ij'),axis=-1);weight=w[:,None]*w[None,:]
    fluxes=[]
    for axis in range(3):
        for side in (0,1):
            x,normal=domain.face(axis,side,uv)
            fluxes.append(np.sum(weight*np.sum(f.evaluate(x,derivatives=False)['velocity']*normal,axis=-1)))
    assert abs(sum(fluxes))<1e-12
    assert max(abs(np.array(fluxes)))>.01
