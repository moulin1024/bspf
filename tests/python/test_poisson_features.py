"""Independent derivative and regularity checks for localized scalar features."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
path=Path(__file__).resolve().parents[2]/'bench/turbulence_3d/poisson_features.py'
spec=importlib.util.spec_from_file_location('poisson_features_test',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def parameters():
    return dict(gaussian_centers=np.array([[.4,.5,.6]]),gaussian_sigmas=np.array([[.05,.08,.12]]),
        gaussian_amplitudes=np.array([.7]),tanh_centers=np.array([[.5,.5,.5]]),
        tanh_radii=np.array([.22]),tanh_widths=np.array([.015]),tanh_amplitudes=np.array([.8]))


def test_analytic_laplacian_with_independent_finite_differences():
    p=parameters();f=mod.PoissonFeatures(p)
    rng=np.random.default_rng(12);direction=rng.normal(size=(24,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    x=np.r_[p['gaussian_centers']+rng.normal(size=(24,3))*.05,
            p['tanh_centers']+direction*(.22+rng.uniform(-.025,.025,(24,1))),p['tanh_centers']]
    value,forcing=f.evaluate(x);h=2e-5;lap=np.zeros(len(x))
    for i in range(3):
        step=np.eye(3)[i]*h
        lap+=(-f.evaluate(x+2*step)[0]+16*f.evaluate(x+step)[0]-30*value+16*f.evaluate(x-step)[0]-f.evaluate(x-2*step)[0])/(12*h*h)
    np.testing.assert_allclose(-lap,forcing,rtol=2e-6,atol=1e-5)
    assert np.isfinite(forcing).all()


def test_gaussian_center_and_tanh_interface():
    p=parameters();f=mod.PoissonFeatures(p)
    g=f.evaluate(p['gaussian_centers'],components=True)
    assert g['gaussian'][0]==.7
    np.testing.assert_allclose(g['gaussian_forcing'],.7*np.sum(1/p['gaussian_sigmas']**2),rtol=1e-15)
    x=p['tanh_centers']+np.array([[.22,0,0]])
    t=f.evaluate(x,components=True)
    np.testing.assert_allclose(t['tanh'],.4,atol=2e-15)
    np.testing.assert_allclose(t['tanh_forcing'],3*.8/(2*.22*.015),rtol=1e-13)


def test_features_are_optional_and_roundtrip(tmp_path):
    empty=tmp_path/'empty.npz';np.savez(empty,unrelated=np.arange(2))
    f=mod.PoissonFeatures.load(empty);assert not f.active
    assert np.array_equal(f.evaluate(np.ones((4,3)))[0],np.zeros(4))
    path=tmp_path/'features.npz';np.savez(path,**parameters())
    a=mod.PoissonFeatures.load(path);b=mod.PoissonFeatures(parameters())
    x=np.random.default_rng(2).random((8,3))
    np.testing.assert_array_equal(a.evaluate(x),b.evaluate(x))


def test_invalid_width_rejected():
    p=parameters();p['tanh_widths'][0]=0
    with pytest.raises(ValueError,match='positive'):mod.PoissonFeatures(p)
