"""Fixed dictionary selection and analytical-data isolation checks."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples/pde'))
    return importlib.import_module('dictionary_spline_shock')


def test_dictionary_labels_and_projection_residual(module):
    base=module.BSPFNavierStokes(25,quadrature=8,line_options=dict(n_basis=10))
    d=module.BlockDictionary(base,spacing=1/64,widths=(1/128,),probe_cells=64)
    p=base.line.evaluate(d.x).matrix
    np.testing.assert_allclose(d.analysis@p[:,1:-1],0.,atol=2e-13)
    for center,width in d.labels:
        assert center*64==round(center*64) and width==1/128
    shock=importlib.import_module('viscous_traveling_shock').BeckerShock()
    index,score=d.select(shock.state(d.x),shock.state(np.array([0.,1.])))
    assert 0<=index<len(d.labels) and score>0
    assert d.ranked_scales(shock.state(d.x),shock.state(np.array([0.,1.])))[0][0]==index


def test_evolution_does_not_query_exact_interior_or_detector(module,monkeypatch):
    shock=importlib.import_module('viscous_traveling_shock').BeckerShock()
    detector=importlib.import_module('shock_tube_enrichment')
    def forbidden(*args,**kwargs):raise AssertionError('Front detector called')
    monkeypatch.setattr(detector,'detect',forbidden)
    original=module.advance;started=False
    def initial(x):
        assert not started,'Exact interior queried after evolution started'
        return shock.state(x)
    def advance(*args,**kwargs):
        nonlocal started
        started=True
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'advance',advance)
    result=module.solve(initial,shock.state(np.array([0.,1.])),final_time=.002,interval=.001,
        probe_cells=64,spacing=1/256,widths=(1/128,),output_points=1025,fit_tolerance=1e-6)
    assert started and len(result['summary']['updates'])==2
    assert result['summary']['rejected_stages']==0
    assert result['summary']['final_modes']==70
    np.testing.assert_allclose(result['output'][[0,-1]],shock.state(np.array([0.,1.])),atol=2e-11)
