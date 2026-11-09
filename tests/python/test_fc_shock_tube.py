"""Physical wall handling and short-time refinement for FC-Gram NS."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def fc_module(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root/'examples/pde'))
    data=root/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    if not data.exists():pytest.skip('Run bench/fetch_fc_data.py to fetch external FC matrices')
    return importlib.import_module('fc_shock_tube'),data


def test_fc_tube_constant_equilibrium(fc_module):
    m,data=fc_module;s=m.FCShockTube(65,data)
    state=np.tile([1.,0.,2.5],(65,1))
    np.testing.assert_array_equal(s.rhs(state),np.zeros_like(state))


def test_wall_density_and_energy_are_not_frozen(fc_module):
    m,data=fc_module;s=m.FCShockTube(129,data)
    u=.01*np.sin(np.pi*s.x)
    state=np.column_stack((np.ones(129),u,2.5+.5*u*u))
    rate=s.rhs(state)
    np.testing.assert_array_equal(rate[[0,-1],1],0.)
    np.testing.assert_allclose(rate[[0,-1],0],[-.01*np.pi,.01*np.pi],atol=1e-8)
    assert np.min(abs(rate[[0,-1],2]))>.09


def test_fc_tube_short_time_refinement_and_balance(fc_module):
    m,data=fc_module;results=[]
    for n in (129,257,513):
        results.append(m.solve_fc_tube(n,data,final_time=.002,output_points=2001))
    reference=results[-1]['output']
    errors=[np.max(abs(r['output']-reference)) for r in results[:-1]]
    assert errors[1]<errors[0]/5
    fine=results[-1]['summary']
    assert abs(fine['relative_mass_drift'])<1e-7
    assert abs(fine['relative_energy_drift'])<1e-7
    assert fine['min_density']>.12 and fine['min_pressure']>.09
    assert fine['rejected_stages']==0
