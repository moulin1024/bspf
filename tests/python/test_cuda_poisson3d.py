"""Independent checks for the native mixed-precision tensor Galerkin solver.

Build bench/turbulence_3d/bspf_poisson first; skipped without its executable/CUDA.
Uses CPU geometry and tensor contractions, never native output as reference.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import scipy.linalg as la

ROOT=Path(__file__).resolve().parents[2]
EXE=Path(os.environ.get('BSPF_POISSON3D_EXE','/tmp/bspf-poisson3d-cuda-build/bspf_poisson3d'))

@pytest.fixture(scope='module')
def harness():
    if not EXE.exists():pytest.skip('CUDA Poisson benchmark not built')
    cp=pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount()<1:pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CUDA unavailable')
    spec=importlib.util.spec_from_file_location('bspf_cuda3d_test_harness',ROOT/'bench/turbulence_3d/bspf_poisson/run.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope='module',params=[(11,28,8),(12,29,7)])
def data(harness,request):
    n,q,degree=request.param
    a=harness.HybridAxis((n-degree-1)//2,degree=degree,nodes='symmetric_qr')
    z,w=np.polynomial.legendre.leggauss(q);z=(z+1)/2;w=w/2
    B,D=a.evaluate(z,1);M=B.T@(w[:,None]*B);K=D.T@(w[:,None]*D)
    lam,V=la.eigh(K[1:-1,1:-1],M[1:-1,1:-1])
    points,J,det=harness.CurvedBox().map(harness.grid(z))
    inv=np.linalg.inv(J);G=inv@inv.swapaxes(-1,-2)
    weight=w[:,None,None]*w[None,:,None]*w[None,None,:]*det
    physical=harness.CurvedBox().map(harness.grid(a.x))[0]
    exact=1+np.sum(physical**2,axis=-1);g=exact.copy();g[1:-1,1:-1,1:-1]=0
    probe=np.random.default_rng(561).normal(size=(n,n,n));probe[[0,-1]]=0;probe[:,[0,-1]]=0;probe[:,:,[0,-1]]=0
    d=dict(B=B,D=D,V=V,**{'lambda':lam},measure=weight.reshape(q*q,q),forcing=np.full((q*q,q),-6.),boundary=g.reshape(n*n,n),probe=probe.reshape(n*n,n))
    for key,(i,j) in zip(['xx','xy','xz','yy','yz','zz'],[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):d[key]=(weight*G[...,i,j]).reshape(q*q,q)
    return n,exact,d

def run(harness,tmp_path,d,check=False):
    harness.write_bundle(tmp_path/'input.bin',d)
    env=os.environ.copy();sdk='/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5'
    env['LD_LIBRARY_PATH']=f'{sdk}/math_libs/lib64:{sdk}/cuda/13.2/lib64'
    return subprocess.run([str(EXE),str(tmp_path/'input.bin'),str(tmp_path/'solution')]+(['check'] if check else []),env=env,capture_output=True,text=True)

def test_curved_operator_preconditioner_and_solution(harness,data,tmp_path):
    n,exact,d=data;result=run(harness,tmp_path,d,True);assert result.returncode==0,result.stderr
    check=harness.check_operator(d,tmp_path,n)
    assert check['operator_relative_error']<2e-12
    u=harness.read_bundle(tmp_path/'solution.bin')['solution'].reshape(exact.shape)
    assert np.max(np.abs(u-exact))<1e-9
    stats=json.loads((tmp_path/'solution.json').read_text())
    assert max(r['relative_residual'] for r in stats['runs'])<1e-11
    assert stats['preconditioner_precision']=='float32' and stats['cg_precision']=='float64'

def test_zero_load_has_no_iterations(harness,data,tmp_path):
    _,_,original=data;d=dict(original);d['forcing']=np.zeros_like(d['forcing']);d['boundary']=np.zeros_like(d['boundary'])
    result=run(harness,tmp_path,d);assert result.returncode==0,result.stderr
    assert not np.any(harness.read_bundle(tmp_path/'solution.bin')['solution'])
    assert all(r['iterations']==0 for r in json.loads((tmp_path/'solution.json').read_text())['runs'])

@pytest.mark.parametrize('invalid',['metric','spectrum','lift'])
def test_rejects_invalid_cg_problem(harness,data,tmp_path,invalid):
    n,_,original=data;d={k:v.copy() for k,v in original.items()}
    if invalid=='metric':d['xx'][0,0]=-1
    elif invalid=='spectrum':d['lambda'][0]=-1
    else:d['boundary'].reshape(n,n,n)[1,1,1]=1
    result=run(harness,tmp_path,d);assert result.returncode!=0
    assert not (tmp_path/'solution.json').exists()


def test_residual_replacement_restarts_cg(harness,data,tmp_path):
    _,_,original=data;d=dict(original);d['refresh']=np.array([[2.]])
    result=run(harness,tmp_path,d);assert result.returncode==0,result.stderr
    stats=json.loads((tmp_path/'solution.json').read_text())
    assert all(r['residual_replacements']>2 and r['relative_residual']<1e-11 for r in stats['runs'])


def test_iteration_limit_is_reported(harness,data,tmp_path):
    _,_,original=data;d=dict(original);d['max_iterations']=np.array([[1.]])
    result=run(harness,tmp_path,d)
    assert result.returncode!=0 and 'iteration limit' in result.stderr
    assert not (tmp_path/'solution.json').exists()


def test_parity_toggle_and_nonsymmetric_fallback(harness,data,tmp_path,monkeypatch):
    n,_,original=data
    for mode in ['enabled','disabled','asymmetric']:
        folder=tmp_path/mode;folder.mkdir()
        d={k:v.copy() for k,v in original.items()}
        monkeypatch.setenv('BSPF_POISSON_PARITY','0' if mode=='disabled' else '1')
        if mode=='asymmetric':d['B'][2,3]+=1e-3
        result=run(harness,folder,d,True);assert result.returncode==0,result.stderr
        stats=json.loads((folder/'solution.json').read_text())
        assert stats['parity_optimized']==(mode=='enabled')
        harness.check_operator(d,folder,n)
    on=harness.read_bundle(tmp_path/'enabled/solution.bin')['solution']
    off=harness.read_bundle(tmp_path/'disabled/solution.bin')['solution']
    assert la.norm(on-off)/la.norm(off)<1e-12


def test_shared_contraction_adjoint_and_equivalence(harness,data,tmp_path,monkeypatch):
    n,_,d=data
    solutions=[]
    for mode in ('0','1'):
        folder=tmp_path/mode;folder.mkdir()
        monkeypatch.setenv('BSPF_POISSON_PARITY','1')
        monkeypatch.setenv('BSPF_POISSON_SHARED',mode)
        result=run(harness,folder,d,True);assert result.returncode==0,result.stderr
        stats=json.loads((folder/'solution.json').read_text())
        assert stats['shared_contractions']==(mode=='1')
        harness.check_operator(d,folder,n)
        solutions.append(harness.read_bundle(folder/'solution.bin')['solution'])
    assert la.norm(solutions[0]-solutions[1])/la.norm(solutions[0])<1e-12


@pytest.mark.parametrize('shared',['0','1'])
def test_fused_layout_matches_unfused_operator(harness,data,tmp_path,monkeypatch,shared):
    n,_,d=data
    checks=[];solutions=[]
    monkeypatch.setenv('BSPF_POISSON_PARITY','1')
    monkeypatch.setenv('BSPF_POISSON_SHARED',shared)
    for mode in ['0','1']:
        folder=tmp_path/mode;folder.mkdir()
        monkeypatch.setenv('BSPF_POISSON_FUSED_LAYOUT',mode)
        result=run(harness,folder,d,True);assert result.returncode==0,result.stderr
        stats=json.loads((folder/'solution.json').read_text())
        assert stats['fused_tensor_layout']==(mode=='1')
        harness.check_operator(d,folder,n)
        checks.append(harness.read_bundle(folder/'solution_check.bin')['apply'])
        solutions.append(harness.read_bundle(folder/'solution.bin')['solution'])
    assert la.norm(checks[0]-checks[1])/la.norm(checks[0])<2e-14
    assert la.norm(solutions[0]-solutions[1])/la.norm(solutions[0])<1e-12
