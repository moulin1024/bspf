"""Verify mixed-MMS Galerkin quadrature and select resolved benchmark results."""
import importlib.util
import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mixed_galerkin',HERE/'bspf_poisson/run.py')
harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
ROOT=HERE/'evidence/gaussian_mms_rtx3070'
MODES=HERE/'evidence/nonperiodic_curved_gaussian/modes.npz'
np,cp=harness.np,harness.cp


def run(basis,n,q,output):
    path=output/f'n{n}_q{q}/result.json'
    if not path.exists():
        harness.case(SimpleNamespace(q=q,output=output,polynomial=False,check=False,basis=basis,
                     modes=MODES),n)
    return path,json.loads(path.read_text())


def probe(basis,n,q):
    folder=ROOT/'quadrature'/basis/f'n{n}_q{q}'
    folder.mkdir(parents=True,exist_ok=True)
    # A probe needs the new solution, but no second expensive exact-MMS audit.
    axis,arrays,sampler,meta=harness.prepare(n,q,False,folder,basis=basis,
                            modes=MODES)
    env=os.environ.copy();sdk=os.environ.get('BSPF_HPC_SDK','/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5')
    env['LD_LIBRARY_PATH']=f'{sdk}/math_libs/lib64:{sdk}/cuda/13.2/lib64'
    with (folder/'native.log').open('w') as log:
        subprocess.run([str(harness.EXE),str(folder/'input.bin'),str(folder/'solution')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    native=json.loads((folder/'solution.json').read_text())
    assert all(r['relative_residual']<2e-12 for r in native['runs'])
    u=harness.read_bundle(folder/'solution.bin')['solution'].reshape(n,n,n)
    wall=np.ones(u.shape,bool);wall[1:-1,1:-1,1:-1]=False
    assert np.max(np.abs(u[wall]-arrays['boundary'].reshape(u.shape)[wall]))==0
    (folder/'probe_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return axis,u,folder


def difference_norm(axis,u,delta,q):
    """Physical L2 norms, without evaluating the analytic plane-wave sum."""
    z,w=np.polynomial.legendre.leggauss(q);z=(z+1)/2;w=w/2
    B=cp.asarray(axis.evaluate(z,0)[0])
    xy=[cp.einsum('ai,bj,kji->kba',B,B,cp.asarray(v),optimize=True) for v in (u,delta)]
    yy,xx=cp.meshgrid(cp.asarray(z),cp.asarray(z),indexing='ij')
    ww=cp.asarray(w[:,None]*w[None,:]);sums=cp.zeros(2)
    for k,zk in enumerate(z):
        _,_,det=harness.CurvedBox().map(cp.stack([xx,yy,cp.full_like(xx,zk)],axis=-1),cp)
        for j,v in enumerate(xy):
            values=cp.einsum('k,kji->ji',B[k],v)
            sums[j]+=cp.sum(values**2*ww*w[k]*det)
    return np.sqrt(cp.asnumpy(sums))


def main():
    global ROOT,MODES
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite-root',type=Path,default=ROOT)
    parser.add_argument('--modes',type=Path,default=MODES)
    args=parser.parse_args();ROOT=args.suite_root.resolve();MODES=args.modes.resolve()
    report={'criterion':'Audited relative L2 error change <= max(1% of the measured error, 1e-12), with integration order increased by 50%; certified by a solution-difference bound when possible',
            'bound':'||u_high-u_low|| / ||u_low|| * (1+e_low) / e_low; reverse triangle inequality bounds the relative error change',
            'checks':[], 'selected':{}}
    for basis in ['bspf','chebyshev']:
        for n in [17,33,65,97,129,145]:
            q=max(96,(3*n+1)//2)
            path,base=run(basis,n,q,ROOT/basis)
            for attempt in range(3):
                higher=(3*q+1)//2
                if higher>384:raise RuntimeError(f'Quadrature cap exceeded for {basis} n={n}; refine explicitly')
                axis,uhigh,folder=probe(basis,n,higher)
                u=harness.read_bundle(path.parent/'solution.bin')['solution'].reshape(n,n,n)
                norms=[difference_norm(axis,u,uhigh-u,k) for k in (higher,higher+8)]
                if np.max(np.abs(norms[0]/norms[1]-1))>1e-5:
                    raise RuntimeError(f'Solution-difference norm quadrature unresolved: {norms}')
                ratio=float(norms[-1][1]/norms[-1][0])
                change=ratio*(1+base['relative_l2'])/base['relative_l2']
                tolerance=max(.01,1e-12/base['relative_l2'])
                report['checks'].append(dict(basis=basis,n=n,q=q,check_q=higher,
                    relative_l2=base['relative_l2'],relative_solution_difference=ratio,
                    relative_error_change_bound=change,relative_change_tolerance=tolerance,norm_quadrature=[higher,higher+8],
                    physical_norms=[v.tolist() for v in norms],probe=str(folder.relative_to(ROOT))))
                cp.get_default_memory_pool().free_all_blocks()
                if change>tolerance:
                    # A conservative bound failing does not prove the error changed.
                    checkpath,check=run(basis,n,higher,ROOT/'quadrature'/basis)
                    change=abs(check['relative_l2']/base['relative_l2']-1)
                    report['checks'][-1].update(check_relative_l2=check['relative_l2'],
                                               relative_error_change=change)
                if change<=tolerance:
                    report['selected'][f'{basis}_{n}']=str(path.relative_to(ROOT))
                    break
                path,base,q=checkpath,check,higher
            else:raise RuntimeError(f'Quadrature not converged for {basis} n={n}')
            (ROOT/'quadrature_validation.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'VALIDATED {basis} n={n} q={q} error_change={change:.4g}',flush=True)


if __name__=='__main__':main()
