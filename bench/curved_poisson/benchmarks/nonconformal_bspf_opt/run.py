"""Single-call, fixed-reference nonconformal BSPF optimization audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,subprocess,argparse,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/curved_quad_poisson'))
# Avoid shadowing this run.py when the historical case imports run.
import importlib.util
spec=importlib.util.spec_from_file_location('run',ROOT/'benchmarks/curved_quad_poisson/run.py');r=importlib.util.module_from_spec(spec);sys.modules['run']=r;spec.loader.exec_module(r)
import run_turbulence_only as case
import numpy as np
from numpy.polynomial.chebyshev import chebder
OUT=ROOT/'evidence/nonconformal_bspf_opt/v1';OUT.mkdir(exist_ok=True,parents=True)
def axis(n):
 path=ROOT/f'evidence/curved_quad_poisson/turbulence_only_v1/axis_{n}.npz'
 if not path.exists():path=OUT/f'axis_{n}.npz'
 if path.exists():
  a=r.HybridAxis.__new__(r.HybridAxis);c=np.load(path);a.n=n;a.x=c['x'];a.coefficients=c['coefficients'];a.derivatives=[chebder(a.coefficients,m=k,axis=0)*2**k for k in range(3)]
 else:
  a=r.HybridAxis((n-8)//2,nodes='symmetric_qr');np.savez(path,x=a.x,coefficients=a.coefficients)
 return a

def configure_hard():
 global OUT
 OUT=ROOT/'evidence/nonconformal_bspf_opt/hard_shell128';OUT.mkdir(exist_ok=True,parents=True)
 path=OUT/'field.npz'
 from turbulence import create,verify
 if not path.exists():
  field,shells,energy=create(seed=20261115,n=512,shell_max=128)
  verification,_,_=verify(field,field.grid_fields(),shells,energy)
  np.savez(path,coefficients=field.coefficients,length=field.length,origin=field.origin)
  r.dump(OUT/'mms.json',dict(seed=20261115,shell_min=4,shell_max=128,parent_length=2.,rms=1.,spectrum='shell energy k^(-5/3)',verification=verification,geometry='unchanged four-cubic-edge Coons patch',quadrature_policy='q=2*n',description='Doubled maximum physical wavenumber; randomized phases and amplitudes; no boundary damping or tanh'))
 r.FIELD=r.SyntheticField.load(path)

def hard_validate(a,u):
 errors=[]
 for order in (640,768):
  z,w=r.roots_legendre(order);z=(z+1)/2;w=w/2;path=OUT/f'exact_global_q{order}.npz'
  if not path.exists():
   p,_,_,j,*_=r.PATCH.jet(z[:,None],z[None,:]);np.savez(path,u=r.truth(p)[0].reshape(order,order),j=j)
  c=np.load(path);B=a.evaluate(z,0)[0];weight=np.outer(w,w)*c['j'];errors.append(float(np.sqrt(np.sum(weight*(B@u@B.T-c['u'])**2)/np.sum(weight*c['u']**2))))
 return dict(l2_relative=errors[-1],quadrature_errors=errors,quadrature_orders=[640,768],quadrature_stable=abs(errors[-1]-errors[-2])<.01*errors[-1]+1e-11)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,required=True);ap.add_argument('--q',type=int,default=None,help='Gauss points per axis; default: twice --n');ap.add_argument('--kernel',choices=['modal','baseline'],default='modal');ap.add_argument('--guard',action='store_true');ap.add_argument('--forcing-source',choices=['direct','nodal'],default='direct');ap.add_argument('--same-nodes',action='store_true',help='Integrate on BSPF nodes with positive interpolatory weights');ap.add_argument('--hard',action='store_true',help='Turbulence shells 4..128, separate evidence');args=ap.parse_args()
 if args.forcing_source=='nodal' and args.kernel!='modal':ap.error('nodal forcing currently requires the modal native adapter')
 if not args.hard:ap.error('This release bundle requires --hard (the fixed shell-4..128 MMS)')
 if args.hard:configure_hard()
 if args.same_nodes:
  if args.q is not None and args.q!=args.n:ap.error('--same-nodes requires q=n')
  args.q=args.n
 if args.q is None:args.q=2*args.n
 name=f'{args.kernel}_n{args.n}_q{args.q}'+('_same_nodes' if args.same_nodes else '')+('_nodal_rhs' if args.forcing_source=='nodal' else '')+('_guard' if args.guard else '')
 folder=OUT/name;folder.mkdir(exist_ok=True);result=folder/'result.json'
 if result.exists():print(result.read_text());return
 row=vars(args)|dict(id=name,status='preparing');r.dump(result,row)
 start=time.perf_counter();a=axis(args.n);rule=None
 if args.same_nodes:
  # Setup only: integrate the represented cardinal basis to obtain its nodal rule.
  # No RHS or MMS values enter these weights, nor does the auxiliary grid enter CG.
  z,w=r.roots_legendre(2*args.n);weights=(w/2)@a.evaluate((z+1)/2,0)[0]
  row['nodal_weight_min']=float(weights.min());row['nodal_weight_sum']=float(weights.sum())
  if np.any(weights<=0):
   row.update(status='unsupported',failure='Nonpositive nodal quadrature weights would invalidate SPD CG');r.dump(result,row);print(row,flush=True);return
  rule=(a.x,weights);np.savez(folder/'nodal_rule.npz',nodes=a.x,weights=weights)
 try:eng=r.MatrixFreeDirichlet(a,r.PATCH,args.q,operator_mode='quadrature',quadrature_rule=rule)
 except (ValueError,np.linalg.LinAlgError) as error:
  row.update(status='setup_failed',failure=str(error),solve_calls=0,setup_seconds=time.perf_counter()-start);r.dump(result,row);print(row,flush=True);return
 row['setup_seconds']=time.perf_counter()-start
 row['quadrature_kind']='BSPF solution nodes, interpolatory weights' if args.same_nodes else 'Gauss-Legendre'
 row['integration_solution_node_max_difference']=float(np.max(abs(eng.q-a.x))) if len(eng.q)==a.n else None
 start=time.perf_counter()
 if args.forcing_source=='nodal':
  nodal_truth=r.truth(eng.nodes,args.guard)
  fn=nodal_truth[3][eng.ids[0]].reshape(a.n,a.n);g=nodal_truth[0][eng.boundary]
  b=r.arrays(eng,np.zeros((1,args.q*args.q)),g);del b['forcing'];b['forcing_nodes']=fn
  row.update(forcing_sample_count=a.n*a.n,forcing_interpolation_inside_timer=True)
 else:
  f,g=eng.sample(lambda p:r.truth(p,args.guard)[3],lambda p,normal=None:r.truth(p,args.guard)[0]);b=r.arrays(eng,f,g)
  row.update(forcing_sample_count=args.q*args.q,forcing_interpolation_inside_timer=False)
 row['sampling_seconds']=time.perf_counter()-start
 r.write_bundle(folder/'input.bin',b)
 exe=HERE/'poisson_single' if args.kernel=='modal' else ROOT/'benchmarks/poisson_campaign/sota_v1/poisson_single'
 if args.forcing_source=='nodal':exe=HERE/'poisson_nodal'
 row['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
 proc=subprocess.run([str(exe),str(folder/'input.bin'),str(folder/'solution')],text=True,capture_output=True);(folder/'native.log').write_text(proc.stdout+proc.stderr)
 if proc.returncode:row.update(status='failed',failure=proc.stderr);r.dump(result,row);print(row,flush=True);return
 row.update(json.loads((folder/'solution.json').read_text()));u=r.read_bundle(folder/'solution.bin')['solution']
 if args.guard:
  z,w=r.roots_legendre(80);z=(z+1)/2;B=a.evaluate(z,0)[0];pts=r.PATCH.jet(z[:,None],z[None,:])[0];row['guard_max_error']=float(np.max(abs(B@u@B.T-r.truth(pts,True)[0].reshape(80,80))))
 else:row.update(hard_validate(a,u) if args.hard else case.validate(a,u,False))
 # Independent original high-quadrature weak defect, never used to generate solution.
 audit_order=max(640,2*args.n+32) if args.hard else 512
 audit=r.MatrixFreeDirichlet(a,r.PATCH,audit_order,operator_mode='quadrature')
 cache=OUT/f'audit_forcing_q{audit_order}_guard{args.guard}.npy'
 if cache.exists():af=np.load(cache)
 else:
  af=r.truth(audit.points[0],args.guard)[3].reshape(1,-1);np.save(cache,af)
 analytic_af=af.copy()
 if args.forcing_source=='nodal':
  af=(audit.basis@fn@audit.basis.T).reshape(1,-1)
  ww=audit.weights[0].ravel();err=(af-analytic_af).ravel()
  row['forcing_interpolation_relative_l2']=float(np.sqrt(np.sum(ww*err**2)/np.sum(ww*analytic_af.ravel()**2)))
 ab=r.arrays(audit,af,np.zeros(len(audit.boundary)))
 B,D=ab['B'],ab['D'];us=D@u@B.T;ut=B@u@D.T
 residual=(B.T@(ab['forcing']*ab['weights'])@B-D.T@(ab['ss']*us+ab['st']*ut)@B-B.T@(ab['st']*us+ab['tt']*ut)@D)[1:-1,1:-1]
 V=ab['V'];row['audit_quadrature_order']=audit_order;row['audit_preconditioned_defect']=float(np.linalg.norm(V@((V.T@residual@V)*ab['inverse'])@V.T)/np.linalg.norm(u))
 if args.forcing_source=='nodal':
  analytic_residual=residual+(B.T@((analytic_af-af).reshape(audit_order,audit_order)*ab['weights'])@B)[1:-1,1:-1]
  row['analytic_rhs_preconditioned_defect']=float(np.linalg.norm(V@((V.T@analytic_residual@V)*ab['inverse'])@V.T)/np.linalg.norm(u))
 row['status']='validated' if row['audit_preconditioned_defect']<1e-11 and (row.get('quadrature_stable',True)) and row.get('guard_max_error',0)<1e-9 else 'audit_failed'
 r.dump(result,row);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
