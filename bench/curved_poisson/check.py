"""Small native numerical checks; these are not campaign timing measurements."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,importlib.util,tempfile,subprocess,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('check_driver',ROOT/'benchmarks/nonconformal_bspf_opt/run.py');opt=importlib.util.module_from_spec(spec);spec.loader.exec_module(opt);r=opt.r
import numpy as np

def solve(exe,data,folder,label):
 inp=folder/(label+'.input');out=folder/label;r.write_bundle(inp,data)
 subprocess.run([str(exe),str(inp),str(out)],check=True)
 stats=json.loads(out.with_suffix('.json').read_text());assert stats['solve_calls']==1 and stats['warmups']==0
 return r.read_bundle(out.with_suffix('.bin'))['solution'],stats

with tempfile.TemporaryDirectory(prefix='curved-poisson-check-') as tmp:
 folder=Path(tmp);a=r.HybridAxis(8,nodes='symmetric_qr');eng=r.MatrixFreeDirichlet(a,r.PATCH,2*a.n,operator_mode='quadrature')
 f,g=eng.sample(lambda p:r.truth(p,True)[3],lambda p,normal=None:r.truth(p,True)[0]);data=r.arrays(eng,f,g)
 z=np.linspace(0,1,37);B=a.evaluate(z,0)[0];exact=r.truth(r.PATCH.jet(z[:,None],z[None,:])[0],True)[0].reshape(37,37)
 for label in ('direct','nodal'):
  payload=dict(data)
  if label=='nodal':payload.pop('forcing');payload['forcing_nodes']=np.full((a.n,a.n),-4.)
  exe=ROOT/'benchmarks/nonconformal_bspf_opt'/('poisson_single' if label=='direct' else 'poisson_nodal')
  u,stats=solve(exe,payload,folder,label);error=float(np.max(abs(B@u@B.T-exact)));assert error<1e-9,(label,error)
  print(label,'curved quadratic MMS max error',error)
 errors=[]
 for n in (25,49):
  z=np.linspace(0,1,n);p,xs,xt,J,xss,xst,xtt=r.PATCH.jet(z[:,None],z[None,:]);A=np.sum(xt*xt,axis=-1)/J**2;C=np.sum(xs*xs,axis=-1)/J**2;B=-np.sum(xs*xt,axis=-1)/J**2;rv=A[...,None]*xss+2*B[...,None]*xst+C[...,None]*xtt;G=(-xt[...,1]*rv[...,0]+xt[...,0]*rv[...,1])/J;H=(xs[...,1]*rv[...,0]-xs[...,0]*rv[...,1])/J
  x,y=p[...,0],p[...,1];wave=np.sin(3*x+.3)*np.cos(4*y-.2);exact=wave+.2*x+.1*y;g=exact.copy();g[1:-1,1:-1]=0
  u,stats=solve(ROOT/'benchmarks/finite_difference_poisson/fd4',dict(A=A,B=B,C=C,G=G,H=H,forcing=25*wave,boundary=g),folder,f'fd{n}')
  assert stats['converged'] and stats['relative_residual']<2e-12
  errors.append(float(np.linalg.norm(u-exact)/np.linalg.norm(exact)));print('FD4',n,'relative nodal error',errors[-1])
 assert errors[0]/errors[1]>12,errors
 print('FD4 observed order',float(np.log2(errors[0]/errors[1])))
print('PASS')
