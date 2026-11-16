"""Single-call native Poisson comparison on a four-spline Coons quadrilateral."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/curved-poisson-mpl'
import sys
sys.dont_write_bytecode=True
import json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'evidence/curved_quad_poisson/v1';OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks/poisson_campaign/sota_v1'),str(ROOT/'benchmarks/internal_soliton/cpp'),str(ROOT/'benchmarks/paper_conformal')]
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import roots_legendre,expit
from scipy.fft import dct
from scipy.linalg import schur
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from optimized_bvp import HybridAxis,MatrixFreeDirichlet
from polynomial_axis import PolynomialAxis
from turbulence import SyntheticField
from bundle import write_bundle,read_bundle
from run_pilot import arrays
FIELD=None  # configure_hard() loads or deterministically generates the shared MMS.
class Patch:
 count=1;boundary_edges=[(0,i) for i in range(4)]
 controls=np.array([[[0,0],[.3,-.12],[.7,-.08],[1,.05]], [[-.06,1],[.28,1.15],[.76,1.18],[1.08,1.05]], [[0,0],[-.12,.28],[-.14,.72],[-.06,1]], [[1,.05],[1.18,.32],[1.15,.77],[1.08,1.05]]])
 def __init__(self):self.edges=[BSpline([0]*4+[1]*4,c,3) for c in self.controls]
 def jet(self,s,t):
  s,t=np.broadcast_arrays(s,t);s=s[...,None];t=t[...,None];b,top,l,r=self.edges
  p00,p10=b(0),b(1);p01,p11=top(0),top(1)
  def ev(f,x,d=0):return f(x[...,0],nu=d)
  bil=(1-s)*(1-t)*p00+s*(1-t)*p10+(1-s)*t*p01+s*t*p11
  x=(1-t)*ev(b,s)+t*ev(top,s)+(1-s)*ev(l,t)+s*ev(r,t)-bil
  xs=(1-t)*ev(b,s,1)+t*ev(top,s,1)-ev(l,t)+ev(r,t)-((1-t)*(p10-p00)+t*(p11-p01))
  xt=-ev(b,s)+ev(top,s)+(1-s)*ev(l,t,1)+s*ev(r,t,1)-((1-s)*(p01-p00)+s*(p11-p10))
  ss=(1-t)*ev(b,s,2)+t*ev(top,s,2);tt=(1-s)*ev(l,t,2)+s*ev(r,t,2)
  st=-ev(b,s,1)+ev(top,s,1)-ev(l,t,1)+ev(r,t,1)-(p11-p01-p10+p00)
  jac=xs[...,0]*xt[...,1]-xs[...,1]*xt[...,0]
  assert np.min(jac)>0
  return x,xs,xt,jac,ss,st,tt
 def map(self,p,s,t):return tuple(a.reshape(-1,2) if i<3 else a.ravel() for i,a in enumerate(self.jet(s,t)[:4]))
PATCH=Patch()
def truth(p,guard=False):
 x,y=np.asarray(p).reshape(-1,2).T
 if guard:return 1+x*x+y*y,2*x,2*y,np.full_like(x,-4.)
 u,ux,uy,f=FIELD.truth(x,y)
 def window(x,c):
  z1=(x-c+.125)/.01;z2=(x-c-.125)/.01;norm=np.tanh(12.5)
  a=np.exp(-2*abs(z1));b=np.exp(-2*abs(z2));sa=4*a/(1+a)**2;sb=4*b/(1+b)**2
  w=(expit((.25-2*abs(x-c))/.01)-expit((-.25-2*abs(x-c))/.01))/norm
  return w,(sa-sb)/(.02*norm),(-np.tanh(z1)*sa+np.tanh(z2)*sb)/(.0001*norm)
 a,ad,add=window(x,.37);b,bd,bdd=window(y,.61)
 return u+3*a*b,ux+3*ad*b,uy+3*a*bd,f-3*(add*b+a*bdd)
def dump(p,data):p.write_text(json.dumps(data,indent=2)+'\n')
class Cheb:
 def __init__(self,n):
  self.n=n;self.x=(1-np.cos(np.arange(n)*np.pi/(n-1)))/2
  self.co=dct(np.eye(n)[::-1],type=1,axis=0)/(n-1);self.co[[0,-1]]*=.5
  self.D1,self.D2=self.evaluate(self.x,2)[1:]
  for d in (self.D1,self.D2):d[np.diag_indices(n)]-=d.sum(axis=1)
 def evaluate(self,x,order=0):
  from numpy.polynomial.chebyshev import chebvander,chebder
  return [chebvander(2*np.asarray(x)-1,self.n-1-k)@chebder(self.co,m=k,axis=0)*2**k for k in range(order+1)]
def collocation(method,n,guard):
 if method!='chebyshev':raise ValueError('FC is outside this scalar release bundle')
 a=Cheb(n)
 p,xs,xt,j,ss,st,tt=PATCH.jet(a.x[:,None],a.x[None,:]);A=np.sum(xt*xt,axis=-1)/j**2;C=np.sum(xs*xs,axis=-1)/j**2;B=-np.sum(xs*xt,axis=-1)/j**2
 rx=A*ss[...,0]+2*B*st[...,0]+C*tt[...,0];ry=A*ss[...,1]+2*B*st[...,1]+C*tt[...,1]
 G=(-xt[...,1]*rx+xt[...,0]*ry)/j;H=(xs[...,1]*rx-xs[...,0]*ry)/j
 T,Q=schur(-a.D2[1:-1,1:-1],output='real');u,_,_,f=truth(p,guard);boundary=u.reshape(n,n).copy();boundary[1:-1,1:-1]=0
 b=dict(D=a.D1,D2=a.D2,A=A,B=B,C=C,G=G,H=H,T=T,Q=Q,averages=np.array([[A.mean(),C.mean()]]),forcing=f.reshape(n,n),boundary=boundary)
 return a,b,HERE/'collocation_single'
class SEM:
 def __init__(self,e):
  self.e=e;self.local=PolynomialAxis(10);self.n=e*9+1;self.x=np.r_[np.concatenate([(i+self.local.x[:-1])/e for i in range(e)]),1.]
 def evaluate(self,x,order=0):
  x=np.asarray(x);e=np.minimum((x*self.e).astype(int),self.e-1);local=self.local.evaluate(x*self.e-e,order);result=[]
  for k in range(order+1):
   b=np.zeros((len(x),self.n));b[np.arange(len(x))[:,None],e[:,None]*9+np.arange(10)]=local[k]*self.e**k;result.append(b)
  return result
 def setup(self,guard):
  e=self.e;n=self.n;q=32;z,w=roots_legendre(q);z=(z+1)/2;w=w/2;B,D=self.local.evaluate(z,1)
  us=np.kron(D,B);ut=np.kron(B,D);ids=[];forcing=[];weights=[];rows=[];cols=[];vals=[]
  for i in range(e):
   for k in range(e):
    p,xs,xt,j,*_=PATCH.jet((i+z[:,None])/e,(k+z[None,:])/e);ww=np.outer(w,w);ss=ww*np.sum(xt*xt,axis=-1)/j;tt=ww*np.sum(xs*xs,axis=-1)/j;st=-ww*np.sum(xs*xt,axis=-1)/j
    K=us.T@(ss.ravel()[:,None]*us)+ut.T@(tt.ravel()[:,None]*ut)+us.T@(st.ravel()[:,None]*ut)+ut.T@(st.ravel()[:,None]*us)
    ix=((i*9+np.arange(10))[:,None]*n+k*9+np.arange(10)[None,:]).ravel();ids.append(ix);rows.extend(np.repeat(ix,100));cols.extend(np.tile(ix,100));vals.extend(K.ravel());forcing.append(truth(p,guard)[3]);weights.append((ww*j/e**2).ravel())
  K=coo_matrix((vals,(rows,cols)),shape=(n*n,n*n)).tocsr();free=np.arange(n*n).reshape(n,n)[1:-1,1:-1].ravel();p=PATCH.jet(self.x[:,None],self.x[None,:])[0];boundary=truth(p,guard)[0].reshape(n,n);boundary[1:-1,1:-1]=0
  lu=splu(K[free][:,free].tocsc());b=dict(ids=np.array(ids),basis=B,forcing=np.array(forcing),weights=np.array(weights),boundary=boundary,free=free[:,None],pr=lu.perm_r[:,None],pc=lu.perm_c[:,None])
  for name,mat in [('K',K),('L',lu.L.tocsr()),('U',lu.U.tocsr())]:
   for suffix,v in [('ptr',mat.indptr),('ind',mat.indices),('val',mat.data)]:b[name+suffix]=(v[:,None] if len(v)<=100000 else np.pad(v,(0,(-len(v))%1024)).reshape(-1,1024))
  return b

def prepare(method,size,guard):
 if method=='bspf':
  a=HybridAxis((size-8)//2,nodes='symmetric_qr');eng=MatrixFreeDirichlet(a,PATCH,max(512,2*size+32));f,g=eng.sample(lambda p:truth(p,guard)[3],lambda p,normal=None:truth(p,guard)[0]);return a,arrays(eng,f,g),ROOT/'benchmarks/poisson_campaign/sota_v1/poisson_single'
 if method=='sem9':
  a=SEM(size);return a,a.setup(guard),HERE/'sem_single'
 return collocation(method,size,guard)

def validate(a,u,guard):
 errors=[]
 for qn in (384,512):
  z,w=roots_legendre(qn);z=(z+1)/2;w=w/2
  path=OUT/f'exact_{"guard" if guard else "turbulence_tanh"}_{qn}.npz'
  if not path.exists():
   p,_,_,j,*_=PATCH.jet(z[:,None],z[None,:]);exact=truth(p,guard)[0].reshape(qn,qn);np.savez(path,u=exact,j=j)
  cache=np.load(path);B=a.evaluate(z,0)[0];diff=B@u@B.T-cache['u'];weight=np.outer(w,w)*cache['j'];errors.append(float(np.sqrt(np.sum(weight*diff**2)/np.sum(weight*cache['u']**2))))
 return dict(l2_relative=errors[-1],quadrature_errors=errors,quadrature_stable=abs(errors[0]-errors[1])<.01*errors[-1]+1e-11)

def main():
 path=OUT/'results.json';data=json.loads(path.read_text()) if path.exists() else dict(protocol='One native timed solve per case, single thread; setup, analytic sampling, IO and independent error evaluation excluded. Reused operator/factor solve cost, no warmups.',geometry=dict(type='Coons patch, four independent cubic Bezier (single-span B-spline) edges',controls=PATCH.controls.tolist()),mms='seed20261115 turbulence shells4..64 plus amplitude3 tanh plateau width .25, transition .01',rows=[])
 jobs=[(m,24 if m!='sem9' else 2,True) for m in ('bspf','fc','chebyshev','sem9')]+[(m,n,False) for n in (64,96,128,192,256,384) for m in ('bspf','fc','chebyshev')]+[('sem9',e,False) for e in (8,12,16,24,32)]
 for method,size,guard in jobs:
  name=f'{method}_{size}_{"guard" if guard else "mms"}'
  previous=next((r for r in data['rows'] if r['id']==name),None)
  if previous is not None:
   if previous.get('status')=='failed' and 'Invalid dimensions' in previous.get('failure',''):
    data.setdefault('input_format_failures',[]).append(previous);data['rows'].remove(previous)
   else:continue
  folder=OUT/name;folder.mkdir(exist_ok=True);r=dict(id=name,method=method,size=size,guard=guard,status='preparing');data['rows'].append(r);dump(path,data)
  try:
   if previous is not None and 'Invalid dimensions' in previous.get('failure',''):
    a=SEM(size);b=read_bundle(folder/'input.bin');exe=HERE/'sem_single'
    for key,v in list(b.items()):
     if max(v.shape)>100000:b[key]=np.pad(v.ravel(),(0,(-v.size)%1024)).reshape(-1,1024)
    r.update(setup_and_sampling_seconds=previous['setup_and_sampling_seconds'],n=a.n,dofs=(a.n-2)**2,input_repacked_before_first_solve=True)
   else:
    start=time.perf_counter();a,b,exe=prepare(method,size,guard);r.update(setup_and_sampling_seconds=time.perf_counter()-start,n=a.n,dofs=(a.n-2)**2)
   write_bundle(folder/'input.bin',b);r['status']='attempted';dump(path,data)
   proc=subprocess.run([str(exe),str(folder/'input.bin'),str(folder/'solution')],capture_output=True,text=True,timeout=180);(folder/'native.log').write_text(proc.stdout+proc.stderr)
   if proc.returncode:raise RuntimeError(proc.stderr)
   r['timing']=json.loads((folder/'solution.json').read_text());u=read_bundle(folder/'solution.bin')['solution'];r.update(validate(a,u,guard));r['status']='validated' if r['quadrature_stable'] and r['timing']['relative_residual']<2e-9 else 'unqualified'
   if guard and r['l2_relative']>1e-7:r['status']='guard_failed'
  except Exception as exc:r.update(status='failed',failure=repr(exc));traceback.print_exc()
  dump(path,data);print(name,r['status'],r.get('timing',{}).get('solve_seconds'),r.get('l2_relative'),flush=True)
if __name__=='__main__':main()
