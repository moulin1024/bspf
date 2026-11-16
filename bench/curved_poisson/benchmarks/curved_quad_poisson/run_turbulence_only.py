"""Same four-method campaign with the tanh contribution removed entirely."""
import run as r
from scipy.special import roots_legendre
import numpy as np
r.OUT=r.ROOT/'evidence/curved_quad_poisson/turbulence_only_v1';r.OUT.mkdir(parents=True,exist_ok=True)
def truth(p,guard=False):
 x,y=np.asarray(p).reshape(-1,2).T
 if guard:return 1+x*x+y*y,2*x,2*y,np.full_like(x,-4.)
 return r.FIELD.truth(x,y)
r.truth=truth
original_prepare=r.prepare
def prepare(method,size,guard):
 a,b,exe=original_prepare(method,size,guard)
 if method=='sem9':exe=r.HERE/'sem_cached'
 elif method in ('fc','chebyshev'):exe=r.HERE/'collocation_cached'
 if method=='bspf':np.savez(r.OUT/f'axis_{size}.npz',x=a.x,coefficients=a.coefficients)
 return a,b,exe
r.prepare=prepare
def validate(a,u,guard):
 errors=[];orders=[]
 for order in ((16,24) if isinstance(a,r.SEM) else (384,512,768)):
  z,w=roots_legendre(order);z=(z+1)/2;w=w/2
  if isinstance(a,r.SEM):z=((np.arange(a.e)[:,None]+z)/a.e).ravel();w=np.tile(w/a.e,a.e)
  label=f'sem_e{a.e}_q{order}' if isinstance(a,r.SEM) else f'global_q{order}'
  path=r.OUT/f'exact_{label}.npz'
  if not path.exists():
   p,_,_,j,*_=r.PATCH.jet(z[:,None],z[None,:]);np.savez(path,u=truth(p,guard)[0].reshape(len(z),len(z)),j=j)
  cache=np.load(path);B=a.evaluate(z,0)[0];weight=np.outer(w,w)*cache['j'];errors.append(float(np.sqrt(np.sum(weight*(B@u@B.T-cache['u'])**2)/np.sum(weight*cache['u']**2))));orders.append(order)
  if len(errors)>=2 and abs(errors[-2]-errors[-1])<.01*errors[-1]+1e-11:break
 return dict(l2_relative=errors[-1],quadrature_errors=errors,quadrature_orders=orders,quadrature_kind='elementwise Gauss' if isinstance(a,r.SEM) else 'global Gauss',quadrature_stable=abs(errors[-2]-errors[-1])<.01*errors[-1]+1e-11)
r.validate=validate
if __name__=='__main__':
 path=r.OUT/'results.json'
 if not path.exists():
  previous=r.json.loads((r.ROOT/'evidence/curved_quad_poisson/v1/results.json').read_text());guards=[v.copy() for v in previous['rows'] if v['guard']]
  for row in guards:row['qualification_reused_from']='../v1/'+row['id']
  r.dump(path,dict(protocol='One native cached-operator solve per MMS configuration, single thread; no warmups or repetitions. Setup/factorization, analytic sampling, IO and independent validation excluded.',geometry=previous['geometry'],mms='Synthetic turbulence only: saved seed20261115, shells4..64, k^(-5/3) shell energy. No tanh or Gaussian addition.',rows=guards))
 r.main()
