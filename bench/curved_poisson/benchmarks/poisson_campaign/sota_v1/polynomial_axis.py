"""Standard degree n-1 GLL cardinal polynomial space (barycentric evaluation)."""
import numpy as np
from scipy.special import roots_jacobi,eval_legendre

class PolynomialAxis:
    def __init__(self,n):
        if not isinstance(n,int) or n<3:raise ValueError('n must be an integer >=3')
        self.n=n
        z=np.r_[-1.,roots_jacobi(n-2,1,1)[0],1.]
        self.x=(z+1)/2
        self.w=1/eval_legendre(n-1,z);self.w/=np.max(abs(self.w))
        diff=self.x[:,None]-self.x[None,:];np.fill_diagonal(diff,1.)
        self.d=(self.w[None,:]/self.w[:,None])/diff
        np.fill_diagonal(self.d,0.);np.fill_diagonal(self.d,-self.d.sum(axis=1))
        self.dd=self.d@self.d
        self.metadata=dict(n=n,degree=n-1,basis='GLL cardinal polynomial',evaluation='barycentric')
    def evaluate(self,x,order=2):
        if order not in (0,1,2):raise ValueError('order must be 0,1,2')
        x=np.atleast_1d(x);diff=x[:,None]-self.x[None,:]
        hits=diff==0;rows=hits.any(axis=1)
        safe=np.where(hits,1.,diff);t=self.w/safe;b=t/t.sum(axis=1)[:,None]
        inv=1/safe;s1=(b*inv).sum(axis=1)[:,None]
        d=b*(s1-inv)
        s2=(b*inv**2).sum(axis=1)[:,None]
        dd=2*b*(s1*s1-s1*inv+inv*inv-s2)
        if rows.any():
            ids=np.argmax(hits[rows],axis=1)
            b[rows]=np.eye(self.n)[ids];d[rows]=self.d[ids];dd[rows]=self.dd[ids]
        return [b,d,dd][:order+1]
