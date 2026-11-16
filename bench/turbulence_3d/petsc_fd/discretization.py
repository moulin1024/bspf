"""FD8 coefficients and analytic transformation of the physical Laplacian."""
from fractions import Fraction
from functools import lru_cache
from math import factorial
import numpy as np

@lru_cache(None)
def weights(offsets, derivative):
    """Exact rational moment solve, converted to double only at the end."""
    size=len(offsets)
    a=[[Fraction(x)**k for x in offsets]+[Fraction(factorial(k) if k==derivative else 0)] for k in range(size)]
    for j in range(size):
        pivot=next(i for i in range(j,size) if a[i][j])
        a[j],a[pivot]=a[pivot],a[j]
        scale=a[j][j];a[j]=[v/scale for v in a[j]]
        for i in range(size):
            if i!=j:
                scale=a[i][j];a[i]=[v-scale*w for v,w in zip(a[i],a[j])]
    return np.array([float(a[i][-1]) for i in range(size)])

def stencils(n):
    if n<11:raise ValueError('FD8 requires at least 11 nodes per axis')
    starts=np.empty((2,n),dtype=np.int32); coeff=np.zeros((2,n,10))
    for d in [1,2]:
        for i in range(n):
            # D2 has ninth-degree exactness: 9 symmetric interior points,
            # 10 points for shifted/one-sided boundary closures.
            length=9 if d==1 or 4<=i<n-4 else 10
            start=min(max(i-4,0),n-length)
            starts[d-1,i]=start
            coeff[d-1,i,:length]=weights(tuple(range(start-i,start-i+length)),d)*(n-1)**d
    return starts,coeff

def geometry(s,box,xp=np):
    physical,J,det=box.map(s,xp)
    inv=xp.linalg.inv(J);G=inv@inv.swapaxes(-1,-2)
    phi=4*s*(1-s);dp=4*(1-2*s)
    lapX=xp.empty(s.shape,dtype=xp.float64)
    for a in range(3):
        j,k=(a+1)%3,(a+2)%3; amplitude=float(box.amplitudes[a]);f=1-2*s[...,a]
        lapX[...,a]=amplitude*(-8*f*(G[...,j,j]*phi[...,k]+G[...,k,k]*phi[...,j])
                        -4*(G[...,a,j]*dp[...,j]*phi[...,k]+G[...,a,k]*phi[...,j]*dp[...,k])
                        +2*f*G[...,j,k]*dp[...,j]*dp[...,k])
    drift=-xp.einsum('...ij,...j->...i',inv,lapX)
    return physical,G,drift,det

def cpu_apply(u,starts,coeff,G,drift):
    """Independent NumPy reference, including nonperiodic boundary closures."""
    n=u.shape[0]
    def diff(a,axis,order):
        out=np.zeros_like(a)
        for i in range(n):
            length=9 if order==1 or 4<=i<n-4 else 10
            sl=[slice(None)]*3;sl[axis]=i
            ind=np.arange(starts[order-1,i],starts[order-1,i]+length)
            out[tuple(sl)]=np.tensordot(coeff[order-1,i,:length],np.moveaxis(np.take(a,ind,axis=axis),axis,0),axes=(0,0))
        return out
    grads=[diff(u,2-i,1) for i in range(3)]
    ans=np.zeros_like(u)
    for i in range(3):
        ans-=G[...,i,i]*diff(u,2-i,2)+drift[...,i]*grads[i]
        for j in range(i+1,3):ans-=2*G[...,i,j]*diff(grads[j],2-i,1)
    return ans[1:-1,1:-1,1:-1]
