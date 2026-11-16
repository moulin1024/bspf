"""Stable independent B-spline/Fourier axis, with nodal trace coordinates.

The spline block is a single-span degree-p B-spline space (P_p). Independent
Fourier pairs have period T on [0,1]. Analytic Chebyshev tails avoid subtraction
of almost identical spline and Fourier columns. This changes the fitted/cardinal
BSPF span; it is based on pybspf scratch/free_bspf/stable_line.py.
"""
import numpy as np
import scipy.linalg as la
from scipy.special import jv
from scipy.interpolate import BSpline
from numpy.polynomial.chebyshev import chebval, chebder
from numpy.polynomial.legendre import Legendre, leggauss

class HybridAxis:
    """Independent polynomial/Fourier space with conforming nodal coordinates.

    ``nodes='gll'`` retains the existing node layout. ``'symmetric_qr'`` selects
    well-conditioned nodes for this space, including both endpoints and exact
    reflection pairs. Use the same axis configuration on conforming edges.
    """
    def __init__(self, modes=4, period=1.5, degree=7, *, nodes='gll'):
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in (modes,degree)):
            raise ValueError('modes and degree must be integers')
        if modes < 1 or degree < 1 or not np.isfinite(period) or period <= 1:
            raise ValueError('Positive modes/degree and finite period > 1 required')
        if nodes not in ('gll','symmetric_qr'):
            raise ValueError("nodes must be 'gll' or 'symmetric_qr'")
        self.node_rule=nodes
        self.n=degree+1+2*modes
        self.spline=BSpline(np.r_[np.zeros(degree+1), np.ones(degree+1)],np.eye(degree+1),degree)
        count=max(128,4*self.n)
        c=np.zeros((count,self.n)); c[:degree+1,:degree+1]=np.eye(degree+1)
        for j in range(modes):
            omega=2*np.pi*(j+1)/period
            for k in range(degree+1,count):
                if k%2==0: c[k,degree+1+j]=2*(-1.)**(k//2)*jv(k,omega/2)
                else: c[k,degree+1+modes+j]=2*(-1.)**((k-1)//2)*jv(k,omega/2)
        c/=np.linalg.norm(c,axis=0)
        q,w=leggauss(max(80,4*self.n)); w=w/2
        # Mixing parity blocks when the raw Fourier tails are nearly dependent
        # amplifies tiny cross-parity roundoff. Keep their exact zero rows intact.
        groups=[list(range(p,degree+1,2))+list(range(degree+1+p*modes,degree+1+(p+1)*modes))
                for p in (0,1)]
        initial_conditions=[]
        # The analytic tails become extremely ill-conditioned as modes increase.
        # A fixed number of passes can leave O(1) Gram defects even though each
        # individual QR succeeds. Check the represented functions, not just R.
        self.orthogonality_tolerance=128*np.finfo(float).eps
        for iteration in range(12):
            for indices in groups:
                block=c[:,indices]
                _,r=la.qr(np.sqrt(w[:,None])*chebval(q,block).T,mode='economic')
                if iteration==0:initial_conditions.append(float(np.linalg.cond(r)))
                c[:,indices]=la.solve_triangular(r.T,block.T,lower=True).T
            if iteration>=2:
                sampled=np.sqrt(w[:,None])*chebval(q,c).T
                self.orthogonality_defect=float(np.max(abs(sampled.T@sampled-np.eye(self.n))))
                if self.orthogonality_defect<=self.orthogonality_tolerance:
                    break
        else:
            raise ValueError('Hybrid basis reorthogonalization did not converge')
        self.orthogonalization_passes=iteration+1
        self.tail_condition=max(initial_conditions)
        self.x=(self._symmetric_nodes(c) if nodes=='symmetric_qr' else
                np.r_[0.,(Legendre.basis(self.n-1).deriv().roots()+1)/2,1.])
        values=chebval(2*self.x-1,c).T
        self.nodal_condition=float(np.linalg.cond(values))
        c=la.solve(values.T,c.T).T
        self.coefficients=c
        self.derivatives=[chebder(c,m=k,axis=0)*2**k for k in range(3)]
        self.cardinal_defect=float(np.max(abs(self.evaluate(self.x,0)[0]-np.eye(self.n))))
        if self.cardinal_defect>1e-10 or np.max(abs(c[-8:]))>1e-14:
            raise ValueError('Unresolved or unstable hybrid basis')
        self.metadata=dict(modes=modes,period=period,degree=degree,n=self.n,
                           nodes=nodes,parity_preserved=True,
                           node_symmetry_defect=float(np.max(abs(self.x+self.x[::-1]-1))),
                           tail_condition=self.tail_condition,nodal_condition=self.nodal_condition,
                           orthogonalization_passes=self.orthogonalization_passes,
                           orthogonality_defect=self.orthogonality_defect,
                           orthogonality_tolerance=self.orthogonality_tolerance,
                           cardinal_defect=self.cardinal_defect)

    def _symmetric_nodes(self,c):
        # Constrain endpoints (and the center for odd n) before selecting pairs.
        # Project candidate evaluation rows onto the remaining coefficient space,
        # then greedily choose the pair with greatest 2-by-2 Gram determinant.
        odd=self.n%2
        fixed=np.array([-1.,1.,0.] if odd else [-1.,1.])
        orthogonal,_=la.qr(chebval(fixed,c),mode='full')
        null=orthogonal[:,len(fixed):]
        candidates=(1-np.cos(np.linspace(0,np.pi,max(2049,32*self.n+1))))/2
        left=candidates[(candidates>0)&(candidates<.5-1e-14)]
        a=chebval(2*left-1,c).T@null
        b=chebval(1-2*left,c).T@null
        chosen=[]
        for _ in range((self.n-len(fixed))//2):
            score=np.sum(a*a,axis=1)*np.sum(b*b,axis=1)-np.sum(a*b,axis=1)**2
            score[chosen]=-np.inf
            index=int(np.argmax(score))
            if not np.isfinite(score[index]) or score[index]<=0:
                raise ValueError('Symmetric node selection lost rank')
            chosen.append(index)
            directions,r=la.qr(np.column_stack((a[index],b[index])),mode='economic')
            if np.min(abs(np.diag(r)))<=np.finfo(float).eps*np.linalg.norm(r):
                raise ValueError('Symmetric node pair is numerically dependent')
            for _ in range(2):
                a-=(a@directions)@directions.T
                b-=(b@directions)@directions.T
        points=np.sort(left[chosen])
        return np.r_[0.,points,[.5] if odd else [],1-points[::-1],1.]
    def evaluate(self,x,order=2):
        if order not in (0,1,2): raise ValueError('derivative order must be 0, 1, or 2')
        return [chebval(2*np.asarray(x)-1,c).T for c in self.derivatives[:order+1]]
