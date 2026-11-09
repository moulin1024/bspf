"""FFT-adjoint barrier for unresolved transformed mass/stiffness coefficients."""
import numpy as np
from .spectral import transform2, evaluate2, derivative


class OperatorBarrier:
    def __init__(self, coefficients, settings):
        self.n = coefficients.shape[-1]
        self.m = max(3*self.n+2, 2*settings.solution_degree+5)
        self.weight = settings.operator_barrier_weight / len(coefficients)
        self.mask = np.maximum.outer(np.arange(self.m), np.arange(self.m)) >= settings.solution_degree+1
        _, _, oc = self.fields(coefficients)
        if oc is None:
            raise ValueError('Initial polynomial atlas folds on operator barrier grid')
        self.bounds = [np.maximum(settings.spectral_tolerance**2, 1.05*self.ratio(v)[0])
                       for v in (oc[:, :1], oc[:, 1:])]

    def fields(self, c):
        a = evaluate2(derivative(c,-2),self.m)
        b = evaluate2(derivative(c,-1),self.m)
        d = np.imag(np.conj(a)*b)
        if np.min(d) <= 0:
            return a,b,None
        oc = transform2(np.stack([d,abs(b)**2/d,-np.real(np.conj(a)*b)/d,abs(a)**2/d],axis=1))
        return a,b,oc

    def ratio(self,c):
        total = np.maximum(np.sum(c*c,axis=(1,2,3)),1e-300)
        high = np.sum((c*c)*self.mask,axis=(1,2,3))
        return high/total,total

    def __call__(self,c):
        a,b,oc = self.fields(c)
        if oc is None:
            return np.inf,np.zeros_like(c)
        value = 0.
        grads = []
        for v,bound in zip((oc[:,:1],oc[:,1:]),self.bounds):
            ratio,total = self.ratio(v)
            if np.any(ratio >= bound):
                return np.inf,np.zeros_like(c)
            value -= self.weight*np.sum(np.log1p(-ratio/bound))
            factor = 2*self.weight/((bound-ratio)*total)
            grads.append(factor[:,None,None,None]*(self.mask-ratio[:,None,None,None])*v)
        g0,g1,g2,g3 = np.moveaxis(transform2(np.concatenate(grads,axis=1),adjoint=True),1,0)
        d = np.imag(np.conj(a)*b)
        cd = g0-(g1*abs(b)**2-g2*np.real(np.conj(a)*b)+g3*abs(a)**2)/d**2
        ga = (2*g3*a-g2*b)/d + cd*(b.imag-1j*b.real)
        gb = (2*g1*b-g2*a)/d + cd*(-a.imag+1j*a.real)
        gc = derivative(evaluate2(ga,self.n,adjoint=True),-2,adjoint=True)
        gc += derivative(evaluate2(gb,self.n,adjoint=True),-1,adjoint=True)
        return float(value),gc
