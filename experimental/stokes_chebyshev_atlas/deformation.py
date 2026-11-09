"""Low-degree coordinate corrections on a separately resolved geometry map."""
import numpy as np
from .spectral import transform2,evaluate2
from .topology import nodal_layout


class ReducedDeformation:
    """Restrict a full atlas objective to conforming polynomial displacements.

    The initial geometry is retained exactly. Only the displacement has reduced
    degree, with zero physical-wall values. Shared traces stay compatible.
    """
    def __init__(self,objective,degree):
        self.full=objective
        self.atlas=objective.atlas
        self.base=self.atlas.positions.copy()
        self.length=objective.length
        self.n=degree+1
        p,self.ids,self.free,_,_=nodal_layout(self.atlas.seed,self.n)
        self.low_count=len(p)
        self.counts=np.bincount(self.atlas.ids.ravel(),minlength=len(self.base))
        self.counts=np.maximum(self.counts,1)

    def pack(self,positions):
        if not np.array_equal(positions,self.base):
            raise ValueError('Reduced deformation pack expects the initial geometry')
        return np.zeros(2*np.count_nonzero(self.free))

    def unpack(self,x):
        low=np.zeros(self.low_count,complex);k=np.count_nonzero(self.free)
        low[self.free]=self.length*(x[:k]+1j*x[k:])
        blocks=evaluate2(transform2(low[self.ids]),self.atlas.degree+1)
        delta=np.zeros_like(self.base)
        np.add.at(delta,self.atlas.ids.ravel(),blocks.ravel())
        return self.base+delta/self.counts

    def __call__(self,x):
        value,g=self.full(self.full.pack(self.unpack(x)))
        if not np.isfinite(value):return value,np.zeros_like(x)
        fullg=np.zeros_like(self.base);k=np.count_nonzero(self.atlas.free)
        fullg[self.atlas.free]=g[:k]+1j*g[k:]
        gc=evaluate2((fullg/self.counts)[self.atlas.ids],self.n,adjoint=True)
        blocks=transform2(gc,adjoint=True)
        lowg=np.zeros(self.low_count,complex)
        np.add.at(lowg,self.ids.ravel(),blocks.ravel())
        gg=lowg[self.free]
        return value,np.r_[gg.real,gg.imag]
