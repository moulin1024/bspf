"""Localized scalar Poisson MMS features; the underlying velocity is unchanged."""
from pathlib import Path
import numpy as np

KEYS=('gaussian_centers','gaussian_sigmas','gaussian_amplitudes',
      'tanh_centers','tanh_radii','tanh_widths','tanh_amplitudes')


class PoissonFeatures:
    def __init__(self,parameters,xp=np,device=None):
        self.xp=xp;self.device=device
        arrays={k:np.asarray(parameters[k],dtype=np.float64) for k in KEYS}
        ng=len(arrays['gaussian_amplitudes']);nt=len(arrays['tanh_amplitudes'])
        shapes=((ng,3),(ng,3),(ng,),(nt,3),(nt,),(nt,),(nt,))
        for k,shape in zip(KEYS,shapes):
            if arrays[k].shape!=shape or not np.isfinite(arrays[k]).all():raise ValueError(f'Invalid {k}')
        for k in ('gaussian_sigmas','tanh_radii','tanh_widths'):
            if np.any(arrays[k]<=0):raise ValueError(f'{k} must be positive')
        convert=lambda x:xp.asarray(x,dtype=xp.float64) if device is None else xp.as_tensor(x,dtype=xp.float64,device=device)
        for k,v in arrays.items():setattr(self,k,convert(v))
        self.active=bool(ng or nt)

    @classmethod
    def load(cls,path,xp=np,device=None):
        with np.load(Path(path)) as d:
            if not any(k in d for k in KEYS):
                parameters={k:np.empty((0,3) if k.endswith(('centers','sigmas')) else (0,)) for k in KEYS}
            else:
                if not all(k in d for k in KEYS):raise ValueError('Incomplete Poisson feature metadata')
                parameters={k:d[k] for k in KEYS}
        return cls(parameters,xp,device)

    def evaluate(self,points,components=False):
        """Return (phi_extra, -physical_laplacian(phi_extra)); NumPy/CuPy/Torch."""
        xp=self.xp
        points=xp.asarray(points,dtype=xp.float64) if self.device is None else xp.as_tensor(points,dtype=xp.float64,device=self.device)
        g=points[...,0]*0;fg=g*0;t=g*0;ft=g*0
        for center,sigma,amplitude in zip(self.gaussian_centers,self.gaussian_sigmas,self.gaussian_amplitudes):
            delta=points-center;inv=1/(sigma*sigma)
            value=amplitude*xp.exp(-.5*(delta*delta*inv).sum(-1))
            g=g+value;fg=fg+value*(inv-delta*delta*inv*inv).sum(-1)
        for center,radius,width,amplitude in zip(self.tanh_centers,self.tanh_radii,self.tanh_widths,self.tanh_amplitudes):
            r2=((points-center)**2).sum(-1);scale=2*radius*width
            v=xp.tanh((r2-radius*radius)/scale);sech2=1-v*v
            t=t+.5*amplitude*(1-v)
            ft=ft+amplitude*sech2*(3/scale-4*r2*v/(scale*scale))
        if components:return dict(gaussian=g,gaussian_forcing=fg,tanh=t,tanh_forcing=ft)
        return g+t,fg+ft
