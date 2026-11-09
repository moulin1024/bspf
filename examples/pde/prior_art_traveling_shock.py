# SPDX-License-Identifier: GPL-3.0-or-later
"""Research baselines: Legendre ESG-I adaptation and published FC-Gram data.

ESG-I: Chen & Shen (2018), doi:10.1007/s10915-018-0862-z.
FC: Albin & Bruno (2011), doi:10.1016/j.jcp.2011.04.023;
continuation matrices from neuraloperator/FC_PINO, FC_Gram_Construction.
These are local NS adaptations, not the authors' complete production solvers.
"""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
from numpy.polynomial.legendre import leggauss,legvander,legder,legval
from scipy.io import loadmat
from scipy.fft import rfft,irfft,rfftfreq
from scipy.signal import resample
from scipy.interpolate import CubicSpline
from viscous_traveling_shock import BeckerShock,primitives,fluxes


class LegendreBase:
    """Boundary lift plus homogeneous Shen basis P_j-P_{j+2}.

    Interior L2 whitening changes coordinates only. The existing enrichment
    and Schur machinery supplies an ESG-I-style NS adaptation, using exactly
    the same selected spline dictionary as BSPF at equal n.
    """
    def __init__(self,n,mu=.001,prandtl=.75,quadrature=32,line_options=None):
        self.n,self.mu,self.prandtl,self.gamma=n,mu,prandtl,1.4
        self.kappa=mu*3.5/prandtl;self.backend='dense'
        self.line_options=dict(basis='Shen-Legendre',degree=n-1)
        z,w=leggauss(quadrature)
        self.q=((np.arange(n-1)[:,None]+(z+1)/2)/(n-1)).ravel()
        self.weights=np.tile(w/(2*(n-1)),n-1)
        self.transform=np.eye(n-2)
        P=self.matrix(self.q)
        G=P[:,1:-1].T@(self.weights[:,None]*P[:,1:-1])
        v,V=eigh(G);self.transform=V/np.sqrt(v)
        self.P=self.matrix(self.q);self.Px=self.matrix(self.q,1)
        M=self.P.T@(self.weights[:,None]*self.P)
        K=self.Px.T@(self.weights[:,None]*self.Px)
        self.factors={False:cho_factor(M),True:cho_factor(M[1:-1,1:-1])}
        self.diffusion_radius=float(eigh(K,M,subset_by_index=[n-1,n-1],eigvals_only=True)[0])
        self.integral=self.weights@self.P
        loads=np.zeros((n,2));loads[[0,-1]]=np.eye(2)
        response=self.solve_mass(loads)
        self.wall_lift=np.linalg.solve(response[[0,-1]].T,response.T).T
        self.line=SimpleNamespace(evaluate=lambda x:SimpleNamespace(matrix=self.matrix(x)))
        self.galerkin=SimpleNamespace(fit_amplification=float(np.linalg.cond(self.P*np.sqrt(self.weights[:,None]))))

    def matrix(self,x,derivative=0):
        x=np.asarray(x);z=2*x-1
        if derivative:
            v=np.empty((len(x),self.n))
            for j in range(self.n):
                c=np.zeros(j+1);c[-1]=1
                v[:,j]=2*legval(z,legder(c))
            ends=np.column_stack((-np.ones(len(x)),np.ones(len(x))))
        else:
            v=legvander(z,self.n-1);ends=np.column_stack((1-x,x))
        return np.column_stack((ends[:,0],(v[:,:-2]-v[:,2:])@self.transform,ends[:,1]))

    def evaluate(self,c,derivative=0):return (self.Px if derivative else self.P)@c
    def project(self,v,derivative=0):return (self.Px if derivative else self.P).T@(self.weights[:,None]*v)
    def solve_mass(self,b,interior=False):return cho_solve(self.factors[interior],b,check_finite=False)


class FCGram:
    def __init__(self,n,data,filter_order=50):
        self.n=n;self.h=1/(n-1)
        matrices=loadmat(data);self.right=matrices['ArQr'];self.left=matrices['AlQl']
        self.C,self.d=self.right.shape;self.m=n+self.C;self.period=self.m*self.h
        self.k=2*np.pi*rfftfreq(self.m,self.h)
        self.filter=np.exp(-16*np.log(10)*(self.k/(np.pi/self.h))**(2*filter_order)) if filter_order else np.ones_like(self.k)

    def extend(self,v):return np.vstack((v,self.right@v[-self.d:]+self.left@v[:self.d]))
    def derivative(self,v):
        return irfft(rfft(self.extend(v),axis=0)*(1j*self.k*self.filter)[:,None],n=self.m,axis=0)[:self.n]

    def at(self,x,v):
        count=self.m*max(1,int(np.ceil(262144/self.m)))
        values=resample(self.extend(v),count,axis=0)
        return CubicSpline(np.arange(count)*self.period/count,values)(x)


def solve_fc(n,data,final_time=.05,cfl=.7,filter_order=50,output_points=64001,fused_flux=False,constant_boundaries=False):
    if fused_flux:
        from compiled_galerkin_step import physical_flux
    start=perf_counter();shock=BeckerShock();fc=FCGram(n,data,filter_order)
    x=np.linspace(0,1,n);U=shock.state(x);t=0.;steps=0
    if constant_boundaries:
        from compiled_galerkin_step import bounds
        endpoints=np.array([0.,1.]);boundary=shock.state(endpoints)
        for check_time in (0.,final_time/2,final_time):
            if (np.max(abs(shock.state(endpoints,check_time)-boundary))>1e-14 or
                    np.max(abs(shock.time_derivative(endpoints,check_time)))>1e-14):
                raise ValueError('Constant-boundary FC path requires stationary endpoint states')
    def rhs(values,time):
        values=values.copy();values[[0,-1]]=boundary if constant_boundaries else shock.state(np.array([0.,1.]),time)
        gradient=fc.derivative(values)
        if fused_flux:
            total_flux=physical_flux(values,gradient,shock.mu,shock.kappa,shock.gamma)
        else:
            convective,viscous=fluxes(values,gradient,shock.mu,shock.kappa,shock.gamma)
            total_flux=convective-viscous
        rate=-fc.derivative(total_flux)
        rate[[0,-1]]=0. if constant_boundaries else shock.time_derivative(np.array([0.,1.]),time)
        return rate
    setup=perf_counter()-start;begin=perf_counter()
    while t<final_time-1e-14:
        if constant_boundaries:
            speed,density=bounds(U,shock.gamma)
            if density<=0:raise ValueError('Nonphysical FC state')
        else:
            rho,u,p,_=primitives(U)
            speed=np.max(abs(u)+np.sqrt(1.4*p/rho));density=min(rho)
        diff=max(4*shock.mu/3,.4*shock.kappa)/density
        kmax=np.pi/fc.h
        dt=min(cfl/(speed*kmax/2.83+diff*kmax*kmax/2.7852935634),final_time-t)
        k1=rhs(U,t);k2=rhs(U+dt*k1/2,t+dt/2);k3=rhs(U+dt*k2/2,t+dt/2);k4=rhs(U+dt*k3,t+dt)
        U+=dt*(k1+2*k2+2*k3+k4)/6;t+=dt;steps+=1
        U[[0,-1]]=boundary if constant_boundaries else shock.state(np.array([0.,1.]),t)
        if not constant_boundaries:primitives(U)
    integration=perf_counter()-begin
    probes=np.linspace(0,1,output_points);out=fc.at(probes,U);exact=shock.state(probes,t)
    errors={key:dict(linf=float(max(abs(a-b)))) for key,a,b in zip(('rho','u','p','T'),primitives(out),primitives(exact))}
    summary=dict(method='FC-Gram',n=n,fft_points=fc.m,matching_points=fc.d,continuation_points=fc.C,
                 filter_order=filter_order,fused_flux=fused_flux,constant_boundaries=constant_boundaries,time=t,cfl=cfl,steps=steps,setup_seconds=setup,
                 integration_seconds=integration,seconds=perf_counter()-start,metrics=errors)
    return dict(summary=summary,x=probes,output=out,exact=exact)
