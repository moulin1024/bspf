# SPDX-License-Identifier: GPL-3.0-or-later
"""Characteristic finite-difference WENO-Z5 / sixth-order viscosity / RK4.

Roe characteristic basis at each face, global Lax-Friedrichs flux splitting,
WENO-Z weights with power 2 and epsilon=1e-40. Nodal conservative unknowns.
"""
from time import perf_counter
import numpy as np
from numba import njit
from scipy.interpolate import make_interp_spline
from viscous_traveling_shock import BeckerShock, primitives

@njit(cache=True)
def wz(a,b,c,d,e):
    b0=13/12*(a-2*b+c)**2+.25*(a-4*b+3*c)**2
    b1=13/12*(b-2*c+d)**2+.25*(b-d)**2
    b2=13/12*(c-2*d+e)**2+.25*(3*c-4*d+e)**2
    tau=abs(b0-b2)
    a0=.1*(1+(tau/(b0+1e-40))**2)
    a1=.6*(1+(tau/(b1+1e-40))**2)
    a2=.3*(1+(tau/(b2+1e-40))**2)
    return (a0*(2*a-7*b+11*c)+a1*(-b+5*c+2*d)+a2*(2*c+5*d-e))/(6*(a0+a1+a2))

@njit(cache=True)
def d6(v,i,dx):
    return (-v[i-3]+9*v[i-2]-45*v[i-1]+45*v[i+1]-9*v[i+2]+v[i+3])/(60*dx)

@njit(cache=True)
def wave_bounds(U,mu,kappa):
    speed=0.;diff=0.
    for i in range(len(U)):
        r=U[i,0];u=U[i,1]/r;p=.4*(U[i,2]-.5*r*u*u)
        if r<=0 or p<=0:raise ValueError('nonphysical WENO state')
        speed=max(speed,abs(u)+np.sqrt(1.4*p/r))
        diff=max(diff,max(4*mu/3,.4*kappa)/r)
    return speed,diff

@njit(cache=True)
def rhs(U,dx,mu,kappa):
    n=len(U);ext=np.empty((n+12,3));ext[6:-6]=U
    for i in range(6):ext[i]=U[0];ext[n+6+i]=U[-1]
    vel=ext[:,1]/ext[:,0]
    pressure=.4*(ext[:,2]-.5*ext[:,1]*vel);temp=pressure/ext[:,0]
    enthalpy=(ext[:,2]+pressure)/ext[:,0]
    F=np.empty_like(ext)
    F[:,0]=ext[:,1];F[:,1]=ext[:,1]*vel+pressure;F[:,2]=vel*(ext[:,2]+pressure)
    speed,_=wave_bounds(U,mu,kappa)
    faces=np.empty((n+1,3))
    for f in range(n+1):
        i=f+5
        rl=np.sqrt(ext[i,0]);rr=np.sqrt(ext[i+1,0])
        u=(rl*vel[i]+rr*vel[i+1])/(rl+rr)
        H=(rl*enthalpy[i]+rr*enthalpy[i+1])/(rl+rr)
        c=np.sqrt(.4*(H-.5*u*u))
        plus=np.empty((3,6));minus=np.empty((3,6))
        for k in range(6):
            j=i-2+k
            for sign in range(2):
                fac=1. if sign==0 else -1.
                v0=.5*(F[j,0]+fac*speed*ext[j,0])
                v1=.5*(F[j,1]+fac*speed*ext[j,1])
                v2=.5*(F[j,2]+fac*speed*ext[j,2])
                contact=v0-.4/(c*c)*(v2-u*v1+.5*u*u*v0)
                acoustic=v0-contact;momentum=(v1-u*v0)/c
                if sign==0:
                    plus[0,k]=.5*(acoustic-momentum);plus[1,k]=contact;plus[2,k]=.5*(acoustic+momentum)
                else:
                    minus[0,k]=.5*(acoustic-momentum);minus[1,k]=contact;minus[2,k]=.5*(acoustic+momentum)
        h=np.empty(3)
        for k in range(3):
            h[k]=wz(plus[k,0],plus[k,1],plus[k,2],plus[k,3],plus[k,4])+wz(minus[k,5],minus[k,4],minus[k,3],minus[k,2],minus[k,1])
        faces[f,0]=h[0]+h[1]+h[2]
        faces[f,1]=(u-c)*h[0]+u*h[1]+(u+c)*h[2]
        faces[f,2]=(H-u*c)*h[0]+.5*u*u*h[1]+(H+u*c)*h[2]
    visc=np.zeros((n+6,3))
    for j in range(n+6):
        i=j+3;tau=4*mu/3*d6(vel,i,dx)
        visc[j,1]=tau;visc[j,2]=vel[i]*tau+kappa*d6(temp,i,dx)
    rate=-(faces[1:]-faces[:-1])/dx
    for i in range(n):
        for k in range(1,3):rate[i,k]+=d6(visc[:,k],i+3,dx)
    rate[0]=0.;rate[-1]=0.
    return rate

@njit(cache=True)
def integrate(U,dx,mu,kappa,time,cfl):
    t=0.;steps=0
    while t<time:
        speed,diff=wave_bounds(U,mu,kappa)
        dt=min(cfl/(speed/dx+3*diff/(dx*dx)),time-t)
        k1=rhs(U,dx,mu,kappa);k2=rhs(U+dt*k1/2,dx,mu,kappa)
        k3=rhs(U+dt*k2/2,dx,mu,kappa);k4=rhs(U+dt*k3,dx,mu,kappa)
        U=U+dt*(k1+2*k2+2*k3+k4)/6;t+=dt;steps+=1
    return U,steps

def solve(n,cfl=.7,final_time=.05):
    shock=BeckerShock();nodes=np.linspace(0,1,n);U=shock.state(nodes)
    integrate(U[:8].copy(),1/7,shock.mu,shock.kappa,0.,cfl) # exclude JIT
    start=perf_counter()
    U,steps=integrate(U,1/(n-1),shock.mu,shock.kappa,final_time,cfl)
    seconds=perf_counter()-start
    x=np.linspace(0,1,64001)
    # Sixth-order diagnostic interpolation; does not enter the evolution.
    output=make_interp_spline(nodes,U,k=5)(x)
    exact=shock.state(x,final_time)
    def metrics(a,b,grid):
        return {k:dict(linf=float(np.max(abs(v-z))),l2=float(np.sqrt(np.trapezoid((v-z)**2,grid))))
                for k,v,z in zip(('rho','u','p','T'),primitives(a),primitives(b))}
    report=dict(n=n,cfl=cfl,time=final_time,mu=shock.mu,prandtl=shock.prandtl,
        steps=steps,seconds=seconds,metrics=metrics(output,exact,x),
        nodal_metrics=metrics(U,shock.state(nodes,final_time),nodes))
    return report,dict(x=x,output=output,exact=exact,nodes=nodes,state=U)
