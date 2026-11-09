# SPDX-License-Identifier: GPL-3.0-or-later
"""Uniform-grid MUSCL-MC/HLLC, centered physical diffusion and SSPRK2.

Conservative cell averages; primitive-variable limited linear reconstruction.
No extra artificial viscosity. Numba compiles the explicit integration loop.
"""
from time import perf_counter
import numpy as np
from numba import njit
from viscous_traveling_shock import BeckerShock, primitives

@njit(cache=True)
def mc(a,b):
    if a*b<=0: return 0.
    return np.sign(a)*min(2*abs(a),2*abs(b),.5*abs(a+b))

@njit(cache=True)
def cons(w):
    r,u,p=w
    return np.array([r,r*u,p/.4+.5*r*u*u])

@njit(cache=True)
def hllc(l,r):
    rl,ul,pl=l;rr,ur,pr=r
    cl=np.sqrt(1.4*pl/rl);cr=np.sqrt(1.4*pr/rr)
    sl=min(ul-cl,ur-cr);sr=max(ul+cl,ur+cr)
    U=cons(l);V=cons(r)
    fl=np.array([U[1],U[1]*ul+pl,ul*(U[2]+pl)])
    fr=np.array([V[1],V[1]*ur+pr,ur*(V[2]+pr)])
    if sl>=0:return fl
    if sr<=0:return fr
    sm=(pr-pl+rl*ul*(sl-ul)-rr*ur*(sr-ur))/(rl*(sl-ul)-rr*(sr-ur))
    if sm>=0:
        rs=rl*(sl-ul)/(sl-sm)
        star=np.array([rs,rs*sm,rs*(U[2]/rl+(sm-ul)*(sm+pl/(rl*(sl-ul))))])
        return fl+sl*(star-U)
    rs=rr*(sr-ur)/(sr-sm)
    star=np.array([rs,rs*sm,rs*(V[2]/rr+(sm-ur)*(sm+pr/(rr*(sr-ur))))])
    return fr+sr*(star-V)

@njit(cache=True)
def primitive_array(U):
    w=np.empty_like(U)
    for i in range(len(U)):
        rho=U[i,0];u=U[i,1]/rho;p=.4*(U[i,2]-.5*rho*u*u)
        if rho<=0 or p<=0:raise ValueError('nonphysical MUSCL state')
        w[i,0]=rho;w[i,1]=u;w[i,2]=p
    return w

@njit(cache=True)
def slopes(w):
    s=np.zeros_like(w)
    for i in range(1,len(w)-1):
        for j in range(3):s[i,j]=mc(w[i,j]-w[i-1,j],w[i+1,j]-w[i,j])
    return s

@njit(cache=True)
def rhs(U,dx,mu,kappa,left,right):
    n=len(U);ext=np.empty((n+4,3));ext[2:-2]=U
    ext[0]=left;ext[1]=left;ext[-2]=right;ext[-1]=right
    w=primitive_array(ext);s=slopes(w);flux=np.empty((n+1,3))
    for f in range(n+1):
        i=f+1;l=w[i]+.5*s[i];r=w[i+1]-.5*s[i+1]
        flux[f]=hllc(l,r)
        tau=4*mu/3*(w[i+1,1]-w[i,1])/dx
        tx=(w[i+1,2]/w[i+1,0]-w[i,2]/w[i,0])/dx
        flux[f,1]-=tau
        flux[f,2]-=.5*(w[i,1]+w[i+1,1])*tau+kappa*tx
    return -(flux[1:]-flux[:-1])/dx

@njit(cache=True)
def integrate(U,dx,mu,kappa,left,right,final_time,cfl):
    t=0.;steps=0
    while t<final_time:
        w=primitive_array(U);speed=0.;diff=0.
        for i in range(len(U)):
            speed=max(speed,abs(w[i,1])+np.sqrt(1.4*w[i,2]/w[i,0]))
            diff=max(diff,max(4*mu/3,.4*kappa)/w[i,0])
        dt=min(cfl/(speed/dx+2*diff/(dx*dx)),final_time-t)
        stage=U+dt*rhs(U,dx,mu,kappa,left,right)
        U=.5*U+.5*(stage+dt*rhs(stage,dx,mu,kappa,left,right))
        t+=dt;steps+=1
    return U,steps

def averages(shock,n,t,order=16):
    z,w=np.polynomial.legendre.leggauss(order)
    x=(np.arange(n)+.5)/n
    points=x[:,None]+z[None,:]/(2*n)
    return (shock.state(points.ravel(),t).reshape(n,order,3)*w[None,:,None]).sum(axis=1)/2

def reconstruct(U,x):
    w=primitive_array(U);ext=np.vstack((w[0],w,w[-1]));s=slopes(ext)[1:-1]
    n=len(U);index=np.minimum((x*n).astype(int),n-1)
    values=w[index]+(x*n-index-.5)[:,None]*s[index]
    return np.column_stack((values[:,0],values[:,1],values[:,2],values[:,2]/values[:,0]))

def solve(n,cfl=.4,final_time=.05):
    shock=BeckerShock();dx=1/n
    U=averages(shock,n,0)
    # Both boundaries lie hundreds of shock widths away. These far-field
    # states equal the analytic boundary/ghost states to double precision.
    left=shock.state(np.array([0.]))[0];right=shock.state(np.array([1.]))[0]
    integrate(U[:4].copy(),.25,shock.mu,shock.kappa,left,right,0.,cfl) # JIT excluded
    start=perf_counter();U,steps=integrate(U,dx,shock.mu,shock.kappa,left,right,final_time,cfl)
    seconds=perf_counter()-start
    exact_avg=averages(shock,n,final_time)
    avg_error=U-exact_avg
    x=np.linspace(0,1,64001);values=reconstruct(U,x);exact=np.column_stack(primitives(shock.state(x,final_time)))
    metrics={k:dict(linf=float(np.max(abs(values[:,j]-exact[:,j]))),
        l2=float(np.sqrt(np.trapezoid((values[:,j]-exact[:,j])**2,x)))) for j,k in enumerate(('rho','u','p','T'))}
    return dict(n=n,cfl=cfl,time=final_time,steps=steps,seconds=seconds,metrics=metrics,
                cell_average_linf=np.max(abs(avg_error),axis=0).tolist()),dict(x=x,output=values,exact=exact,averages=U)
