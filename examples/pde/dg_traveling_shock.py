# SPDX-License-Identifier: GPL-3.0-or-later
"""Overintegrated modal Legendre DG, HLLC convection, BR1 viscosity and RK4.

Uniform elements; auxiliary gradients of velocity and temperature; central BR1
traces. No filtering, artificial viscosity or subcell blending. This is not
claimed to be the entropy-stable split-form DGSEM variant.
"""
from time import perf_counter
import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss,legvander,legder,legval
from viscous_traveling_shock import BeckerShock,primitives
from muscl_traveling_shock import hllc

def operators(p,q=None):
    q=2*p+3 if q is None else q
    z,w=leggauss(q);V=legvander(z,p);D=np.empty_like(V)
    for j in range(p+1):
        c=np.zeros(p+1);c[j]=1;D[:,j]=legval(z,legder(c))
    E=legvander(np.array([-1.,1.]),p)
    K=D.T@(w[:,None]*V);scale=np.diag(2*np.arange(p+1)+1)
    ra=rd=0.
    for theta in np.linspace(0,np.pi,257):
        r,l=E[1],E[0];phase=np.exp(1j*theta)
        G=scale@(.5*np.outer(r,r+phase*l)-.5*np.outer(l,l+r/phase)-K)
        A=scale@(np.outer(r,r)-np.outer(l,r)/phase-K)
        ra=max(ra,float(max(abs(np.linalg.eigvals(A)))))
        rd=max(rd,float(max(abs(np.linalg.eigvals(G))))**2)
    return z,w,np.ascontiguousarray(V),np.ascontiguousarray(D*w[:,None]),E,ra,rd

@njit(cache=True)
def evaluate(U,V):
    ne,m,nf=U.shape;out=np.zeros((ne,len(V),nf))
    for e in range(ne):
        for q in range(len(V)):
            for j in range(m):
                for k in range(nf):out[e,q,k]+=V[q,j]*U[e,j,k]
    return out

@njit(cache=True)
def primitive(U):
    r=U[0];u=U[1]/r;p=.4*(U[2]-.5*r*u*u)
    if r<=0 or p<=0:raise ValueError('Nonphysical DG state')
    return np.array([r,u,p])

@njit(cache=True)
def rhs(U,h,mu,kappa,V,DW,E,left,right):
    ne,m,_=U.shape;nq=len(V)
    values=evaluate(U,V);traces=evaluate(U,E)
    fields=np.empty((ne,nq,2));edge=np.empty((ne,2,3));speed=0.;diff=0.
    for e in range(ne):
        for q in range(nq):
            w=primitive(values[e,q]);fields[e,q,0]=w[1];fields[e,q,1]=w[2]/w[0]
            speed=max(speed,abs(w[1])+np.sqrt(1.4*w[2]/w[0]))
            diff=max(diff,max(4*mu/3,.4*kappa)/w[0])
        for side in range(2):edge[e,side]=primitive(traces[e,side])
    face_values=np.empty((ne+1,2))
    for f in range(ne+1):
        a=left if f==0 else edge[f-1,1]
        b=right if f==ne else edge[f,0]
        face_values[f,0]=.5*(a[1]+b[1]);face_values[f,1]=.5*(a[2]/a[0]+b[2]/b[0])
    grad=np.zeros((ne,m,2))
    for e in range(ne):
        for j in range(m):
            for k in range(2):
                v=E[1,j]*face_values[e+1,k]-E[0,j]*face_values[e,k]
                for q in range(nq):v-=DW[q,j]*fields[e,q,k]
                grad[e,j,k]=(2*j+1)/h*v
    gq=evaluate(grad,V);ge=evaluate(grad,E)
    volume=np.empty((ne,nq,3));visc_edge=np.zeros((ne,2,3))
    for e in range(ne):
        for q in range(nq):
            r=values[e,q,0];u=fields[e,q,0];p=r*fields[e,q,1];tau=4*mu/3*gq[e,q,0]
            volume[e,q,0]=values[e,q,1]
            volume[e,q,1]=values[e,q,1]*u+p-tau
            volume[e,q,2]=u*(values[e,q,2]+p)-u*tau-kappa*gq[e,q,1]
        for side in range(2):
            tau=4*mu/3*ge[e,side,0]
            visc_edge[e,side,1]=tau
            visc_edge[e,side,2]=edge[e,side,1]*tau+kappa*ge[e,side,1]
    flux=np.empty((ne+1,3))
    for f in range(ne+1):
        a=left if f==0 else edge[f-1,1]
        b=right if f==ne else edge[f,0]
        flux[f]=hllc(a,b)
        if f==0:flux[f]-=.5*visc_edge[0,0]
        elif f==ne:flux[f]-=.5*visc_edge[-1,1]
        else:flux[f]-=.5*(visc_edge[f-1,1]+visc_edge[f,0])
    rate=np.zeros_like(U)
    for e in range(ne):
        for j in range(m):
            for k in range(3):
                v=E[0,j]*flux[e,k]-E[1,j]*flux[e+1,k]
                for q in range(nq):v+=DW[q,j]*volume[e,q,k]
                rate[e,j,k]=(2*j+1)/h*v
    return rate,speed,diff

@njit(cache=True)
def integrate(U,h,mu,kappa,V,DW,E,left,right,ra,rd,time,cfl):
    t=0.;steps=0
    while t<time:
        k1,a,d=rhs(U,h,mu,kappa,V,DW,E,left,right)
        dt=min(cfl/(a*ra/(2*h)+d*rd/(2.785293563405282*h*h)),time-t)
        k2,_,_=rhs(U+dt*k1/2,h,mu,kappa,V,DW,E,left,right)
        k3,_,_=rhs(U+dt*k2/2,h,mu,kappa,V,DW,E,left,right)
        k4,_,_=rhs(U+dt*k3,h,mu,kappa,V,DW,E,left,right)
        U+=dt*(k1+2*k2+2*k3+k4)/6;t+=dt;steps+=1
    return U,steps

def sample(U,x):
    ne,m,_=U.shape;i=np.minimum((x*ne).astype(int),ne-1)
    z=2*(x*ne-i)-1
    return np.einsum('ij,ijk->ik',legvander(z,m-1),U[i])

def solve(ne,p,cfl=.7,final_time=.05,quadrature=None):
    shock=BeckerShock();h=1/ne
    # Compile all kernels before timing; warm call does not evolve a state.
    z,w,V,DW,E,ra,rd=operators(p,quadrature)
    left=np.array([1.,shock.v0+shock.speed,1.]);right=np.array([2.,shock.v1+shock.speed,2.75])
    integrate(np.zeros((1,p+1,3)),1.,shock.mu,shock.kappa,V,DW,E,left,right,ra,rd,0.,cfl)
    start=perf_counter()
    # Operator construction is part of end-to-end timing, even after warmup.
    z,w,V,DW,E,ra,rd=operators(p,quadrature)
    points=(np.arange(ne)[:,None]+(z[None,:]+1)/2)*h
    exact=shock.state(points.ravel()).reshape(ne,len(z),3)
    U=np.einsum('qj,eqk->ejk',w[:,None]*V,exact)*(2*np.arange(p+1)+1)[None,:,None]/2
    t0=perf_counter();U,steps=integrate(U,h,shock.mu,shock.kappa,V,DW,E,left,right,ra,rd,final_time,cfl)
    integration=perf_counter()-t0
    x=np.linspace(0,1,64001);output=sample(U,x);target=shock.state(x,final_time)
    metrics={k:dict(linf=float(np.max(abs(a-b))),l2=float(np.sqrt(np.trapezoid((a-b)**2,x))))
             for k,a,b in zip(('rho','u','p','T'),primitives(output),primitives(target))}
    # Both traces are checked since DG output can be discontinuous at faces.
    faces=np.arange(ne+1)/ne
    trace_values=evaluate(U,E).reshape(-1,3)
    trace_exact=shock.state(np.column_stack((faces[:-1],faces[1:])).ravel(),final_time)
    trace_errors={k:float(np.max(abs(a-b))) for k,a,b in zip(('rho','u','p','T'),primitives(trace_values),primitives(trace_exact))}
    report=dict(elements=ne,degree=p,modes=ne*(p+1),cfl=cfl,time=final_time,quadrature=len(z),steps=steps,
        integration_seconds=integration,wall_seconds=perf_counter()-start,metrics=metrics,trace_linf=trace_errors,
        worst_error=max(max(v['linf'] for v in metrics.values()),max(trace_errors.values())),advection_radius=ra,diffusion_radius=rd)
    return report,dict(x=x,output=output,exact=target,coefficients=U)
