"""PyTorch CUDA FC-Gram strong-form Poisson with FP32 fast diagonalization."""
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
import tensorly as tl
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from fourier_continuation import FCGram

tl.set_backend('pytorch')

class FCOperator:
    def __init__(self,n,metric,drift,degree=8,device='cuda'):
        if n<2*degree:raise ValueError('Grid must have at least twice the matching degree')
        self.n=n;self.m=n-2;self.device=device;self.degree=degree
        self.fc=FCGram(d=degree,n_additional_pts=50).to(device=device,dtype=torch.float64)
        # Cache the public implementation's extension matrix without changing it.
        dummy=torch.zeros(n,device=device,dtype=torch.float64)
        self.E=self.fc._get_extension_matrix_for_axis(dummy,0)
        self.freq=2*torch.pi*torch.fft.fftfreq(n+50,d=1/(n-1),device=device,dtype=torch.float64)
        self.metric=torch.as_tensor(metric,device=device,dtype=torch.float64)
        self.drift=torch.as_tensor(drift,device=device,dtype=torch.float64)
        self.interior=(slice(1,-1),)*3
        # Build the exact reference FC second-derivative matrix in CPU FP64.
        E=self.E.cpu().numpy();freq=2*np.pi*np.fft.fftfreq(n+50,d=1/(n-1))
        D2=np.fft.ifft(-freq[:,None]**2*np.fft.fft(E,axis=0),axis=0).real[25:25+n]
        self.D1_cpu=np.fft.ifft((1j*freq)[:,None]*np.fft.fft(E,axis=0),axis=0).real[25:25+n]
        self.D2_cpu=D2
        lam,V=la.eig(-D2[1:-1,1:-1]);Vi=la.inv(V)
        if lam.real.min()<=0:raise RuntimeError('Reference FC operator is not positive stable')
        self.eigenvector_condition=float(np.linalg.cond(V))
        self.eigenpair_defect=float(la.norm(-D2[1:-1,1:-1]@V-V*lam)/la.norm(D2[1:-1,1:-1]@V))
        self.V=torch.as_tensor(V,device=device,dtype=torch.complex64)
        self.Vi=torch.as_tensor(Vi,device=device,dtype=torch.complex64)
        a=self.metric.mean(dim=(0,1,2)).cpu().numpy()
        # Spatial array axes z,y,x; geometry components x,y,z.
        denominator=a[0]*lam[None,None,:]+a[1]*lam[None,:,None]+a[2]*lam[:,None,None]
        self.denominator=torch.as_tensor(denominator,device=device,dtype=torch.complex64)

    def derivative(self,u,axis,order):
        rows=torch.movedim(u,axis,-1)
        extended=rows@self.E.T
        spectrum=torch.fft.fft(extended,dim=-1)
        factor=1j*self.freq if order==1 else -self.freq**2
        values=torch.fft.ifft(spectrum*factor,dim=-1).real[...,25:25+self.n]
        return torch.movedim(values,-1,axis)

    def full(self,u):
        sl=self.interior
        grad=[self.derivative(u,2-i,1) for i in range(3)]
        result=torch.zeros((self.m,)*3,device=self.device,dtype=torch.float64)
        for i in range(3):
            result-=self.metric[...,i]*self.derivative(u,2-i,2)[sl]+self.drift[...,i]*grad[i][sl]
        c=3
        for i in range(3):
            for j in range(i+1,3):
                result-=2*self.metric[...,c]*self.derivative(grad[j],2-i,1)[sl];c+=1
        return result.reshape(-1)

    def __call__(self,x):
        u=torch.zeros((self.n,)*3,device=self.device,dtype=torch.float64)
        u[self.interior]=x.reshape((self.m,)*3)
        return self.full(u)

    @staticmethod
    def transform(x,matrix):
        for axis in range(3):
            x=torch.movedim(torch.movedim(x,axis,-1)@matrix.T,-1,axis)
        return x

    def precondition(self,r):
        scale=torch.linalg.vector_norm(r)
        if scale.item()==0:return torch.zeros_like(r)
        c=(r/scale).reshape((self.m,)*3).to(torch.complex64)
        c=self.transform(c,self.Vi)/self.denominator
        return self.transform(c,self.V).real.to(torch.float64).reshape(-1)*scale

    def cpu_full(self,u):
        def diff(v,axis,D):return np.moveaxis(np.moveaxis(v,axis,-1)@D.T,-1,axis)
        grad=[diff(u,2-i,self.D1_cpu) for i in range(3)];sl=self.interior
        g=self.metric.cpu().numpy();b=self.drift.cpu().numpy();out=np.zeros((self.m,)*3)
        for i in range(3):out-=g[...,i]*diff(u,2-i,self.D2_cpu)[sl]+b[...,i]*grad[i][sl]
        c=3
        for i in range(3):
            for j in range(i+1,3):out-=2*g[...,c]*diff(grad[j],2-i,self.D1_cpu)[sl];c+=1
        return out.reshape(-1)


def fgmres(A,b,rtol=1e-12,restart=40,maxiter=400):
    """Right-preconditioned FGMRES, two-pass CGS, true residual at restarts."""
    x=torch.zeros_like(b);bnorm=torch.linalg.vector_norm(b).item()
    if bnorm==0:return x,dict(iterations=0,true_residual=0.,restarts=0)
    N=len(b);V=torch.empty((N,restart+1),device=b.device,dtype=b.dtype);Z=torch.empty((N,restart),device=b.device,dtype=b.dtype)
    iterations=0;cycles=0
    while iterations<maxiter:
        r=b-A(x);beta=torch.linalg.vector_norm(r).item()
        if beta/bnorm<=rtol:return x,dict(iterations=iterations,true_residual=beta/bnorm,restarts=cycles)
        V[:,0]=r/beta;H=np.zeros((restart+1,restart));g=np.zeros(restart+1);g[0]=beta
        cs=np.zeros(restart);sn=np.zeros(restart);used=0
        for j in range(min(restart,maxiter-iterations)):
            Z[:,j]=A.precondition(V[:,j]);w=A(Z[:,j])
            for _ in range(2):
                h=V[:,:j+1].T@w;w-=V[:,:j+1]@h;H[:j+1,j]+=h.cpu().numpy()
            H[j+1,j]=torch.linalg.vector_norm(w).item()
            if H[j+1,j]>0:V[:,j+1]=w/H[j+1,j]
            for i in range(j):H[i,j],H[i+1,j]=cs[i]*H[i,j]+sn[i]*H[i+1,j],-sn[i]*H[i,j]+cs[i]*H[i+1,j]
            rho=np.hypot(H[j,j],H[j+1,j])
            if rho==0:raise RuntimeError('FGMRES breakdown')
            cs[j]=H[j,j]/rho;sn[j]=H[j+1,j]/rho;H[j,j]=rho;H[j+1,j]=0
            g[j+1]=-sn[j]*g[j];g[j]*=cs[j];iterations+=1;used=j+1
            if abs(g[j+1])/bnorm<=rtol:break
        y=la.solve_triangular(H[:used,:used],g[:used])
        x+=Z[:,:used]@torch.as_tensor(y,device=b.device,dtype=b.dtype);cycles+=1
    relative=torch.linalg.vector_norm(b-A(x)).item()/bnorm
    if relative>rtol:raise RuntimeError(f'FGMRES did not converge: {iterations} iterations, residual {relative}')
    return x,dict(iterations=iterations,true_residual=relative,restarts=cycles)
