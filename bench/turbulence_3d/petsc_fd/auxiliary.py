"""GPU assembly of a full-metric Q1 auxiliary operator on the FD grid."""
import numpy as np
import cupy as cp
from discretization import geometry

SCATTER=cp.RawKernel(r'''
extern "C" __global__ void scatter(int m,int first,int count,const double* K,const double* mass,double* vals,double* lump) {
 int t=blockDim.x*blockIdx.x+threadIdx.x;if(t>=count*64)return;
 int c=t/64,a=(t/8)%8,b=t%8,e=m+1,cell=first+c;
 int cx=cell%e,cy=(cell/e)%e,cz=cell/(e*e);
 int ax=a%2,ay=(a/2)%2,az=a/4,bx=b%2,by=(b/2)%2,bz=b/4;
 int x=cx+ax-1,y=cy+ay-1,z=cz+az-1;
 if(x<0||x>=m||y<0||y>=m||z<0||z>=m)return;
 int row=x+m*(y+m*z);
 int slot=(bx-ax+1)+3*((by-ay+1)+3*(bz-az+1));
 atomicAdd(vals+row*27+slot,K[t]);
 if(b==0)atomicAdd(lump+row,mass[c*8+a]);
}''','scatter')
GATHER=cp.RawKernel(r'''
extern "C" __global__ void gather(int m,int M,const int* offsets,const double* vals,const double* lump,int* col,double* data) {
 int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=M)return;
 int x=row%m,y=(row/m)%m,z=row/(m*m),k=offsets[row];
 // Diagonal first is the HYPRE native CSR convention.
 col[k]=row;data[k++]=vals[row*27+13]/lump[row];
 for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
  if(dx==0&&dy==0&&dz==0)continue;
  if(x+dx<0||x+dx>=m||y+dy<0||y+dy>=m||z+dz<0||z+dz>=m)continue;
  col[k]=(x+dx)+m*((y+dy)+m*(z+dz));
  data[k++]=vals[row*27+(dx+1)+3*((dy+1)+3*(dz+1))]/lump[row];
 }
}''','gather')

def assemble(n,box):
    e=n-1;m=n-2;M=m**3;h=1/e
    g=(np.polynomial.legendre.leggauss(2)[0]+1)/2
    zz,yy,xx=np.meshgrid(g,g,g,indexing='ij');q=np.stack([xx.ravel(),yy.ravel(),zz.ravel()],axis=1)
    vertices=np.array([[i%2,(i//2)%2,i//4] for i in range(8)])
    shape=np.prod(np.where(vertices[None,:,:],q[:,None,:],1-q[:,None,:]),axis=2)
    grad=np.empty((8,8,3))
    factors=np.where(vertices[None,:,:],q[:,None,:],1-q[:,None,:])
    for a in range(3):
        others=[j for j in range(3) if j!=a]
        grad[:,:,a]=(2*vertices[None,:,a]-1)*np.prod(factors[:,:,others],axis=2)/h
    # Tensor-product two-point Gauss rule; metric evaluated at physical-map Jacobians.
    q=cp.asarray(q);shape=cp.asarray(shape);grad=cp.asarray(grad)
    vals=cp.zeros((M,27));mass=cp.zeros(M)
    for first in range(0,e**3,16384):
        count=min(16384,e**3-first);c=cp.arange(first,first+count)
        origin=cp.stack([c%e,(c//e)%e,c//(e*e)],axis=1)
        points=(origin[:,None,:]+q[None,:,:])*h
        _,G,_,det=geometry(points,box,cp)
        wt=det*(h**3/8)
        K=cp.ascontiguousarray(cp.einsum('qia,cqab,qjb,cq->cij',grad,G,grad,wt,optimize=True))
        lm=cp.ascontiguousarray(cp.einsum('qi,cq->ci',shape,wt))
        SCATTER(((count*64+255)//256,),(256,),(np.int32(m),np.int32(first),np.int32(count),K,lm,vals,mass))
    idx=cp.arange(M);x=idx%m;y=(idx//m)%m;z=idx//(m*m)
    counts=(1+(x>0).astype(cp.int32)+(x<m-1).astype(cp.int32))*(1+(y>0).astype(cp.int32)+(y<m-1).astype(cp.int32))*(1+(z>0).astype(cp.int32)+(z<m-1).astype(cp.int32))
    rows=cp.empty(M+1,dtype=cp.int32);rows[0]=0;rows[1:]=cp.cumsum(counts,dtype=cp.int32)
    nnz=int(rows[-1]);cols=cp.empty(nnz,dtype=cp.int32);data=cp.empty(nnz)
    GATHER(((M+255)//256,),(256,),(np.int32(m),np.int32(M),rows,vals,mass,cols,data))
    if not bool(cp.all(mass>0)):raise RuntimeError('Nonpositive lumped auxiliary mass')
    result=tuple(cp.asnumpy(v) for v in [rows,cols,data,mass])
    return result
