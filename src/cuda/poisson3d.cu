// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/cuda_poisson.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace bspf::cuda {
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void ck(cublasStatus_t e) { if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error "+std::to_string(int(e))); }
struct Context {
    cudaStream_t stream{}; cublasHandle_t blas{};
    Context(){ck(cudaStreamCreate(&stream));try{ck(cublasCreate(&blas));ck(cublasSetStream(blas,stream));ck(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));}catch(...){if(blas)cublasDestroy(blas);cudaStreamDestroy(stream);throw;}}
    ~Context(){if(blas)cublasDestroy(blas);if(stream)cudaStreamDestroy(stream);}
    void sync() const {ck(cudaStreamSynchronize(stream));}
};
template<class T> struct Buffer {
    T* p=nullptr;size_t count=0;
    Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(p)cudaFree(p);}
    void alloc(size_t n){if(p)throw std::logic_error("buffer already allocated");ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));count=n;}
    void upload(const std::vector<T>& v){alloc(v.size());ck(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
};
bool finite(const std::vector<double>& v){return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);});}
void shape(const std::vector<double>& v,size_t n){if(v.size()!=n||!finite(v))throw std::invalid_argument("Poisson3D: invalid/nonfinite input array");}
int blocks(size_t n){return int((n+255)/256);}
size_t cube(int n){return size_t(n)*n*n;}
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}

// Rotate x-fastest (x,y,z) -> (y,z,x); no atomic accumulation.
template<class T> __global__ void rotate(const T* in,T* out,int nx,int ny,int nz){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(nx)*ny*nz)return;
    int x=i%nx,y=(i/nx)%ny,z=i/(size_t(nx)*ny);out[y+size_t(ny)*(z+size_t(nz)*x)]=in[i];
}
// Orthonormal reflection coordinates: paired sums and differences, then center.
template<bool rotated=false> __global__ void parity_pack(const double* in,double* out,int n,int cols){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(n)*cols)return;
    int i=k%n,j=k/n,h=n/2,e=(n+1)/2;const double r=0.7071067811865475244;
    const size_t base=rotated?j:size_t(n)*j,stride=rotated?cols:1;
    if(i<h)out[k]=(in[base+stride*i]+in[base+stride*(n-1-i)])*r;
    else if(i<e)out[k]=in[base+stride*h];
    else {int a=i-e;out[k]=(in[base+stride*a]-in[base+stride*(n-1-a)])*r;}
}
template<bool rotated=false> __global__ void parity_unpack(const double* in,double* out,int n,int cols){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(n)*cols)return;
    int i=k%n,j=k/n,h=n/2,e=(n+1)/2;const double r=0.7071067811865475244;
    const size_t dst=rotated?j+size_t(cols)*i:k;
    if(i<h)out[dst]=(in[i+size_t(n)*j]+in[e+i+size_t(n)*j])*r;
    else if(i==h && n%2)out[dst]=in[h+size_t(n)*j];
    else {int a=n-1-i;out[dst]=(in[a+size_t(n)*j]-in[e+a+size_t(n)*j])*r;}
}
// Undo rotate: input (ny,nz,nx) -> output (nx,ny,nz).
__global__ void unrotate(const double* in,double* out,int nx,int ny,int nz){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(nx)*ny*nz)return;
    int x=i%nx,y=(i/nx)%ny,z=i/(size_t(nx)*ny);
    out[i]=in[y+size_t(ny)*(z+size_t(nz)*x)];
}
__device__ bool wall(size_t i,int n){int x=i%n,y=(i/n)%n,z=i/(size_t(n)*n);return x==0||y==0||z==0||x==n-1||y==n-1||z==n-1;}
__global__ void mask(double* a,int n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(n)*n*n && wall(i,n))a[i]=0; }
__global__ void product(size_t N,const double* a,const double* b,double* c){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<N)c[i]=a[i]*b[i];}
__global__ void flux(size_t N,const double* xx,const double* xy,const double* xz,const double* yy,const double* yz,const double* zz,double* x,double* y,double* z){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=N)return;double a=x[i],b=y[i],c=z[i];
    x[i]=xx[i]*a+xy[i]*b+xz[i]*c;y[i]=xy[i]*a+yy[i]*b+yz[i]*c;z[i]=xz[i]*a+yz[i]*b+zz[i]*c;
}
__global__ void combine(size_t N,double* y,const double* x,double a,double b){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<N)y[i]=a*x[i]+b*y[i];}
__global__ void extract(int n,const double* a,float* b,double scale){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;int m=n-2;if(i>=size_t(m)*m*m)return;int x=i%m,y=(i/m)%m,z=i/(size_t(m)*m);b[i]=float(a[(x+1)+size_t(n)*(y+1+size_t(n)*(z+1))]/scale);}
__global__ void extend(int n,const float* a,double* b,double scale){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*n*n)return;int m=n-2;int x=i%n,y=(i/n)%n,z=i/(size_t(n)*n);b[i]=wall(i,n)?0:scale*double(a[x-1+size_t(m)*(y-1+size_t(m)*(z-1))]);}
__global__ void inverse(int m,const float* lam,float ax,float ay,float az,float* a){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(m)*m*m)a[i]/=(ax*lam[i%m]+ay*lam[(i/m)%m]+az*lam[i/(size_t(m)*m)]);}
void gemm(cublasHandle_t h,bool t,int out,int cols,int in,const double* a,int ld,const double* b,double* c){double one=1,zero=0;ck(cublasDgemm(h,t?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,out,cols,in,&one,a,ld,b,in,&zero,c,out));}
void gemm(cublasHandle_t h,bool t,int out,int cols,int in,const float* a,int ld,const float* b,float* c){float one=1,zero=0;ck(cublasSgemm(h,t?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,out,cols,in,&one,a,ld,b,in,&zero,c,out));}
#ifdef BSPF_POISSON_DIRECT_LAYOUT
// Apply y independently to each z slab, then z with a right-side GEMM.
// The tensor stays x-fastest throughout: no full-volume permutations.
void middle(Context& c, bool t, int ni, int no, const double* a,
            const double* by, int ld, double* b) {
    double one=1, zero=0;
    ck(cublasDgemmStridedBatched(c.blas, CUBLAS_OP_N,
        t?CUBLAS_OP_N:CUBLAS_OP_T, no, no, ni, &one,
        a, no, size_t(no)*ni, by, ld, 0, &zero,
        b, no, size_t(no)*no, ni));
}
void middle(Context& c, bool t, int ni, int no, const float* a,
            const float* by, int ld, float* b) {
    float one=1, zero=0;
    ck(cublasSgemmStridedBatched(c.blas, CUBLAS_OP_N,
        t?CUBLAS_OP_N:CUBLAS_OP_T, no, no, ni, &one,
        a, no, size_t(no)*ni, by, ld, 0, &zero,
        b, no, size_t(no)*no, ni));
}
void last(Context& c, bool t, int ni, int no, const double* b,
          const double* bz, int ld, double* out) {
    double one=1, zero=0;
    ck(cublasDgemm(c.blas, CUBLAS_OP_N, t?CUBLAS_OP_N:CUBLAS_OP_T,
        no*no, no, ni, &one, b, no*no, bz, ld, &zero, out, no*no));
}
void last(Context& c, bool t, int ni, int no, const float* b,
          const float* bz, int ld, float* out) {
    float one=1, zero=0;
    ck(cublasSgemm(c.blas, CUBLAS_OP_N, t?CUBLAS_OP_N:CUBLAS_OP_T,
        no*no, no, ni, &one, b, no*no, bz, ld, &zero, out, no*no));
}
#endif
template<class T> void tensor(Context& c, const T* in, T* out, int ni, int no,
                              const T* ax, const T* ay, const T* az, int ld,
                              bool transpose, T* a, T* b) {
#ifndef BSPF_POISSON_DIRECT_LAYOUT
    // Default: rotation-based layout. Direct layout remains an A/B experiment.
    gemm(c.blas,transpose,no,ni*ni,ni,ax,ld,in,a);
    rotate<<<blocks(size_t(no)*ni*ni),256,0,c.stream>>>(a,b,no,ni,ni);
    gemm(c.blas,transpose,no,ni*no,ni,ay,ld,b,a);
    rotate<<<blocks(size_t(no)*ni*no),256,0,c.stream>>>(a,b,no,ni,no);
    gemm(c.blas,transpose,no,no*no,ni,az,ld,b,a);
    rotate<<<blocks(cube(no)),256,0,c.stream>>>(a,out,no,no,no);
#else
    gemm(c.blas,transpose,no,ni*ni,ni,ax,ld,in,a);
    middle(c,transpose,ni,no,a,ay,ld,b);
    last(c,transpose,ni,no,b,az,ld,out);
#endif
    ck(cudaGetLastError());
}
} // namespace

struct GalerkinPoisson3D::Impl {
    Context ctx; int n,q,m;size_t N,Q,M,bytes=0;bool rhs_ready=false,solved=false;
    Buffer<double> B,D,measure,G[6],scratch1,scratch2,grad[3],temp,force,boundary,load,rhs,x,r,z,p,ap,answer;
    Buffer<float> V,lam,small1,small2,fin,fout;
    float average[3];
    bool parity=false,shared=false,fused=false;
    Buffer<double> common;
    Buffer<double> parityB[2],parityD[2],packedIn,packedOut;
    // Blocks T_q^T A T_n. Reflection preserves parity for B and flips it for D.
    void make_blocks(const std::vector<double>& a,Buffer<double>* dest,bool derivative){
        const double r=0.7071067811865475244;
        for(int op=0;op<2;++op){
            int ip=op^int(derivative),rows=(q+1-op)/2,cols=(n+1-ip)/2;
            std::vector<double> block(size_t(rows)*cols);
            for(int j=0;j<cols;++j)for(int i=0;i<rows;++i){
                double value=0;
                for(int u=0;u<2;++u)for(int v=0;v<2;++v){
                    bool ci=(q%2 && op==0 && i==q/2),cj=(n%2 && ip==0 && j==n/2);
                    if((ci&&u)||(cj&&v))continue;
                    double wi=ci?1:r*((op&&u)?-1:1),wj=cj?1:r*((ip&&v)?-1:1);
                    value+=wi*wj*a[(u?q-1-i:i)+size_t(q)*(v?n-1-j:j)];
                }
                block[i+size_t(rows)*j]=value;
            }
            dest[op].upload(block);
        }
    }
    void parity_gemm(const double* in,double* out,int ni,int no,int cols,bool derivative,bool transpose,
                     bool rotated_input=false,bool rotated_output=false){
        if(rotated_input)parity_pack<true><<<blocks(size_t(ni)*cols),256,0,ctx.stream>>>(in,packedIn.p,ni,cols);
        else parity_pack<false><<<blocks(size_t(ni)*cols),256,0,ctx.stream>>>(in,packedIn.p,ni,cols);
        double one=1,zero=0;
        for(int op=0;op<2;++op){
            int ip=op^int(derivative),nr=(no+1-op)/2,nc=(ni+1-ip)/2;
            int stored=transpose?ip:op;
            auto& matrix=derivative?parityD[stored]:parityB[stored];
            ck(cublasDgemm(ctx.blas,transpose?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,
                nr,cols,nc,&one,matrix.p,transpose?nc:nr,
                packedIn.p+(ip?(ni+1)/2:0),ni,&zero,
                packedOut.p+(op?(no+1)/2:0),no));
        }
        if(rotated_output)parity_unpack<true><<<blocks(size_t(no)*cols),256,0,ctx.stream>>>(packedOut.p,out,no,cols);
        else parity_unpack<false><<<blocks(size_t(no)*cols),256,0,ctx.stream>>>(packedOut.p,out,no,cols);
    }
    explicit Impl(const Poisson3DData& d):n(d.n),q(d.q),m(0){
        if(n<4||q<n||q>512)throw std::invalid_argument("Poisson3D requires 4<=n<=q<=512");
        m=n-2;N=cube(n);Q=cube(q);M=cube(m);
        shape(d.basis,size_t(q)*n);shape(d.derivative,size_t(q)*n);shape(d.eigenvectors,size_t(m)*m);shape(d.eigenvalues,m);shape(d.measure,Q);
        for(auto& g:d.metric)shape(g,Q);
        for(auto l:d.eigenvalues)if(l<=0)throw std::invalid_argument("nonpositive preconditioner eigenvalue");
        // Sylvester's criterion: a positive metric is essential for CG.
        for(size_t i=0;i<Q;++i){double a=d.metric[0][i],b=d.metric[1][i],c=d.metric[2][i],e=d.metric[3][i],f=d.metric[4][i],g=d.metric[5][i];
            if(!(d.measure[i]>0&&a>0&&a*e-b*b>0&&a*(e*g-f*f)-b*(b*g-c*f)+c*(b*f-c*e)>0))throw std::invalid_argument("non-SPD quadrature metric");}
        B.upload(d.basis);D.upload(d.derivative);measure.upload(d.measure);
        for(int i=0;i<6;++i)G[i].upload(d.metric[i]);
        for(int a=0;a<3;++a){int k=a==0?0:(a==1?3:5);average[a]=float(std::accumulate(d.metric[k].begin(),d.metric[k].end(),0.));if(!(average[a]>0&&std::isfinite(average[a])))throw std::invalid_argument("invalid mean metric");}
        std::vector<float> vf(d.eigenvectors.begin(),d.eigenvectors.end()),lf(d.eigenvalues.begin(),d.eigenvalues.end());
        for(float v:vf)if(!std::isfinite(v))throw std::invalid_argument("eigenvector outside FP32 range");
        for(float l:lf)if(!(l>0&&std::isfinite(l)))throw std::invalid_argument("eigenvalue outside FP32 range");
        V.upload(vf);lam.upload(lf);
#ifndef BSPF_POISSON_DIRECT_LAYOUT
        const size_t workspace=Q;
#else
        const size_t workspace=size_t(n)*q*q;
#endif
        scratch1.alloc(workspace);scratch2.alloc(workspace);
        for(auto b:{&grad[0],&grad[1],&grad[2],&force})b->alloc(Q);
        for(auto b:{&temp,&boundary,&load,&rhs,&x,&r,&z,&p,&ap,&answer})b->alloc(N);
        for(auto b:{&small1,&small2,&fin,&fout})b->alloc(M);
        bytes=sizeof(double)*(2*size_t(q)*n+11*Q+2*workspace+10*N)+sizeof(float)*(size_t(m)*m+m+4*M);
        const char* setting=std::getenv("BSPF_POISSON_PARITY");
        parity=!(setting && std::string(setting)=="0");
        for(int derivative=0;derivative<2;++derivative){
            const auto& a=derivative?d.derivative:d.basis;double scale=0,defect=0;
            for(size_t i=0;i<a.size();++i){scale=std::max(scale,std::abs(a[i]));defect=std::max(defect,std::abs(a[i]-(derivative?-1:1)*a[a.size()-1-i]));}
            if(defect>1e-12*scale)parity=false;
        }
        if(parity){
            make_blocks(d.basis,parityB,false);make_blocks(d.derivative,parityD,true);
            packedIn.alloc(Q);packedOut.alloc(Q);
            bytes+=sizeof(double)*(2*Q+parityB[0].count+parityB[1].count+parityD[0].count+parityD[1].count);
        }
        const char* sharing=std::getenv("BSPF_POISSON_SHARED");
#ifndef BSPF_POISSON_DIRECT_LAYOUT
        shared=parity && !(sharing && std::string(sharing)=="0");
#endif
        if(shared){common.alloc(size_t(n)*n*q);bytes+=common.count*sizeof(double);}
        const char* fusion=std::getenv("BSPF_POISSON_FUSED_LAYOUT");
        fused=parity && !(fusion && std::string(fusion)=="0");
        ctx.sync();
    }
    void copy(const double* a,double* b,size_t count){ck(cudaMemcpyAsync(b,a,count*sizeof(double),cudaMemcpyDeviceToDevice,ctx.stream));}
    void zero(double* a){ck(cudaMemsetAsync(a,0,N*sizeof(double),ctx.stream));}
    void mask(double* a){::bspf::cuda::mask<<<blocks(N),256,0,ctx.stream>>>(a,n);ck(cudaGetLastError());}
    double norm(const double* a){double v;ck(cublasDnrm2(ctx.blas,int(N),a,1,&v));return v;}
    double dot(const double* a,const double* b){double v;ck(cublasDdot(ctx.blas,int(N),a,1,b,1,&v));return v;}
    void add(double* y,const double* a,double alpha,double beta=1){combine<<<blocks(N),256,0,ctx.stream>>>(N,y,a,alpha,beta);ck(cudaGetLastError());}
    void transform(const double* a,double* b,const double* bx,const double* by,const double* bz,bool transpose){
        if(!parity){tensor(ctx,a,b,transpose?q:n,transpose?n:q,bx,by,bz,q,transpose,scratch1.p,scratch2.p);return;}
        int ni=transpose?q:n,no=transpose?n:q;
        if(fused){
            parity_gemm(a,scratch2.p,ni,no,ni*ni,bx==D.p,transpose,false,true);
            parity_gemm(scratch2.p,scratch2.p,ni,no,ni*no,by==D.p,transpose,false,true);
            parity_gemm(scratch2.p,b,ni,no,no*no,bz==D.p,transpose,false,true);
            ck(cudaGetLastError());return;
        }
        parity_gemm(a,scratch1.p,ni,no,ni*ni,bx==D.p,transpose);
        rotate<<<blocks(size_t(no)*ni*ni),256,0,ctx.stream>>>(scratch1.p,scratch2.p,no,ni,ni);
        parity_gemm(scratch2.p,scratch1.p,ni,no,ni*no,by==D.p,transpose);
        rotate<<<blocks(size_t(no)*ni*no),256,0,ctx.stream>>>(scratch1.p,scratch2.p,no,ni,no);
        // packedOut is internal to parity_gemm; final rotation needs a full Q workspace.
        parity_gemm(scratch2.p,packedIn.p,ni,no,no*no,bz==D.p,transpose);
        rotate<<<blocks(cube(no)),256,0,ctx.stream>>>(packedIn.p,b,no,no,no);
        ck(cudaGetLastError());
    }
    void shared_gradient(const double* in){
        // gx has its own Dx prefix. gy and gz share Bx and its rotation.
        transform(in,grad[0].p,D.p,B.p,B.p,false);
        if(fused){
            parity_gemm(in,common.p,n,q,n*n,false,false,false,true);
            for(int c=1;c<3;++c){
                parity_gemm(common.p,scratch2.p,n,q,n*q,c==1,false,false,true);
                parity_gemm(scratch2.p,grad[c].p,n,q,q*q,c==2,false,false,true);
            }
            return;
        }
        parity_gemm(in,scratch1.p,n,q,n*n,false,false);
        rotate<<<blocks(size_t(q)*n*n),256,0,ctx.stream>>>(scratch1.p,common.p,q,n,n);
        for(int c=1;c<3;++c){
            parity_gemm(common.p,scratch1.p,n,q,n*q,c==1,false);
            rotate<<<blocks(size_t(q)*n*q),256,0,ctx.stream>>>(scratch1.p,scratch2.p,q,n,q);
            parity_gemm(scratch2.p,packedIn.p,n,q,q*q,c==2,false);
            rotate<<<blocks(Q),256,0,ctx.stream>>>(packedIn.p,grad[c].p,q,q,q);
        }
    }
    void shared_divergence(double* out){
        if(fused){
            // Fuse inverse rotations into parity packing. Keep the shared
            // gy/gz accumulator rotated until the final Bx^T contraction.
            for(int c=0;c<3;++c){
                parity_gemm(grad[c].p,scratch1.p,q,n,q*q,c==2,true,true,false);
                parity_gemm(scratch1.p,scratch2.p,q,n,n*q,c==1,true,true,false);
                if(c==0)parity_gemm(scratch2.p,out,q,n,n*n,true,true,true,false);
                else if(c==1)copy(scratch2.p,common.p,common.count);
                else {
                    combine<<<blocks(common.count),256,0,ctx.stream>>>(common.count,common.p,scratch2.p,1,1);
                    parity_gemm(common.p,temp.p,q,n,n*n,false,true,true,false);add(out,temp.p,1);
                }
            }
            return;
        }
        // Reverse the shared forward tree exactly: accumulate gy/gz before Bx^T.
        for(int c=0;c<3;++c){
            unrotate<<<blocks(Q),256,0,ctx.stream>>>(grad[c].p,scratch1.p,q,q,q);
            parity_gemm(scratch1.p,scratch2.p,q,n,q*q,c==2,true);
            unrotate<<<blocks(size_t(n)*q*q),256,0,ctx.stream>>>(scratch2.p,scratch1.p,q,n,q);
            parity_gemm(scratch1.p,scratch2.p,q,n,n*q,c==1,true);
            unrotate<<<blocks(size_t(n)*n*q),256,0,ctx.stream>>>(scratch2.p,scratch1.p,q,n,n);
            if(c==0)parity_gemm(scratch1.p,out,q,n,n*n,true,true);
            else if(c==1)copy(scratch1.p,common.p,common.count);
            else {
                combine<<<blocks(common.count),256,0,ctx.stream>>>(common.count,common.p,scratch1.p,1,1);
                parity_gemm(common.p,temp.p,q,n,n*n,false,true);add(out,temp.p,1);
            }
        }
    }
    void apply(const double* in,double* out){
        if(shared){
            shared_gradient(in);
            flux<<<blocks(Q),256,0,ctx.stream>>>(Q,G[0].p,G[1].p,G[2].p,G[3].p,G[4].p,G[5].p,grad[0].p,grad[1].p,grad[2].p);
            shared_divergence(out);ck(cudaGetLastError());return;
        }
        transform(in,grad[0].p,D.p,B.p,B.p,false);
        transform(in,grad[1].p,B.p,D.p,B.p,false);
        transform(in,grad[2].p,B.p,B.p,D.p,false);
        flux<<<blocks(Q),256,0,ctx.stream>>>(Q,G[0].p,G[1].p,G[2].p,G[3].p,G[4].p,G[5].p,grad[0].p,grad[1].p,grad[2].p);
        transform(grad[0].p,out,D.p,B.p,B.p,true);
        transform(grad[1].p,temp.p,B.p,D.p,B.p,true);add(out,temp.p,1);
        transform(grad[2].p,temp.p,B.p,B.p,D.p,true);add(out,temp.p,1);
    }
    void precondition(const double* in,double* out){
        double scale=norm(in);if(scale==0){zero(out);return;}if(!std::isfinite(scale))throw std::runtime_error("nonfinite preconditioner input");
        extract<<<blocks(M),256,0,ctx.stream>>>(n,in,fin.p,scale);
        tensor(ctx,fin.p,fout.p,m,m,V.p,V.p,V.p,m,true,small1.p,small2.p);
        inverse<<<blocks(M),256,0,ctx.stream>>>(m,lam.p,average[0],average[1],average[2],fout.p);
        tensor(ctx,fout.p,fin.p,m,m,V.p,V.p,V.p,m,false,small1.p,small2.p);
        extend<<<blocks(N),256,0,ctx.stream>>>(n,fin.p,out,scale);ck(cudaGetLastError());
    }
};
GalerkinPoisson3D::GalerkinPoisson3D(const Poisson3DData& d):d_(std::make_unique<Impl>(d)){}
GalerkinPoisson3D::~GalerkinPoisson3D()=default;
std::size_t GalerkinPoisson3D::device_bytes()const{return d_->bytes;}
bool GalerkinPoisson3D::parity_optimized()const{return d_->parity;}
bool GalerkinPoisson3D::shared_contractions()const{return d_->shared;}
bool GalerkinPoisson3D::fused_tensor_layout()const{return d_->fused;}
void GalerkinPoisson3D::set_rhs(const std::vector<double>& f,const std::vector<double>& g){
    auto& d=*d_;shape(f,d.Q);shape(g,d.N);
    for(int k=1;k<d.n-1;++k)for(int j=1;j<d.n-1;++j)for(int i=1;i<d.n-1;++i)if(g[i+size_t(d.n)*(j+size_t(d.n)*k)]!=0)throw std::invalid_argument("Dirichlet lift must have zero interior");
    ck(cudaMemcpy(d.force.p,f.data(),d.Q*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemcpy(d.boundary.p,g.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));d.rhs_ready=true;d.solved=false;
}
Poisson3DStats GalerkinPoisson3D::solve(const Poisson3DOptions& o){
    auto& d=*d_;if(!d.rhs_ready)throw std::logic_error("set_rhs required before solve");d.solved=false;
    if(!(o.relative_tolerance>0&&o.relative_tolerance<1)||o.max_iterations<1||o.residual_refresh<1)throw std::invalid_argument("invalid CG options");
    Poisson3DStats s;d.ctx.sync();auto start=Clock::now();
    product<<<blocks(d.Q),256,0,d.ctx.stream>>>(d.Q,d.force.p,d.measure.p,d.grad[0].p);
    d.transform(d.grad[0].p,d.load.p,d.B.p,d.B.p,d.B.p,true);
    d.apply(d.boundary.p,d.ap.p);d.copy(d.load.p,d.rhs.p,d.N);d.add(d.rhs.p,d.ap.p,-1);d.mask(d.rhs.p);
    d.zero(d.x.p);d.copy(d.rhs.p,d.r.p,d.N);double bnorm=d.norm(d.rhs.p),target=o.relative_tolerance*bnorm;
    d.ctx.sync();auto cgstart=Clock::now();
    if(bnorm>0){
        d.precondition(d.r.p,d.z.p);d.copy(d.z.p,d.p.p,d.N);double rho=d.dot(d.r.p,d.z.p);
        while(true){
            if(s.iterations>=o.max_iterations)throw std::runtime_error("Poisson3D CG iteration limit");
            d.apply(d.p.p,d.ap.p);d.mask(d.ap.p);double pap=d.dot(d.p.p,d.ap.p);
            if(!(rho>0&&pap>0&&std::isfinite(rho)&&std::isfinite(pap)))throw std::runtime_error("Poisson3D CG lost positive definiteness");
            double alpha=rho/pap;d.add(d.x.p,d.p.p,alpha);d.add(d.r.p,d.ap.p,-alpha);++s.iterations;
            double rnorm=d.norm(d.r.p);bool refresh=(rnorm<=target||s.iterations%o.residual_refresh==0);
            if(refresh){d.apply(d.x.p,d.ap.p);d.mask(d.ap.p);d.copy(d.rhs.p,d.r.p,d.N);d.add(d.r.p,d.ap.p,-1);++s.residual_replacements;rnorm=d.norm(d.r.p);}
            if(!std::isfinite(rnorm))throw std::runtime_error("nonfinite CG residual");
            if(refresh&&rnorm<=target)break;
            d.precondition(d.r.p,d.z.p);double next=d.dot(d.r.p,d.z.p);
            d.add(d.p.p,d.z.p,1,refresh?0:next/rho);rho=next;
        }
    }
    d.ctx.sync();s.cg_seconds=seconds(cgstart);
    d.copy(d.boundary.p,d.answer.p,d.N);d.add(d.answer.p,d.x.p,1);d.ctx.sync();s.solve_seconds=seconds(start);
    // Recompute against full boundary lift, not just the recursive CG residual.
    d.apply(d.answer.p,d.ap.p);d.copy(d.load.p,d.r.p,d.N);d.add(d.r.p,d.ap.p,-1);d.mask(d.r.p);
    double rn=d.norm(d.r.p);s.relative_residual=bnorm>0?rn/bnorm:rn;
    if(!std::isfinite(s.relative_residual)||s.relative_residual>std::max(10*o.relative_tolerance,1e-11))throw std::runtime_error("full Galerkin residual check failed");
    d.solved=true;return s;
}
std::vector<double> GalerkinPoisson3D::solution()const{auto& d=*d_;if(!d.solved)throw std::logic_error("no successful solve");std::vector<double> v(d.N);d.ctx.sync();ck(cudaMemcpy(v.data(),d.answer.p,d.N*sizeof(double),cudaMemcpyDeviceToHost));return v;}
std::vector<double> GalerkinPoisson3D::apply_host(const std::vector<double>& v){auto& d=*d_;shape(v,d.N);ck(cudaMemcpy(d.x.p,v.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));d.apply(d.x.p,d.ap.p);std::vector<double> out(d.N);d.ctx.sync();ck(cudaMemcpy(out.data(),d.ap.p,d.N*sizeof(double),cudaMemcpyDeviceToHost));return out;}
std::vector<double> GalerkinPoisson3D::precondition_host(const std::vector<double>& v){auto& d=*d_;shape(v,d.N);ck(cudaMemcpy(d.r.p,v.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));d.precondition(d.r.p,d.z.p);std::vector<double> out(d.N);d.ctx.sync();ck(cudaMemcpy(out.data(),d.z.p,d.N*sizeof(double),cudaMemcpyDeviceToHost));return out;}
} // namespace bspf::cuda
