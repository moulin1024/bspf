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
    void* workspace=nullptr;size_t workspace_bytes=0;int cublas_version=0;
    std::string emulation="native";
    Context(){
        ck(cudaStreamCreate(&stream));
        try{
            ck(cublasCreate(&blas));ck(cublasSetStream(blas,stream));ck(cublasGetVersion(blas,&cublas_version));
            const char* setting=std::getenv("BSPF_POISSON_FP64_EMULATION");
#if CUBLAS_VERSION >= 130100
            emulation=setting?setting:"performant";
            if(emulation!="native"&&emulation!="performant"&&emulation!="eager")throw std::invalid_argument("FP64 emulation must be native, performant or eager");
            if(emulation!="native"){
                ck(cublasSetMathMode(blas,CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
                ck(cublasSetEmulationStrategy(blas,emulation=="eager"?CUBLAS_EMULATION_STRATEGY_EAGER:CUBLAS_EMULATION_STRATEGY_PERFORMANT));
                // Preserve FP64-level accuracy: never reduce the mantissa target.
                ck(cublasSetFixedPointEmulationMantissaControl(blas,CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
                ck(cublasSetFixedPointEmulationMantissaBitOffset(blas,0));
                ck(cublasSetFixedPointEmulationMaxMantissaBitCount(blas,79));
                // Reuse emulation scratch across the synchronized CG iterations.
                workspace_bytes=size_t(512)*1024*1024;ck(cudaMalloc(&workspace,workspace_bytes));
                ck(cublasSetWorkspace(blas,workspace,workspace_bytes));
            }else ck(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
#else
            if(setting&&std::string(setting)!="native")throw std::invalid_argument("FP64 emulation requires cuBLAS >=13.1 (CUDA 13.0 Update 2)");
            ck(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
#endif
        }catch(...){if(blas)cublasDestroy(blas);if(workspace)cudaFree(workspace);cudaStreamDestroy(stream);throw;}
    }
    ~Context(){if(blas)cublasDestroy(blas);if(workspace)cudaFree(workspace);if(stream)cudaStreamDestroy(stream);}
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
// Orthogonal reflection coordinates, packed by parity (x=bit0,y=bit1,z=bit2).
// An odd-sized nodal center plane belongs to the even sector with unit weight.
struct OctantLayout { int dims[8][3]; size_t offset[8]; };
template<bool inverse> __global__ void octant_convert(const double* in,double* out,int n,OctantLayout layout){
    int e=(n+1)/2;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(e)*e*e)return;
    int pos[3]={int(i%e),int(i/e%e),int(i/(size_t(e)*e))};double a[8];
    for(int s=0;s<8;++s){
        bool valid=true;for(int d=0;d<3;++d)if(pos[d]>=(n+1-((s>>d)&1))/2)valid=false;
        if(inverse)a[s]=valid?in[layout.offset[s]+pos[0]+size_t(layout.dims[s][0])*(pos[1]+size_t(layout.dims[s][1])*pos[2])]:0;
        else {int x=(s&1)?n-1-pos[0]:pos[0],y=(s&2)?n-1-pos[1]:pos[1],z=(s&4)?n-1-pos[2]:pos[2];a[s]=in[x+size_t(n)*(y+size_t(n)*z)];}
    }
    for(int step=0;step<3;++step){int d=inverse?2-step:step,bit=1<<d;
        for(int s=0;s<8;++s)if(!(s&bit)){
            if(n%2&&pos[d]==n/2){a[s|bit]=0;}
            else {double u=a[s],v=a[s|bit];a[s]=(u+v)*0.7071067811865475244;a[s|bit]=(u-v)*0.7071067811865475244;}
        }
    }
    for(int s=0;s<8;++s){
        bool valid=true;for(int d=0;d<3;++d)if(pos[d]>=(n+1-((s>>d)&1))/2)valid=false;
        if(!valid)continue;
        if(!inverse)out[layout.offset[s]+pos[0]+size_t(layout.dims[s][0])*(pos[1]+size_t(layout.dims[s][1])*pos[2])]=a[s];
        else {int x=(s&1)?n-1-pos[0]:pos[0],y=(s&2)?n-1-pos[1]:pos[1],z=(s&4)?n-1-pos[2]:pos[2];out[x+size_t(n)*(y+size_t(n)*z)]=a[s];}
    }
}
__global__ void flux_octant(int q,int stride,bool rotated,const double* a,const double* b,const double* c,const double* d,const double* e,const double* f,double* x,double* y,double* z){
    // Gradient components occupy different parities. Reflection-covariant G
    // couples those sectors using its values in the left quadrature octant.
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(stride)*stride*stride)return;
    int ix=i%stride,iy=i/stride%stride,iz=i/(size_t(stride)*stride);
    size_t sector_size=size_t(stride)*stride*stride;
    size_t j=rotated?i:ix+size_t(q)*(iy+size_t(q)*iz);
    double aa=a[j],bb=b[j],cc=c[j],dd=d[j],ee=e[j],ff=f[j];
    // Reuse each metric load across all eight sectors. Gradients occupy the
    // same total storage as the full quadrature grid, but G is read only once.
    for(int s=0;s<8;++s){size_t k=i+s*sector_size;double u=x[k],v=y[k],w=z[k];x[k]=aa*u+bb*v+cc*w;y[k]=bb*u+dd*v+ee*w;z[k]=cc*u+ee*v+ff*w;}
}
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}

// Rotate x-fastest (x,y,z) -> (y,z,x); no atomic accumulation.
template<class T> __global__ void rotate(const T* in,T* out,int nx,int ny,int nz){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(nx)*ny*nz)return;
    int x=i%nx,y=(i/nx)%ny,z=i/(size_t(nx)*ny);out[y+size_t(ny)*(z+size_t(nz)*x)]=in[i];
}
// A cyclic tensor rotation is a matrix transpose: x versus the flattened yz
// plane. Tile it so both global reads and writes are coalesced. The extra
// shared-memory column avoids transpose bank conflicts.
template<class T> __global__ void transpose_tiled(const T* in,T* out,int width,int height){
    __shared__ T tile[32][33];
    int x=int(blockIdx.x)*32+int(threadIdx.x),y=int(blockIdx.y)*32+int(threadIdx.y);
    for(int j=0;j<32;j+=8)if(x<width&&y+j<height)
        tile[threadIdx.y+j][threadIdx.x]=in[size_t(y+j)*width+x];
    __syncthreads();
    x=int(blockIdx.y)*32+int(threadIdx.x);y=int(blockIdx.x)*32+int(threadIdx.y);
    for(int j=0;j<32;j+=8)if(x<height&&y+j<width)
        out[size_t(y+j)*height+x]=tile[threadIdx.x][threadIdx.y+j];
}
bool tiled_rotations_enabled(){
    static const bool enabled=[](){const char* p=std::getenv("BSPF_POISSON_TILED_ROTATE");return !(p&&std::string(p)=="0");}();
    return enabled;
}
template<class T> void rotate_into(cudaStream_t stream,const T* in,T* out,int nx,int ny,int nz){
    if(tiled_rotations_enabled())transpose_tiled<<<dim3((nx+31)/32,(ny*nz+31)/32),dim3(32,8),0,stream>>>(in,out,nx,ny*nz);
    else rotate<<<blocks(size_t(nx)*ny*nz),256,0,stream>>>(in,out,nx,ny,nz);
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
// Fused reflection conversion and transpose, with both directions coalesced.
// These retain the arithmetic/order of the scalar reflection kernels above.
__global__ void parity_unpack_rotated_tiled(const double* in,double* out,int n,int cols){
    __shared__ double tile[32][33];
    int i=int(blockIdx.x)*32+int(threadIdx.x),j=int(blockIdx.y)*32+int(threadIdx.y);
    int h=n/2,e=(n+1)/2;const double r=0.7071067811865475244;
    for(int k=0;k<32;k+=8)if(i<n&&j+k<cols){
        const double* a=in+size_t(n)*(j+k);double value;
        if(i<h)value=(a[i]+a[e+i])*r;
        else if(i==h&&n%2)value=a[h];
        else{int s=n-1-i;value=(a[s]-a[e+s])*r;}
        tile[threadIdx.y+k][threadIdx.x]=value;
    }
    __syncthreads();
    int x=int(blockIdx.y)*32+int(threadIdx.x),y=int(blockIdx.x)*32+int(threadIdx.y);
    for(int k=0;k<32;k+=8)if(x<cols&&y+k<n)
        out[size_t(y+k)*cols+x]=tile[threadIdx.x][threadIdx.y+k];
}
__global__ void parity_pack_rotated_tiled(const double* in,double* out,int n,int cols){
    __shared__ double tile[32][33];
    int j=int(blockIdx.x)*32+int(threadIdx.x),i=int(blockIdx.y)*32+int(threadIdx.y);
    int h=n/2,e=(n+1)/2;const double r=0.7071067811865475244;
    for(int k=0;k<32;k+=8)if(j<cols&&i+k<n){
        int p=i+k;double value;
        if(p<h)value=(in[j+size_t(cols)*p]+in[j+size_t(cols)*(n-1-p)])*r;
        else if(p<e)value=in[j+size_t(cols)*h];
        else{int a=p-e;value=(in[j+size_t(cols)*a]-in[j+size_t(cols)*(n-1-a)])*r;}
        tile[threadIdx.y+k][threadIdx.x]=value;
    }
    __syncthreads();
    int x=int(blockIdx.y)*32+int(threadIdx.x),y=int(blockIdx.x)*32+int(threadIdx.y);
    for(int k=0;k<32;k+=8)if(x<n&&y+k<cols)
        out[size_t(y+k)*n+x]=tile[threadIdx.x][threadIdx.y+k];
}
// Undo rotate: input (ny,nz,nx) -> output (nx,ny,nz).
__global__ void unrotate(const double* in,double* out,int nx,int ny,int nz){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(nx)*ny*nz)return;
    int x=i%nx,y=(i/nx)%ny,z=i/(size_t(nx)*ny);
    out[i]=in[y+size_t(ny)*(z+size_t(nz)*x)];
}
void unrotate_into(cudaStream_t stream,const double* in,double* out,int nx,int ny,int nz){
    if(tiled_rotations_enabled())transpose_tiled<<<dim3((ny*nz+31)/32,(nx+31)/32),dim3(32,8),0,stream>>>(in,out,ny*nz,nx);
    else unrotate<<<blocks(size_t(nx)*ny*nz),256,0,stream>>>(in,out,nx,ny,nz);
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
    rotate_into(c.stream,a,b,no,ni,ni);
    gemm(c.blas,transpose,no,ni*no,ni,ay,ld,b,a);
    rotate_into(c.stream,a,b,no,ni,no);
    gemm(c.blas,transpose,no,no*no,ni,az,ld,b,a);
    rotate_into(c.stream,a,out,no,no,no);
#else
    gemm(c.blas,transpose,no,ni*ni,ni,ax,ld,in,a);
    middle(c,transpose,ni,no,a,ay,ld,b);
    last(c,transpose,ni,no,b,az,ld,out);
#endif
    ck(cudaGetLastError());
}
} // namespace

struct GalerkinPoisson3D::Impl {
    Context ctx; int n,q,m;size_t N,Q,M,bytes=0;bool rhs_ready=false,solved=false,preassembled_load=false;
    Buffer<double> B,D,measure,G[6],scratch1,scratch2,grad[3],temp,force,boundary,load,rhs,x,r,z,p,ap,answer;
    Buffer<float> V,lam,small1,small2,fin,fout;
    float average[3];
    bool parity=false,shared=false,fused=false,octants=false;
    OctantLayout octant_layout{};
    int octant_q=0;
    bool octant_rotated=false;
    Buffer<double> octantG[6];
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
        if(rotated_input&&tiled_rotations_enabled())
            parity_pack_rotated_tiled<<<dim3((cols+31)/32,(ni+31)/32),dim3(32,8),0,ctx.stream>>>(in,packedIn.p,ni,cols);
        else if(rotated_input)parity_pack<true><<<blocks(size_t(ni)*cols),256,0,ctx.stream>>>(in,packedIn.p,ni,cols);
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
        if(rotated_output&&tiled_rotations_enabled())
            parity_unpack_rotated_tiled<<<dim3((no+31)/32,(cols+31)/32),dim3(32,8),0,ctx.stream>>>(packedOut.p,out,no,cols);
        else if(rotated_output)parity_unpack<true><<<blocks(size_t(no)*cols),256,0,ctx.stream>>>(packedOut.p,out,no,cols);
        else parity_unpack<false><<<blocks(size_t(no)*cols),256,0,ctx.stream>>>(packedOut.p,out,no,cols);
    }
    explicit Impl(const Poisson3DData& d):n(d.n),q(d.q),m(0){
        if(n<4||q<n)throw std::invalid_argument("Poisson3D requires 4<=n<=q");
        // cuBLAS uses signed-int matrix dimensions and vector lengths.
        // CUDA launches below use up to INT_MAX blocks of 256 threads.
        const size_t int_max=static_cast<size_t>(std::numeric_limits<int>::max());
        if(static_cast<unsigned long long>(q)*q>int_max ||
           cube(n)>int_max || cube(q)>int_max*size_t(256) ||
           cube(q)>std::numeric_limits<size_t>::max()/sizeof(double))
            throw std::invalid_argument("Poisson3D dimensions exceed cuBLAS/CUDA index limits");
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
        // Accumulated table-evaluation roundoff grows with axis size. Retain
        // the original floor and cap the allowance; genuinely nonsymmetric
        // tables must still take the full operator path. The discarded blocks
        // are independently checked by the 257^3 operator/solution campaign.
        const double parity_tolerance=std::min(1e-11,std::max(1e-12,
            32*std::numeric_limits<double>::epsilon()*std::max(n,q)));
        for(int derivative=0;derivative<2;++derivative){
            const auto& a=derivative?d.derivative:d.basis;double scale=0,defect=0;
            for(size_t i=0;i<a.size();++i){scale=std::max(scale,std::abs(a[i]));defect=std::max(defect,std::abs(a[i]-(derivative?-1:1)*a[a.size()-1-i]));}
            if(defect>parity_tolerance*scale)parity=false;
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
        const char* oct=std::getenv("BSPF_POISSON_OCTANTS");
        octants=parity && q%2==0 && !(oct && std::string(oct)=="0");
        if(octants){
            // This is an eligibility check, not a geometry approximation:
            // G_ab must change sign under reflection of exactly one of a,b.
            double scales[6]={},defects[6]={};int aa[6]={0,0,0,1,1,2},bb[6]={0,1,2,1,2,2};
            for(int z=0;z<q;++z)for(int y=0;y<q;++y)for(int x=0;x<q;++x){
                size_t i=x+size_t(q)*(y+size_t(q)*z),j=std::min(x,q-1-x)+size_t(q)*(std::min(y,q-1-y)+size_t(q)*std::min(z,q-1-z));
                int right[3]={x>=q/2,y>=q/2,z>=q/2};
                for(int k=0;k<6;++k){double v=d.metric[k][i],ref=d.metric[k][j]*(right[aa[k]]!=right[bb[k]]?-1:1);scales[k]=std::max(scales[k],std::abs(v));defects[k]=std::max(defects[k],std::abs(v-ref));}
            }
            for(int k=0;k<6;++k)if(defects[k]>1e-12*scales[k])octants=false;
        }
        octant_q=q/2;
        size_t offset=0;for(int s=0;s<8;++s){octant_layout.offset[s]=offset;size_t count=1;for(int d=0;d<3;++d){octant_layout.dims[s][d]=(n+1-((s>>d)&1))/2;count*=octant_layout.dims[s][d];}offset+=count;}
        const char* rotated=std::getenv("BSPF_POISSON_OCTANT_ROTATED");
        octant_rotated=octants && !(rotated && std::string(rotated)=="0");
        if(octant_rotated){
            // Match the last forward GEMM's (z,x,y) layout once during setup;
            // its adjoint starts in this layout, eliminating six rotations.
            for(int k=0;k<6;++k){
                std::vector<double> values(cube(octant_q),0.);
                for(int y=0;y<q/2;++y)for(int x=0;x<q/2;++x)for(int z=0;z<q/2;++z)
                    values[z+size_t(octant_q)*(x+size_t(octant_q)*y)]=d.metric[k][x+size_t(q)*(y+size_t(q)*z)];
                octantG[k].upload(values);bytes+=values.size()*sizeof(double);
            }
        }
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
        rotate_into(ctx.stream,scratch1.p,scratch2.p,no,ni,ni);
        parity_gemm(scratch2.p,scratch1.p,ni,no,ni*no,by==D.p,transpose);
        rotate_into(ctx.stream,scratch1.p,scratch2.p,no,ni,no);
        // packedOut is internal to parity_gemm; final rotation needs a full Q workspace.
        parity_gemm(scratch2.p,packedIn.p,ni,no,no*no,bz==D.p,transpose);
        rotate_into(ctx.stream,packedIn.p,b,no,no,no);
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
        rotate_into(ctx.stream,scratch1.p,common.p,q,n,n);
        for(int c=1;c<3;++c){
            parity_gemm(common.p,scratch1.p,n,q,n*q,c==1,false);
            rotate_into(ctx.stream,scratch1.p,scratch2.p,q,n,q);
            parity_gemm(scratch2.p,packedIn.p,n,q,q*q,c==2,false);
            rotate_into(ctx.stream,packedIn.p,grad[c].p,q,q,q);
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
            unrotate_into(ctx.stream,grad[c].p,scratch1.p,q,q,q);
            parity_gemm(scratch1.p,scratch2.p,q,n,q*q,c==2,true);
            unrotate_into(ctx.stream,scratch2.p,scratch1.p,q,n,q);
            parity_gemm(scratch1.p,scratch2.p,q,n,n*q,c==1,true);
            unrotate_into(ctx.stream,scratch2.p,scratch1.p,q,n,n);
            if(c==0)parity_gemm(scratch1.p,out,q,n,n*n,true,true);
            else if(c==1)copy(scratch1.p,common.p,common.count);
            else {
                combine<<<blocks(common.count),256,0,ctx.stream>>>(common.count,common.p,scratch1.p,1,1);
                parity_gemm(common.p,temp.p,q,n,n*n,false,true);add(out,temp.p,1);
            }
        }
    }
    void block_transform(const double* in,double* out,const int* ni,const int* no,const double* const* matrices,bool transpose){
        double one=1,zero=0;auto op=transpose?CUBLAS_OP_T:CUBLAS_OP_N;
        ck(cublasDgemm(ctx.blas,op,CUBLAS_OP_N,no[0],ni[1]*ni[2],ni[0],&one,matrices[0],octant_q,in,ni[0],&zero,scratch1.p,no[0]));
        rotate_into(ctx.stream,scratch1.p,scratch2.p,no[0],ni[1],ni[2]);
        ck(cublasDgemm(ctx.blas,op,CUBLAS_OP_N,no[1],ni[2]*no[0],ni[1],&one,matrices[1],octant_q,scratch2.p,ni[1],&zero,scratch1.p,no[1]));
        rotate_into(ctx.stream,scratch1.p,scratch2.p,no[1],ni[2],no[0]);
        bool keep_rotated=octant_rotated&&!transpose;
        ck(cublasDgemm(ctx.blas,op,CUBLAS_OP_N,no[2],no[0]*no[1],ni[2],&one,matrices[2],octant_q,scratch2.p,ni[2],&zero,keep_rotated?out:scratch1.p,no[2]));
        if(!keep_rotated)rotate_into(ctx.stream,scratch1.p,out,no[2],no[0],no[1]);
    }
    void apply_octants(const double* in,double* out){
        octant_convert<false><<<blocks(cube((n+1)/2)),256,0,ctx.stream>>>(in,packedIn.p,n,octant_layout);
        int half[3]={octant_q,octant_q,octant_q};
        // Finish all gradients before the single metric pass, then apply the
        // adjoint. Do not pack/unpack reflection coordinates between axes.
        for(int pass=0;pass<2;++pass){
        auto* metric=octant_rotated?octantG:G;
        if(pass==1)flux_octant<<<blocks(cube(octant_q)),256,0,ctx.stream>>>(q,octant_q,octant_rotated,metric[0].p,metric[1].p,metric[2].p,metric[3].p,metric[4].p,metric[5].p,grad[0].p,grad[1].p,grad[2].p);
        for(int s=0;s<8;++s){
            const int* dims=octant_layout.dims[s];size_t offset=octant_layout.offset[s],count=size_t(dims[0])*dims[1]*dims[2];
            double* sector_grad[3]={grad[0].p+s*cube(octant_q),grad[1].p+s*cube(octant_q),grad[2].p+s*cube(octant_q)};
            const double* matrices[3][3];
            for(int c=0;c<3;++c)for(int a=0;a<3;++a){int p=(s>>a)&1;matrices[c][a]=a==c?parityD[p^1].p:parityB[p].p;}
            if(pass==0){
            if(shared){
                int h=octant_q;
                block_transform(packedIn.p+offset,sector_grad[0],dims,half,matrices[0],false);
                gemm(ctx.blas,false,h,dims[1]*dims[2],dims[0],matrices[1][0],h,packedIn.p+offset,scratch1.p);
                rotate_into(ctx.stream,scratch1.p,common.p,h,dims[1],dims[2]);
                for(int c=1;c<3;++c){
                    gemm(ctx.blas,false,h,dims[2]*h,dims[1],matrices[c][1],h,common.p,scratch1.p);
                    rotate_into(ctx.stream,scratch1.p,scratch2.p,h,dims[2],h);
                    gemm(ctx.blas,false,h,h*h,dims[2],matrices[c][2],h,scratch2.p,octant_rotated?sector_grad[c]:scratch1.p);
                    if(!octant_rotated)rotate_into(ctx.stream,scratch1.p,sector_grad[c],h,h,h);
                }
            }else for(int c=0;c<3;++c)block_transform(packedIn.p+offset,sector_grad[c],dims,half,matrices[c],false);
            continue;
            }
            if(shared||octant_rotated){
                int h=octant_q;size_t middle=size_t(h)*dims[1]*dims[2];
                // Reverse the forward contraction order for the adjoint.
                // The y/z contributions share the final B_x^T contraction.
                for(int c=0;c<3;++c){
                    if(!octant_rotated)unrotate_into(ctx.stream,sector_grad[c],scratch1.p,h,h,h);
                    gemm(ctx.blas,true,dims[2],h*h,h,matrices[c][2],h,octant_rotated?sector_grad[c]:scratch1.p,scratch2.p);
                    unrotate_into(ctx.stream,scratch2.p,scratch1.p,h,dims[2],h);
                    gemm(ctx.blas,true,dims[1],dims[2]*h,h,matrices[c][1],h,scratch1.p,scratch2.p);
                    unrotate_into(ctx.stream,scratch2.p,scratch1.p,h,dims[1],dims[2]);
                    if(c==0)gemm(ctx.blas,true,dims[0],dims[1]*dims[2],h,matrices[c][0],h,scratch1.p,packedOut.p+offset);
                    else if(!shared){
                        gemm(ctx.blas,true,dims[0],dims[1]*dims[2],h,matrices[c][0],h,scratch1.p,temp.p);
                        combine<<<blocks(count),256,0,ctx.stream>>>(count,packedOut.p+offset,temp.p,1.,1.);
                    }
                    else if(c==1)copy(scratch1.p,common.p,middle);
                    else {
                        combine<<<blocks(middle),256,0,ctx.stream>>>(middle,common.p,scratch1.p,1.,1.);
                        gemm(ctx.blas,true,dims[0],dims[1]*dims[2],h,matrices[c][0],h,common.p,temp.p);
                        combine<<<blocks(count),256,0,ctx.stream>>>(count,packedOut.p+offset,temp.p,1.,1.);
                    }
                }
            }else for(int c=0;c<3;++c){block_transform(sector_grad[c],c==0?packedOut.p+offset:temp.p,half,dims,matrices[c],true);if(c)combine<<<blocks(count),256,0,ctx.stream>>>(count,packedOut.p+offset,temp.p,1.,1.);}
        }
        }
        octant_convert<true><<<blocks(cube((n+1)/2)),256,0,ctx.stream>>>(packedOut.p,out,n,octant_layout);ck(cudaGetLastError());
    }
    void apply(const double* in,double* out){
        if(octants){apply_octants(in,out);return;}
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
std::size_t GalerkinPoisson3D::device_bytes()const{return d_->bytes+d_->ctx.workspace_bytes;}
int GalerkinPoisson3D::cublas_version()const{return d_->ctx.cublas_version;}
std::string GalerkinPoisson3D::fp64_emulation()const{return d_->ctx.emulation;}
bool GalerkinPoisson3D::parity_optimized()const{return d_->parity;}
bool GalerkinPoisson3D::shared_contractions()const{return d_->shared;}
bool GalerkinPoisson3D::fused_tensor_layout()const{return d_->fused;}
bool GalerkinPoisson3D::octant_optimized()const{return d_->octants;}
void GalerkinPoisson3D::set_rhs(const std::vector<double>& f,const std::vector<double>& g){
    auto& d=*d_;shape(f,d.Q);shape(g,d.N);
    for(int k=1;k<d.n-1;++k)for(int j=1;j<d.n-1;++j)for(int i=1;i<d.n-1;++i)if(g[i+size_t(d.n)*(j+size_t(d.n)*k)]!=0)throw std::invalid_argument("Dirichlet lift must have zero interior");
    ck(cudaMemcpy(d.force.p,f.data(),d.Q*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemcpy(d.boundary.p,g.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));d.rhs_ready=true;d.solved=false;d.preassembled_load=false;
}
void GalerkinPoisson3D::set_load(const std::vector<double>& f,const std::vector<double>& g){
    auto& d=*d_;shape(f,d.N);shape(g,d.N);
    for(int k=1;k<d.n-1;++k)for(int j=1;j<d.n-1;++j)for(int i=1;i<d.n-1;++i)if(g[i+size_t(d.n)*(j+size_t(d.n)*k)]!=0)throw std::invalid_argument("Dirichlet lift must have zero interior");
    ck(cudaMemcpy(d.load.p,f.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemcpy(d.boundary.p,g.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));d.rhs_ready=true;d.solved=false;d.preassembled_load=true;
}
Poisson3DStats GalerkinPoisson3D::solve(const Poisson3DOptions& o){
    auto& d=*d_;if(!d.rhs_ready)throw std::logic_error("set_rhs required before solve");d.solved=false;
    if(!(o.relative_tolerance>0&&o.relative_tolerance<1)||o.max_iterations<1||o.residual_refresh<1)throw std::invalid_argument("invalid CG options");
    Poisson3DStats s;d.ctx.sync();auto start=Clock::now();
    if(!d.preassembled_load){
        product<<<blocks(d.Q),256,0,d.ctx.stream>>>(d.Q,d.force.p,d.measure.p,d.grad[0].p);
        d.transform(d.grad[0].p,d.load.p,d.B.p,d.B.p,d.B.p,true);
    }
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
