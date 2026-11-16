// Link against a private FP32 HYPRE archive. Hide all its symbols so MFEM's
// FP64 HYPRE cannot interpose on this library (or vice versa).
#include "hypre_fp32_bridge.h"
#include "_hypre_parcsr_ls.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
static_assert(sizeof(HYPRE_Real)==sizeof(float), "Bridge requires FP32 HYPRE");
static_assert(sizeof(HYPRE_Complex)==sizeof(float), "Real FP32 HYPRE required");
namespace {
thread_local std::string error;
void check(int status) { if(status) throw std::runtime_error("FP32 HYPRE error "+std::to_string(status)); }
void cuda_check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
struct AMG {
  int n;
  hypre_ParCSRMatrix *a=nullptr;
  hypre_ParVector *b=nullptr,*x=nullptr;
  HYPRE_Solver solver=nullptr;
  ~AMG() {
    if(solver) HYPRE_BoomerAMGDestroy(solver);
    if(x) hypre_ParVectorDestroy(x);
    if(b) hypre_ParVectorDestroy(b);
    if(a) hypre_ParCSRMatrixDestroy(a);
    HYPRE_Finalize();
  }
};
__global__ void to_float(int n,const double *src,float *dst,float *zero) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {dst[i]=float(src[i]);zero[i]=0.f;}
}
__global__ void to_double(int n,const float *src,double *dst) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) dst[i]=double(src[i]);
}
}
extern "C" int bspf_amg32_scalar_bytes() {return sizeof(HYPRE_Real);}
extern "C" const char *bspf_amg32_error() {return error.c_str();}
extern "C" void bspf_amg32_destroy(void *ptr) {delete static_cast<AMG*>(ptr);}
extern "C" void *bspf_amg32_create(int n,int nnz,const int *row,const int *col,const double *values) {
  AMG *s=nullptr;
  try {
    int ranks;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(ranks!=1) throw std::runtime_error("FP32 bridge supports one MPI rank");
    check(HYPRE_Init());s=new AMG;s->n=n;
    check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
    check(HYPRE_SetSpGemmUseCusparse(0)); // match MFEM's defaults
    HYPRE_BigInt starts[2]={0,n};
    s->a=hypre_ParCSRMatrixCreate(MPI_COMM_WORLD,n,n,starts,starts,0,nnz,0);
    check(hypre_ParCSRMatrixInitialize_v2(s->a,HYPRE_MEMORY_DEVICE));
    auto *diag=hypre_ParCSRMatrixDiag(s->a);
    std::vector<float> data(nnz);
    for(int i=0;i<nnz;++i) data[i]=float(values[i]);
    cuda_check(cudaMemcpy(hypre_CSRMatrixI(diag),row,(n+1)*sizeof(int),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(hypre_CSRMatrixJ(diag),col,nnz*sizeof(int),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(hypre_CSRMatrixData(diag),data.data(),nnz*sizeof(float),cudaMemcpyHostToDevice));
    check(hypre_ParCSRMatrixSetNumNonzeros(s->a));
    check(hypre_MatvecCommPkgCreate(s->a));
    s->b=hypre_ParVectorCreate(MPI_COMM_WORLD,n,starts);
    s->x=hypre_ParVectorCreate(MPI_COMM_WORLD,n,starts);
    check(hypre_ParVectorInitialize_v2(s->b,HYPRE_MEMORY_DEVICE));
    check(hypre_ParVectorInitialize_v2(s->x,HYPRE_MEMORY_DEVICE));
    check(hypre_ParVectorSetConstantValues(s->b,0.f));
    check(hypre_ParVectorSetConstantValues(s->x,0.f));
    check(HYPRE_BoomerAMGCreate(&s->solver));
    // Exactly MFEM 4.9 HypreBoomerAMG GPU defaults.
    check(HYPRE_BoomerAMGSetCoarsenType(s->solver,8));
    check(HYPRE_BoomerAMGSetAggNumLevels(s->solver,0));
    check(HYPRE_BoomerAMGSetRelaxType(s->solver,18));
    check(HYPRE_BoomerAMGSetNumSweeps(s->solver,1));
    check(HYPRE_BoomerAMGSetStrongThreshold(s->solver,.25f));
    check(HYPRE_BoomerAMGSetInterpType(s->solver,6));
    check(HYPRE_BoomerAMGSetPMaxElmts(s->solver,4));
    check(HYPRE_BoomerAMGSetPrintLevel(s->solver,1));
    check(HYPRE_BoomerAMGSetMaxLevels(s->solver,25));
    check(HYPRE_BoomerAMGSetMaxIter(s->solver,1));
    check(HYPRE_BoomerAMGSetTol(s->solver,0.f));
    check(HYPRE_BoomerAMGSetup(s->solver,(HYPRE_ParCSRMatrix)s->a,(HYPRE_ParVector)s->b,(HYPRE_ParVector)s->x));
    cuda_check(cudaDeviceSynchronize());return s;
  } catch(const std::exception &e) {error=e.what();delete s;return nullptr;}
}
extern "C" int bspf_amg32_apply(void *ptr,const double *rhs,double *solution) {
  try {
    auto *s=static_cast<AMG*>(ptr);
    auto *b=hypre_VectorData(hypre_ParVectorLocalVector(s->b));
    auto *x=hypre_VectorData(hypre_ParVectorLocalVector(s->x));
    // Synchronize across independent MFEM/HYPRE stream ownership. These costs
    // and both casts are included in reported outer CG time.
    cuda_check(cudaDeviceSynchronize());
    to_float<<<(s->n+255)/256,256>>>(s->n,rhs,b,x);
    cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());
    check(HYPRE_BoomerAMGSolve(s->solver,(HYPRE_ParCSRMatrix)s->a,(HYPRE_ParVector)s->b,(HYPRE_ParVector)s->x));
    cuda_check(cudaDeviceSynchronize());
    to_double<<<(s->n+255)/256,256>>>(s->n,x,solution);
    cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());return 0;
  } catch(const std::exception &e) {error=e.what();return 1;}
}
