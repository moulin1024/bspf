#include "bundle.hpp"
#include <petscksp.h>
#include <petscdevice_cuda.h>
#include <cuda_runtime.h>
#include "../mfem_poisson/hypre_fp32_bridge.h"
static_assert(sizeof(PetscScalar)==sizeof(double),"FD8 requires real FP64 PETSc");
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
static void ck(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct DeviceData {
  int n=0,m=0,N=0,M=0;
  int *starts=nullptr;
  double *weights=nullptr,*metric=nullptr,*drift=nullptr,*full=nullptr,*grad=nullptr;
  void *amg=nullptr;
  ~DeviceData() {
    bspf_amg32_destroy(amg);
    cudaFree(starts);cudaFree(weights);cudaFree(metric);cudaFree(drift);cudaFree(full);cudaFree(grad);
  }
};
template<class T> void upload(T*&dst,const T*src,size_t size) {
  ck(cudaMalloc(&dst,size*sizeof(T)));ck(cudaMemcpy(dst,src,size*sizeof(T),cudaMemcpyHostToDevice));
}
__device__ int width(int i,int n,int order) {return order==1 || (i>=4 && i<n-4)?9:10;}
__global__ void extend(int n,int N,const double *x,double *full) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=N)return;
  int i=t%n,j=(t/n)%n,k=t/(n*n),m=n-2;
  full[t]=(i>0&&i<n-1&&j>0&&j<n-1&&k>0&&k<n-1)?x[i-1+m*(j-1+m*(k-1))]:0.;
}
__global__ void gradient(int n,int N,const int *start,const double *w,const double *u,double *g) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=N)return;
  int xyz[3]={t%n,(t/n)%n,t/(n*n)},stride[3]={1,n,n*n};
  for(int a=0;a<3;++a) {
    int pos=xyz[a],base=t+(start[pos]-pos)*stride[a];double value=0.;
    for(int l=0;l<9;++l)value+=w[pos*10+l]*u[base+l*stride[a]];
    g[a*N+t]=value;
  }
}
__global__ void laplacian(int n,int M,const int *start,const double *w,const double *u,const double *g,const double *metric,const double *drift,double *out) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=M)return;
  int m=n-2,N=n*n*n,xyz[3]={t%m+1,(t/m)%m+1,t/(m*m)+1},stride[3]={1,n,n*n};
  int at=xyz[0]+n*(xyz[1]+n*xyz[2]);double value=0.;
  for(int a=0;a<3;++a) {
    int pos=xyz[a],base=at+(start[n+pos]-pos)*stride[a];double d2=0.;
    for(int l=0;l<width(pos,n,2);++l)d2+=w[(n+pos)*10+l]*u[base+l*stride[a]];
    value+=metric[a*M+t]*d2+drift[a*M+t]*g[a*N+at];
  }
  // metric packing: xx,yy,zz,xy,xz,yz; g packing: dx,dy,dz.
  int term=3;
  for(int a=0;a<3;++a)for(int b=a+1;b<3;++b) {
    int pos=xyz[a],base=at+(start[pos]-pos)*stride[a];double mixed=0.;
    for(int l=0;l<9;++l)mixed+=w[pos*10+l]*g[b*N+base+l*stride[a]];
    value+=2*metric[term++*M+t]*mixed;
  }
  out[t]=-value;
}
static PetscErrorCode apply_full(DeviceData &d,double *y) {
  PetscFunctionBeginUser;
  gradient<<<(d.N+255)/256,256>>>(d.n,d.N,d.starts,d.weights,d.full,d.grad);
  laplacian<<<(d.M+255)/256,256>>>(d.n,d.M,d.starts,d.weights,d.full,d.grad,d.metric,d.drift,y);
  PetscCallCUDA(cudaGetLastError());PetscCallCUDA(cudaDeviceSynchronize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode multiply(Mat A,Vec x,Vec y) {
  PetscFunctionBeginUser;
  DeviceData *d;const double *in;double *out;
  PetscCall(MatShellGetContext(A,&d));PetscCall(VecCUDAGetArrayRead(x,&in));PetscCall(VecCUDAGetArrayWrite(y,&out));
  PetscCallCUDA(cudaDeviceSynchronize());
  extend<<<(d->N+255)/256,256>>>(d->n,d->N,in,d->full);
  PetscCall(apply_full(*d,out));
  PetscCall(VecCUDARestoreArrayRead(x,&in));PetscCall(VecCUDARestoreArrayWrite(y,&out));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode precondition(PC pc,Vec x,Vec y) {
  PetscFunctionBeginUser;
  DeviceData *d;const double *in;double *out;
  PetscCall(PCShellGetContext(pc,&d));PetscCall(VecCUDAGetArrayRead(x,&in));PetscCall(VecCUDAGetArrayWrite(y,&out));
  PetscCheck(!bspf_amg32_apply(d->amg,in,out),PETSC_COMM_SELF,PETSC_ERR_LIB,"%s",bspf_amg32_error());
  PetscCall(VecCUDARestoreArrayRead(x,&in));PetscCall(VecCUDARestoreArrayWrite(y,&out));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode run() {
  PetscFunctionBeginUser;
  char input[PETSC_MAX_PATH_LEN]="",output[PETSC_MAX_PATH_LEN]="",kind[32]="fp32";PetscBool found;
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-input",input,sizeof(input),&found));
  PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_USER,"Specify -input bundle");
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-output",output,sizeof(output),&found));
  PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_USER,"Specify -output prefix");
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-pc_kind",kind,sizeof(kind),nullptr));
  bool fp32=std::string(kind)=="fp32";
  PetscCheck(fp32||std::string(kind)=="hypre",PETSC_COMM_SELF,PETSC_ERR_USER,"pc_kind must be fp32 or hypre");
  int ranks;MPI_Comm_size(PETSC_COMM_WORLD,&ranks);PetscCheck(ranks==1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Single-rank benchmark");
  auto data=fd_io::load(input);DeviceData d;d.n=int(data.at("n").v[0]);d.m=d.n-2;
  PetscCheck(d.n>=11 && d.n<=257,PETSC_COMM_SELF,PETSC_ERR_USER,"Supported n range: 11..257");
  d.N=d.n*d.n*d.n;d.M=d.m*d.m*d.m;
  auto require=[&](const char *name,size_t size) {if(data.at(name).size()!=size)throw std::runtime_error(std::string("Invalid size: ")+name);};
  require("starts",2*d.n);require("weights",20*d.n);require("metric",6*d.M);require("drift",3*d.M);
  require("boundary",d.N);require("forcing",d.M);require("probe",d.N);
  auto setup=Clock::now();std::vector<int> starts(2*d.n);
  for(int i=0;i<2*d.n;++i)starts[i]=int(data.at("starts").v[i]);
  upload(d.starts,starts.data(),starts.size());upload(d.weights,data.at("weights").v.data(),20*d.n);
  upload(d.metric,data.at("metric").v.data(),6*d.M);upload(d.drift,data.at("drift").v.data(),3*d.M);
  ck(cudaMalloc(&d.full,d.N*sizeof(double)));ck(cudaMalloc(&d.grad,3*d.N*sizeof(double)));
  Vec rhs,x,residual,work;PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF,d.M,&rhs));
  PetscCall(VecDuplicate(rhs,&x));PetscCall(VecDuplicate(rhs,&residual));PetscCall(VecDuplicate(rhs,&work));
  double *ptr;
  PetscCall(VecCUDAGetArrayWrite(rhs,&ptr));ck(cudaMemcpy(ptr,data.at("forcing").v.data(),d.M*sizeof(double),cudaMemcpyHostToDevice));PetscCall(VecCUDARestoreArrayWrite(rhs,&ptr));
  ck(cudaMemcpy(d.full,data.at("boundary").v.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));
  PetscCall(VecCUDAGetArrayWrite(work,&ptr));PetscCall(apply_full(d,ptr));PetscCall(VecCUDARestoreArrayWrite(work,&ptr));
  PetscCall(VecAXPY(rhs,-1.,work)); // exact nodal Dirichlet lifting
  // Export a full-grid probe application for an independent CPU check.
  ck(cudaMemcpy(d.full,data.at("probe").v.data(),d.N*sizeof(double),cudaMemcpyHostToDevice));
  PetscCall(VecCUDAGetArrayWrite(work,&ptr));PetscCall(apply_full(d,ptr));PetscCall(VecCUDARestoreArrayWrite(work,&ptr));
  fd_io::Mat probe(d.m*d.m,d.m);const double *host;PetscCall(VecGetArrayRead(work,&host));std::copy_n(host,d.M,probe.v.data());PetscCall(VecRestoreArrayRead(work,&host));
  Mat A,P;PetscCall(MatCreateShell(PETSC_COMM_SELF,d.M,d.M,d.M,d.M,&d,&A));
  PetscCall(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))multiply));PetscCall(MatShellSetVecType(A,VECCUDA));
  // SPD low-order reference-grid Laplacian, scaled by mean diagonal metric.
  // This is only the AMG auxiliary operator; the solve uses the full FD8 map.
  double scale[3]={0,0,0};for(int a=0;a<3;++a){for(int i=0;i<d.M;++i)scale[a]+=data.at("metric").v[a*d.M+i];scale[a]*=double((d.n-1)*(d.n-1))/d.M;}
  std::vector<PetscInt> row(d.M+1),col;std::vector<double> values;col.reserve(7*d.M);values.reserve(7*d.M);
  bool curved_auxiliary=data.count("pc_nnz");
  if(curved_auxiliary) {
    size_t nnz=size_t(data.at("pc_nnz").v[0]);col.resize(nnz);values.resize(nnz);
    PetscCheck(data.at("pc_rows").size()>=size_t(d.M+1) && data.at("pc_cols").size()>=nnz && data.at("pc_values").size()>=nnz,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid auxiliary CSR arrays");
    for(int i=0;i<=d.M;++i)row[i]=PetscInt(data.at("pc_rows").v[i]);
    for(size_t i=0;i<nnz;++i){col[i]=PetscInt(data.at("pc_cols").v[i]);values[i]=data.at("pc_values").v[i];}
  } else {
  for(int t=0;t<d.M;++t) {
    row[t]=col.size();col.push_back(t);values.push_back(2*(scale[0]+scale[1]+scale[2]));
    int pos[3]={t%d.m,(t/d.m)%d.m,t/(d.m*d.m)},stride[3]={1,d.m,d.m*d.m};
    for(int a=0;a<3;++a)for(int side : {-1,1})if(pos[a]+side>=0 && pos[a]+side<d.m){col.push_back(t+side*stride[a]);values.push_back(-scale[a]);}
  }
  row[d.M]=col.size();
  }
  PetscCall(MatCreate(PETSC_COMM_SELF,&P));PetscCall(MatSetSizes(P,d.M,d.M,d.M,d.M));PetscCall(MatSetType(P,MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocationCSR(P,row.data(),col.data(),values.data()));
  PetscCall(MatConvert(P,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,&P));
  KSP ksp;PC pc;PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));PetscCall(KSPSetOperators(ksp,A,P));
  PetscCall(KSPSetType(ksp,KSPFGMRES));PetscCall(KSPGMRESSetRestart(ksp,60));
  PetscCall(KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED));PetscCall(KSPSetTolerances(ksp,1e-11,1e-14,PETSC_DEFAULT,2000));
  PetscCall(KSPGetPC(ksp,&pc));
  if(fp32) {
    static_assert(sizeof(PetscInt)==sizeof(int),"Bridge needs 32-bit indices");
    d.amg=bspf_amg32_create(d.M,col.size(),row.data(),col.data(),values.data());
    PetscCheck(d.amg,PETSC_COMM_SELF,PETSC_ERR_LIB,"%s",bspf_amg32_error());
    PetscCall(PCSetType(pc,PCSHELL));PetscCall(PCShellSetName(pc,"FP32 GPU Hypre BoomerAMG"));
    PetscCall(PCShellSetContext(pc,&d));PetscCall(PCShellSetApply(pc,precondition));
  } else {
    PetscCall(PCSetType(pc,PCHYPRE));PetscCall(PCHYPRESetType(pc,"boomeramg"));
  }
  PetscCall(KSPSetFromOptions(ksp));PetscCall(KSPSetUp(ksp));ck(cudaDeviceSynchronize());double setup_s=seconds(setup);
  std::vector<double> times;std::vector<PetscInt> iterations;
  double true_res=0.;
  for(int repeat=0;repeat<4;++repeat) {
    PetscCall(VecSet(x,0.));PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE));ck(cudaDeviceSynchronize());auto t=Clock::now();
    PetscCall(KSPSolve(ksp,rhs,x));ck(cudaDeviceSynchronize());times.push_back(seconds(t));
    PetscInt it;KSPConvergedReason reason;PetscCall(KSPGetIterationNumber(ksp,&it));iterations.push_back(it);PetscCall(KSPGetConvergedReason(ksp,&reason));
    PetscCheck(reason>0,PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"KSP failed: reason %d after %d iterations",int(reason),int(it));
    PetscCall(MatMult(A,x,residual));PetscCall(VecAXPY(residual,-1.,rhs));PetscReal rn,bn;
    PetscCall(VecNorm(residual,NORM_2,&rn));PetscCall(VecNorm(rhs,NORM_2,&bn));true_res=rn/std::max(double(bn),1e-300);
    PetscCheck(std::isfinite(true_res)&&true_res<5e-10,PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"True residual %g",true_res);
    std::cout<<"solve "<<repeat<<" seconds "<<times.back()<<" iterations "<<it<<" true residual "<<true_res<<std::endl;
  }
  fd_io::Mat sol(d.m*d.m,d.m);PetscCall(VecGetArrayRead(x,&host));std::copy_n(host,d.M,sol.v.data());PetscCall(VecRestoreArrayRead(x,&host));
  fd_io::save(std::string(output)+".bin",{{"interior",std::move(sol)},{"probe_apply",std::move(probe)}});
  char version[256];PetscCall(PetscGetVersion(version,sizeof(version)));cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
  KSPType ksp_type;PCType pc_type;VecType vec_type;MatType mat_type;
  PetscCall(KSPGetType(ksp,&ksp_type));PetscCall(PCGetType(pc,&pc_type));PetscCall(VecGetType(x,&vec_type));PetscCall(MatGetType(P,&mat_type));
  std::ofstream report(std::string(output)+".json");report<<std::setprecision(17)
    <<"{\"n\":"<<d.n<<",\"dofs\":"<<d.N<<",\"free_dofs\":"<<d.M<<",\"order\":8,\"boundary_order\":8,"
    <<"\"gpu\":\""<<prop.name<<"\",\"petsc_version\":\""<<version<<"\",\"operator_precision\":\"float64\",\"krylov_precision\":\"float64\","
    <<"\"ksp_type\":\""<<ksp_type<<"\",\"pc_type\":\""<<pc_type<<"\",\"vec_type\":\""<<vec_type<<"\",\"auxiliary_matrix_type\":\""<<mat_type<<"\","
    <<"\"auxiliary_operator\":\""<<(curved_auxiliary?"full-metric mass-lumped Q1":"mean-metric reference Laplacian")<<"\","
    <<"\"preconditioner_precision\":\""<<(fp32?"float32":"float64")<<"\",\"preconditioner\":\""<<(fp32?"FP32 GPU BoomerAMG bridge":"PETSc PCHYPRE GPU BoomerAMG")<<"\","
    <<"\"setup_s\":"<<setup_s<<",\"true_residual\":"<<true_res<<",\"solve_s\":[";
  for(int i=0;i<4;++i)report<<(i?",":"")<<times[i];report<<"],\"iterations\":[";
  for(int i=0;i<4;++i)report<<(i?",":"")<<iterations[i];report<<"]}\n";
  PetscCall(KSPDestroy(&ksp));PetscCall(MatDestroy(&P));PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&work));PetscCall(VecDestroy(&residual));PetscCall(VecDestroy(&x));PetscCall(VecDestroy(&rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc,char **argv) {
  PetscErrorCode status=PetscInitialize(&argc,&argv,nullptr,nullptr);if(status)return int(status);
  try {status=run();} catch(const std::exception &e){std::cerr<<e.what()<<std::endl;status=PETSC_ERR_LIB;}
  PetscFinalize();return int(status);
}
