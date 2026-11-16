// Stock MFEM partial-assembly diffusion through libCEED. No custom PDE kernels.
#include "mfem.hpp"
#include "fem/ceed/interface/ceed.hpp"
#include "general/forall.hpp"
#ifdef MFEM_USE_MPI
#include "fem/lor/lor_batched.hpp"
#endif
#include "../../curved_poisson/benchmarks/poisson_campaign/native_common.hpp"
#ifdef MFEM_USE_CUDA
#include <cuda_runtime_api.h>
void RegisterLORGeometryKernel();
#endif
using namespace mfem;
// Reuse MFEM's tensor load kernel with fixed-mesh Jacobians computed during
// coordinate export/setup. This avoids the generic q^3-thread geometry kernel.
class CachedGeometryDomainLFIntegrator : public DomainLFIntegrator {
  const IntegrationRule &rule;
  const Vector &det, &values;
public:
  CachedGeometryDomainLFIntegrator(Coefficient &c,const IntegrationRule &ir,
                                  const Vector &j,const Vector &f)
    : DomainLFIntegrator(c,&ir),rule(ir),det(j),values(f) { }
  void AssembleDevice(const FiniteElementSpace &fes,const Array<int> &markers,
                      Vector &out) override {
    const auto &fe=*fes.GetTypicalFE();
    const auto &maps=fe.GetDofToQuad(rule,DofToQuad::TENSOR);
    DomainLFIntegrator::AssembleKernels::Run(3,maps.ndof,maps.nqpt,
      fes.GetVDim(),fes.GetNE(),maps.ndof,maps.nqpt,fe.GetMapType(),
      markers.Read(),maps.B.Read(),det.Read(),rule.GetWeights().Read(),
      values,out.ReadWrite());
  }
};
#ifdef BSPF_HYPRE_FP32
#include "hypre_fp32_bridge.h"
class FloatHypreBoomerAMG : public Solver {
  const HypreParMatrix *matrix=nullptr;
  void *handle=nullptr;
public:
  FloatHypreBoomerAMG() : Solver(0) { }
  ~FloatHypreBoomerAMG() override { bspf_amg32_destroy(handle); }
  void SetOperator(const mfem::Operator &op) override {
    if(handle) throw std::runtime_error("FP32 AMG operator already initialized");
    matrix=dynamic_cast<const HypreParMatrix*>(&op);
    if(!matrix) throw std::runtime_error("FP32 AMG requires a HYPRE LOR matrix");
    height=op.Height();width=op.Width();
  }
  void Setup() {
    if(bspf_amg32_scalar_bytes()!=4) throw std::runtime_error("AMG is not FP32");
    SparseMatrix diag;matrix->GetDiag(diag);
    handle=bspf_amg32_create(height,diag.NumNonZeroElems(),diag.HostReadI(),diag.HostReadJ(),diag.HostReadData());
    if(!handle) throw std::runtime_error(bspf_amg32_error());
  }
  void Mult(const Vector &x,Vector &y) const override {
    if(!handle) throw std::runtime_error("FP32 AMG not initialized");
    if(bspf_amg32_apply(handle,x.Read(),y.Write())) throw std::runtime_error(bspf_amg32_error());
  }
};
#endif
#ifdef MFEM_USE_MPI
// For a fixed operator, retain the assembled LOR matrix but release the
// element assembly cache. AssembleSystem can recreate it if needed later.
class StaticParLORDiscretization : public ParLORDiscretization {
public:
  using ParLORDiscretization::ParLORDiscretization;
  double AuditVertexCoordinates() {
    const int d=fes_ho.GetMaxElementOrder()+1,n=d*d*d;
    const double *v=batched_lor->GetLORVertexCoordinates().HostRead();
    const double *nodes=poly1d.GetPoints(d-1,BasisType::GaussLobatto);
    Vector point(3);double err=0.;
    for(int el : {0,fes_ho.GetNE()-1}) for(int z=0;z<d;++z) for(int y=0;y<d;++y) for(int x=0;x<d;++x) {
      IntegrationPoint ip;ip.Set3(nodes[x],nodes[y],nodes[z]);
      fes_ho.GetMesh()->GetElementTransformation(el)->Transform(ip,point);
      for(int c=0;c<3;++c) {
        double value=v[3*(x+d*(y+d*z)+n*el)+c];
        if(!std::isfinite(value)) throw std::runtime_error("Nonfinite LOR vertex coordinate");
        err=std::max(err,std::abs(value-point[c]));
      }
    }
    std::cout<<"LOR vertex audit error "<<err<<std::endl;
    if(err>2e-13) throw std::runtime_error("LOR GPU vertex interpolation mismatch");
    return err;
  }
  void ReleaseAssemblyWorkspace() { delete batched_lor;batched_lor=nullptr; }
};
#endif
// CG updates its system operator each repetition. The auxiliary LOR operator
// must remain fixed instead of being replaced by the high-order system.
class FixedPreconditioner : public Solver {
  Solver &inner;
public:
  explicit FixedPreconditioner(Solver &s) : Solver(s.Height()),inner(s) { }
  void SetOperator(const mfem::Operator &op) override {
    if(op.Height()!=Height()) throw std::runtime_error("Preconditioner size mismatch");
  }
  void Mult(const Vector &x,Vector &y) const override { inner.Mult(x,y); }
};

void sync_device(bool gpu) {
#ifdef MFEM_USE_CUDA
  if (gpu) {
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
#else
  if (gpu) throw std::runtime_error("MFEM was built without CUDA");
#endif
}

// Exact Q2 map from turbulence_3d/mms.py; physical, nonperiodic boundaries.
void curved_map(const Vector &s, Vector &x) {
  static const double a[3]={.08,-.06,.07};
  x.SetSize(3);
  for(int i=0;i<3;++i) {
    const int j=(i+1)%3,k=(i+2)%3;
    x[i]=s[i]+a[i]*(1-2*s[i])*16*s[j]*(1-s[j])*s[k]*(1-s[k]);
  }
}
Mat packed(int n) { return n <= 100000 ? Mat(1,n) : Mat((n+1023)/1024,1024); }
void array_json(std::ostream &out, const std::vector<double> &values) {
  out << '['; for(size_t i=0;i<values.size();++i) out << (i ? "," : "") << values[i]; out << ']';
}

int main(int argc, char **argv) { try {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc,argv);Hypre::Init();
  if(Mpi::WorldSize()!=1) throw std::runtime_error("This GPU benchmark requires exactly one MPI rank");
#endif
  if(argc!=10 && argc!=11) {
    std::cerr << "usage: mfem_poisson3d elements degree q export|solve device folder audit_q warmups repeats [jacobi|lor-amg|lor-amg-fp32]\n";
    return 2;
  }
  const int e=std::stoi(argv[1]), p=std::stoi(argv[2]), q=std::stoi(argv[3]);
  const std::string action=argv[4], requested=argv[5], folder=argv[6];
  const int audit_q=std::stoi(argv[7]), warmups=std::stoi(argv[8]), repeats=std::stoi(argv[9]);
  if(e<1||p<1||q<p+1||audit_q<p+1||warmups<0||repeats<1||(action!="export"&&action!="solve"))
    throw std::runtime_error("Invalid benchmark arguments");
  const std::string preconditioner_kind=argc==11 ? argv[10] : "jacobi";
  const bool fp32_amg=preconditioner_kind=="lor-amg-fp32";
  const bool use_lor=preconditioner_kind=="lor-amg" || fp32_amg;
  if(preconditioner_kind!="jacobi" && !use_lor)
    throw std::runtime_error("Unknown preconditioner");
#ifndef BSPF_HYPRE_FP32
  if(fp32_amg) throw std::runtime_error("Build with BSPF_HYPRE_FP32_ROOT for FP32 AMG");
#endif
#ifndef MFEM_USE_MPI
  if(use_lor) throw std::runtime_error("LOR-AMG requires MPI/HYPRE-enabled MFEM");
#endif
  const bool gpu = action=="solve" && requested.rfind("ceed-cuda:",0)==0;
  if(fp32_amg && action=="solve" && !gpu) throw std::runtime_error("FP32 AMG bridge requires CUDA");
  if(action=="solve" && !gpu && requested.rfind("ceed-cpu:",0)!=0)
    throw std::runtime_error("Select an explicit ceed-cuda: or ceed-cpu: backend");
  const auto start=Clock::now();
  Device device(action=="export" ? "cpu" : requested.c_str());
  device.Print();
  std::string actual="export", gpu_name="none";
  if(action=="solve") {
    const char *resource=nullptr;
    if(CeedGetResource(mfem::internal::ceed,&resource)) throw std::runtime_error("CeedGetResource failed");
    actual=resource;
    if(gpu && actual.rfind("/gpu/cuda",0)!=0) throw std::runtime_error("libCEED did not select CUDA");
  }
#ifdef MFEM_USE_CUDA
  if(gpu) {
    cudaDeviceProp prop{}; int id=0;
    if(cudaGetDevice(&id)!=cudaSuccess || cudaGetDeviceProperties(&prop,id)!=cudaSuccess)
      throw std::runtime_error("Unable to query active GPU");
    gpu_name=prop.name;
  }
#endif
  std::vector<std::pair<std::string,size_t>> memory_observations;
  auto memory_snapshot = [&](const std::string &stage) {
#ifdef MFEM_USE_CUDA
    if(gpu) {
      sync_device(true);size_t free_bytes=0,total_bytes=0;
      if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)
        throw std::runtime_error("Unable to query CUDA memory");
      memory_observations.emplace_back(stage,total_bytes-free_bytes);
      std::cout << "memory " << stage << ": used=" << total_bytes-free_bytes
                << " free=" << free_bytes << " bytes" << std::endl;
    }
#endif
  };
  memory_snapshot("startup");
  Mesh serial_mesh=Mesh::MakeCartesian3D(e,e,e,Element::HEXAHEDRON,1.,1.,1.,false);
#ifdef MFEM_USE_MPI
  Array<int> partition(serial_mesh.GetNE());partition=0;
  ParMesh mesh(MPI_COMM_WORLD,serial_mesh,partition.GetData());
#else
  Mesh &mesh=serial_mesh;
#endif
  Mesh reference=Mesh::MakeCartesian3D(e,e,e,Element::HEXAHEDRON,1.,1.,1.,false);
  mesh.SetCurvature(2); mesh.Transform(curved_map);
  H1_FECollection fec(p,3,BasisType::GaussLobatto);
#ifdef MFEM_USE_MPI
  ParFiniteElementSpace fes(&mesh,&fec);
#else
  FiniteElementSpace fes(&mesh,&fec);
#endif
  const IntegrationRule &ir=IntRules.Get(Geometry::CUBE,2*q-1);
  if(ir.GetNPoints()!=q*q*q) throw std::runtime_error("Unexpected quadrature rule");
  const int count=mesh.GetNE()*ir.GetNPoints(), ndofs=fes.GetVSize();
  Vector cached_det(count);cached_det.UseDevice(true);
  double *det_host=cached_det.HostWrite();
  Bundle coordinates{{"x",packed(count)},{"y",packed(count)},{"z",packed(count)},
                     {"s",packed(count)},{"t",packed(count)},{"r",packed(count)},
                     {"bx",packed(ndofs)},{"by",packed(ndofs)},{"bz",packed(ndofs)},
                     {"count",scalar(count)},{"ndofs",scalar(ndofs)}};
  Vector xy(3),st(3);
  double min_jac=std::numeric_limits<double>::max();
  for(int el=0;el<mesh.GetNE();++el) {
    auto *tr=mesh.GetElementTransformation(el), *ref=reference.GetElementTransformation(el);
    for(int j=0;j<ir.GetNPoints();++j) {
      const auto &ip=ir.IntPoint(j); int k=el*ir.GetNPoints()+j;
      tr->Transform(ip,xy); ref->Transform(ip,st); tr->SetIntPoint(&ip);
      det_host[k]=tr->Weight();min_jac=std::min(min_jac,det_host[k]);
      coordinates["x"].v[k]=xy[0];coordinates["y"].v[k]=xy[1];coordinates["z"].v[k]=xy[2];
      coordinates["s"].v[k]=st[0];coordinates["t"].v[k]=st[1];coordinates["r"].v[k]=st[2];
    }
  }
  if(!(min_jac>0)) throw std::runtime_error("Nonpositive geometry Jacobian");
  for(int k=0;k<3;++k) {
    FunctionCoefficient component([k](const Vector &x){return x[k];});
    GridFunction nodal(&fes); nodal.ProjectCoefficient(component);
    const double *values=nodal.HostRead();
    std::copy(values,values+ndofs,coordinates[k==0 ? "bx" : (k==1 ? "by" : "bz")].v.begin());
  }
  if(action=="export") {save(folder+"/coordinates.bin",coordinates);return 0;}
  Bundle samples=load(folder+"/samples.bin");
  for(const auto &key:{"x","y","z","s","t","r","bx","by","bz","count","ndofs"}) {
    same(coordinates.at(key),samples.at(key));
    for(size_t i=0;i<coordinates.at(key).size();++i)
      if(std::abs(coordinates.at(key).v[i]-samples.at(key).v[i])>2e-13)
        throw std::runtime_error("Sample coordinates mismatch");
  }
  if(samples.at("forcing").size()<size_t(count)||samples.at("boundary").size()<size_t(ndofs))
    throw std::runtime_error("Incomplete input samples");
  QuadratureSpace qs(mesh,ir); QuadratureFunction forcing(qs);
  std::copy_n(samples.at("forcing").v.data(),count,forcing.HostWrite());
  QuadratureFunctionCoefficient coefficient(forcing);
  const bool fast_rhs = p+1<=DeviceDofQuadLimits::Get().MAX_D1D && q<=DeviceDofQuadLimits::Get().MAX_Q1D;
#ifdef MFEM_USE_MPI
  ParLinearForm b(&fes);
#else
  LinearForm b(&fes);
#endif
  if(fast_rhs) b.AddDomainIntegrator(new CachedGeometryDomainLFIntegrator(coefficient,ir,cached_det,forcing));
  else b.AddDomainIntegrator(new DomainLFIntegrator(coefficient,&ir));
  b.UseFastAssembly(fast_rhs);
  Array<int> walls(mesh.bdr_attributes.Max()); walls=1;
  Array<int> ess; fes.GetEssentialTrueDofs(walls,ess);
  double load_check=-1.;
  if(fast_rhs && mesh.GetNE()<=8) {
    LinearForm host_load(&fes);host_load.UseFastAssembly(false);
    host_load.AddDomainIntegrator(new DomainLFIntegrator(coefficient,&ir));host_load.Assemble();
    b.Assemble();sync_device(gpu);
    Vector difference(b);difference-=host_load;
    load_check=difference.Norml2()/std::max(host_load.Norml2(),1e-300);
    if(load_check>2e-12) throw std::runtime_error("Device/host RHS assembly mismatch");
  }
  ConstantCoefficient one(1.);
#ifdef MFEM_USE_MPI
  ParBilinearForm a(&fes);
#else
  BilinearForm a(&fes);
#endif
  a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  auto *diffusion=new DiffusionIntegrator(one); diffusion->SetIntRule(&ir);
  a.AddDomainIntegrator(diffusion); a.Assemble();
  memory_snapshot("operator_assembled");
  coordinates.clear();
  double lor_assembly_seconds=0.,amg_setup_seconds=0.,lor_vertex_error=0.;
  bool hypre_gpu=false;std::string hypre_version="none";
  long long lor_nnz=0;
  sync_device(gpu);auto preconditioner_start=Clock::now();
#ifdef MFEM_USE_MPI
  std::unique_ptr<StaticParLORDiscretization> lor_discretization;
#endif
  std::unique_ptr<Solver> preconditioner;
#ifdef MFEM_USE_MPI
  hypre_version=HYPRE_RELEASE_VERSION;hypre_gpu=HypreUsingGPU();
  if(use_lor) {
    if(gpu && !hypre_gpu) throw std::runtime_error("HYPRE is not using GPU execution");
#ifdef MFEM_USE_CUDA
    if(gpu) RegisterLORGeometryKernel();
#endif
    lor_discretization=std::make_unique<StaticParLORDiscretization>(a,ess);
    memory_snapshot("lor_assembled");
    lor_vertex_error=lor_discretization->AuditVertexCoordinates();
    lor_discretization->ReleaseAssemblyWorkspace();
    memory_snapshot("lor_workspace_released");
    auto *matrix=dynamic_cast<const HypreParMatrix*>(lor_discretization->GetAssembledSystem().Ptr());
    if(!matrix) throw std::runtime_error("Expected HYPRE LOR matrix");
    lor_nnz=matrix->NNZ();
#ifdef BSPF_HYPRE_FP32
    if(fp32_amg) {
      auto lor=std::make_unique<LORSolver<FloatHypreBoomerAMG>>(*lor_discretization);
      sync_device(gpu);lor_assembly_seconds=elapsed(preconditioner_start);
      auto amg_start=Clock::now();lor->GetSolver().Setup();
      sync_device(gpu);amg_setup_seconds=elapsed(amg_start);
      preconditioner=std::move(lor);
    } else
#endif
    {
    auto lor=std::make_unique<LORSolver<HypreBoomerAMG>>(*lor_discretization);
    sync_device(gpu);lor_assembly_seconds=elapsed(preconditioner_start);
    lor->GetSolver().SetPrintLevel(1);
    Vector zero(fes.GetTrueVSize()),scratch(fes.GetTrueVSize());
    zero.UseDevice(true);scratch.UseDevice(true);zero=0.;scratch=0.;
    sync_device(gpu);auto amg_start=Clock::now();
    lor->GetSolver().Setup(zero,scratch);
    sync_device(gpu);amg_setup_seconds=elapsed(amg_start);
    preconditioner=std::move(lor);
    }
  } else
#endif
  {preconditioner=std::make_unique<OperatorJacobiSmoother>(a,ess);}
  sync_device(gpu);double preconditioner_setup_seconds=elapsed(preconditioner_start);
  memory_snapshot("preconditioner_ready");
  FixedPreconditioner fixed_preconditioner(*preconditioner);
#ifdef MFEM_USE_MPI
  ParGridFunction x(&fes);CGSolver cg(MPI_COMM_WORLD);
#else
  GridFunction x(&fes);CGSolver cg;
#endif
  x.UseDevice(true); Vector X,B; OperatorPtr A;
  Vector initial_boundary(ndofs);initial_boundary.UseDevice(true);
  double *initial=initial_boundary.HostWrite();std::fill(initial,initial+ndofs,0.);
  for(int i=0;i<ess.Size();++i) initial[ess[i]]=samples.at("boundary").v[ess[i]];
  initial_boundary.Read();cached_det.Read();forcing.Read();
  cg.SetRelTol(1e-12);cg.SetAbsTol(0.);cg.SetMaxIter(20000);cg.SetPrintLevel(0);
  cg.SetPreconditioner(fixed_preconditioner);
  sync_device(gpu); const double setup_seconds=elapsed(start);
  double last_linear_seconds=0.,last_rhs_seconds=0.;
  auto solve = [&]() {
    sync_device(gpu); auto t=Clock::now();
    x=initial_boundary;
    sync_device(gpu);auto rhs_start=Clock::now();
    b.Assemble();
    sync_device(gpu);last_rhs_seconds=elapsed(rhs_start);
    a.FormLinearSystem(ess,x,b,A,X,B);
    cg.SetOperator(*A);
    sync_device(gpu);const auto linear_start=Clock::now();cg.Mult(B,X);sync_device(gpu);
    last_linear_seconds=elapsed(linear_start);
    a.RecoverFEMSolution(X,b,x);
    sync_device(gpu); double seconds=elapsed(t);
    if(!cg.GetConverged()) throw std::runtime_error("CG did not converge");
    return seconds;
  };
  double first=solve(), first_linear=last_linear_seconds;int first_iterations=cg.GetNumIterations();
  memory_snapshot("first_solve");
  std::cout << "first solve " << first << " seconds, iterations " << first_iterations << std::endl;
  for(int i=0;i<warmups;++i) solve();
  std::vector<double> times, linear_times, rhs_times;
  for(int i=0;i<repeats;++i) {times.push_back(solve());linear_times.push_back(last_linear_seconds);rhs_times.push_back(last_rhs_seconds);
    std::cout << "repeat " << i+1 << ": " << times.back() << " seconds" << std::endl;}
  Vector residual(B.Size()); residual.UseDevice(true); A->Mult(X,residual); residual-=B;
  double relres=residual.Norml2()/std::max(B.Norml2(),1e-300);
  if(!std::isfinite(relres)||relres>2e-11) throw std::runtime_error("True residual check failed");
  memory_snapshot("residual_checked");
  x.HostRead();
  const auto *restriction=fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
  Vector local(restriction->Height());local.UseDevice(true);restriction->Mult(x,local);
  Mat element_solution(mesh.GetNE(),(p+1)*(p+1)*(p+1));
  std::copy_n(local.HostRead(),local.Size(),element_solution.v.begin());
  Mat nodes(1,p+1);std::copy_n(poly1d.GetPoints(p,BasisType::GaussLobatto),p+1,nodes.v.begin());
  Mat origins(mesh.GetNE(),3);IntegrationPoint zero;zero.Set3(0.,0.,0.);
  for(int el=0;el<mesh.GetNE();++el) {
    auto *tr=reference.GetElementTransformation(el);tr->Transform(zero,st);tr->SetIntPoint(&zero);
    for(int i=0;i<3;++i) {
      origins.v[3*el+i]=st[i];
      for(int j=0;j<3;++j)
        if(std::abs(tr->Jacobian()(i,j)-(i==j ? 1./e : 0.))>1e-13)
          throw std::runtime_error("Unexpected reference element orientation");
    }
  }
  // Independent native probes verify the Python tensor reconstruction ordering.
  const int stride=std::max(1,mesh.GetNE()/32), nprobe=(mesh.GetNE()+stride-1)/stride;
  Mat probes(nprobe,5);int probe=0;IntegrationPoint ip;ip.Set3(.213,.371,.627);
  for(int el=0;el<mesh.GetNE();el+=stride,++probe) {
    probes.v[5*probe]=el;probes.v[5*probe+1]=ip.x;probes.v[5*probe+2]=ip.y;probes.v[5*probe+3]=ip.z;
    probes.v[5*probe+4]=x.GetValue(el,ip);
  }
  save(folder+"/solution_local.bin",{{"coefficients",element_solution},{"nodes",nodes},
       {"origins",origins},{"probes",probes},{"elements_per_axis",scalar(e)}});
  std::ofstream sol(folder+"/solution.gf");sol<<std::setprecision(17);x.Save(sol);
  memory_snapshot("solution_exported");
  int major,minor,patch;bool release;CeedGetVersion(&major,&minor,&patch,&release);
  std::ofstream report(folder+"/timing.json");report<<std::setprecision(17)
    <<"{\"backend_requested\":\""<<requested<<"\",\"backend_actual\":\""<<actual
    <<"\",\"gpu\":\""<<gpu_name<<"\",\"mfem_version\":\""<<GetVersionStr()
    <<"\",\"ceed_version\":\""<<major<<'.'<<minor<<'.'<<patch
    <<"\",\"precision\":\"float64\",\"preconditioner\":\""
    <<(fp32_amg ? "LORSolver<FloatHypreBoomerAMG>" : use_lor ? "LORSolver<HypreBoomerAMG>" : "OperatorJacobiSmoother")<<"\","
    <<"\"preconditioner_precision\":\""<<(fp32_amg ? "float32" : "float64")<<"\","
    <<"\"operator_precision\":\"float64\",\"cg_precision\":\"float64\","
    <<"\"preconditioner_scalar_bytes\":"<<(fp32_amg ? 4 : 8)<<","
    <<"\"preconditioner_conversion\":\""<<(fp32_amg ? "CUDA FP64-to-FP32 residual and FP32-to-FP64 correction" : "none")<<"\","
    <<"\"hypre_version\":\""<<hypre_version<<"\",\"hypre_gpu\":"<<(hypre_gpu ? "true" : "false")<<","
    <<"\"preconditioner_setup_seconds\":"<<preconditioner_setup_seconds<<","
    <<"\"lor_assembly_seconds\":"<<lor_assembly_seconds<<",\"amg_setup_seconds\":"<<amg_setup_seconds<<","
    <<"\"lor_nnz\":"<<lor_nnz<<",\"lor_vertex_error\":"<<lor_vertex_error<<","
    <<"\"dimension\":3,\"assembly\":\"PARTIAL\",\"dofs\":"<<ndofs<<",\"free_dofs\":"<<ndofs-ess.Size()
    <<",\"rhs_assembly\":\""<<(fast_rhs ? "mfem_device_cached_geometry" : "mfem_host")<<"\""
    <<",\"rhs_host_device_relative_error\":"<<load_check
    <<",\"boundary_initialization\":\"device_copy\""
    <<",\"setup_seconds\":"<<setup_seconds<<",\"first_solve_seconds\":"<<first
    <<",\"first_iterations\":"<<first_iterations<<",\"iterations\":"<<cg.GetNumIterations()
    <<",\"relative_residual\":"<<relres<<",\"min_jacobian\":"<<min_jac
    <<",\"warmups\":"<<warmups<<",\"repeats\":"<<repeats<<",\"samples_seconds\":";
  array_json(report,times);report<<",\"rhs_samples_seconds\":";array_json(report,rhs_times);
  report<<",\"first_linear_seconds\":"<<first_linear<<",\"linear_samples_seconds\":";
  array_json(report,linear_times);report<<",\"device_used_bytes_observed\":{";
  for(size_t i=0;i<memory_observations.size();++i)
    report<<(i ? "," : "")<<'"'<<memory_observations[i].first<<"\":"<<memory_observations[i].second;
  report<<"}}\n";
  return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;} }
