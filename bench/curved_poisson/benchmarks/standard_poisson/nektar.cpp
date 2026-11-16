// Native single-solve benchmark: one timed solve, no warm-up or repetition.
// Uses production Nektar++ ContField::HelmSolve; times in-process solves.
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstdlib>
// Reuse the binary bundle format; its unused BLAS wrapper would otherwise
// collide with Nektar's global Blas namespace. No BSPF algebra is used here.
#define Blas BspfBundleBlas
#include "../poisson_campaign/native_common.hpp"
#undef Blas
using namespace Nektar;
Mat copy_array(const Array<OneD,NekDouble>& a){
  size_t n=a.size();Mat m(n>100000?n/64:1,n>100000?64:n);
  if(m.size()!=n)throw std::runtime_error("Coordinate export shape");
  std::copy(a.data(),a.data()+n,m.v.begin());return m;
}
void load_array(const Mat& m,Array<OneD,NekDouble>& a){
  if(m.size()!=a.size())throw std::runtime_error("Sample size mismatch");
  std::copy(m.v.begin(),m.v.end(),a.data());
}
void check_coordinates(const Mat& m,const Array<OneD,NekDouble>& a){
  if(m.size()!=a.size())throw std::runtime_error("Coordinate size mismatch");
  for(size_t i=0;i<m.size();++i)if(std::abs(m.v[i]-a[i])>1e-13)
    throw std::runtime_error("Sample coordinates do not match Nektar geometry");
}
// Expose only factor construction from the production ContField implementation.
// GetGlobalLinSys initialises DirectStaticCond and its LAPACK factors without solving.
class PreparedField : public MultiRegions::ContField {
 public:
  using MultiRegions::ContField::ContField;
  void Prepare(const StdRegions::ConstFactorMap& factors){
    MultiRegions::GlobalLinSysKey key(StdRegions::eHelmholtz,m_locToGloMap,factors);
    GetGlobalLinSys(key);
  }
};
int main(int argc,char**argv){try{
  setenv("VECLIB_MAXIMUM_THREADS","1",1);setenv("OPENBLAS_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);
  auto start=Clock::now();auto session=LibUtilities::SessionReader::CreateInstance(argc,argv);
  auto graph=SpatialDomains::MeshGraphIO::Read(session);
  auto exp=MemoryManager<PreparedField>::AllocateSharedPtr(session,graph,session->GetVariable(0));
  int nq=exp->GetTotPoints();Array<OneD,NekDouble> x(nq),y(nq),z(nq,0.),f(nq);
  exp->GetCoords(x,y);
  auto boundaries=exp->GetBndCondExpansions();
  Bundle coordinates{{"x",copy_array(x)},{"y",copy_array(y)}};
  for(int i=0;i<boundaries.size();++i){
    auto edge=boundaries[i];int n=edge->GetTotPoints();Array<OneD,NekDouble> bx(n),by(n),bz(n);
    edge->GetCoords(bx,by,bz);coordinates["bx"+std::to_string(i)]=copy_array(bx);coordinates["by"+std::to_string(i)]=copy_array(by);
  }
  if(std::getenv("BSPF_EXPORT_COORDINATES")){
    std::ofstream param("reference_coordinates.txt");param<<std::setprecision(17);
    for(int e=0;e<exp->GetExpSize();++e){auto el=exp->GetExp(e);auto z0=el->GetBasis(0)->GetZ();auto z1=el->GetBasis(1)->GetZ();for(int j=0;j<z1.size();++j)for(int i=0;i<z0.size();++i)param<<el->GetGeom()->GetGlobalID()<<' '<<(z0[i]+1)/2<<' '<<(z1[j]+1)/2<<'\n';}
    save("coordinates.bin",coordinates);session->Finalise();return 0;
  }
  Bundle samples=load("nektar_samples.bin");
  check_coordinates(samples.at("x"),x);check_coordinates(samples.at("y"),y);
  load_array(samples.at("forcing"),f);Array<OneD,NekDouble> exact(nq);load_array(samples.at("exact"),exact);
  for(int i=0;i<boundaries.size();++i){
    auto edge=boundaries[i];std::string key=std::to_string(i);
    for(const std::string prefix:{"bx","by"}){
      const Mat& a=coordinates.at(prefix+key);const Mat& b=samples.at(prefix+key);
      same(a,b);for(size_t j=0;j<a.size();++j)if(std::abs(a.v[j]-b.v[j])>1e-13)throw std::runtime_error("Boundary coordinate mismatch");
    }
    // Same Dirichlet projection as DisContField::v_EvaluateBoundaryConditions.
    load_array(samples.at("boundary"+key),edge->UpdatePhys());
    // Projection is performed inside the timed sampled-RHS-to-solution region.
  }
  StdRegions::ConstFactorMap factors;factors[StdRegions::eFactorLambda]=0.;
  exp->Prepare(factors);
  double setup=elapsed(start);
  Vmath::Zero(exp->GetNcoeffs(),exp->UpdateCoeffs(),1);
  start=Clock::now();for(int i=0;i<boundaries.size();++i){auto edge=boundaries[i];edge->FwdTransBndConstrained(edge->GetPhys(),edge->UpdateCoeffs());}exp->HelmSolve(f,exp->UpdateCoeffs(),factors);double first=elapsed(start);
  int batch=1;std::vector<double> times{first};
  std::ofstream coeff("coefficients.txt");coeff<<std::setprecision(17);
  for(int e=0;e<exp->GetExpSize();++e){auto element=exp->GetExp(e);coeff<<element->GetGeom()->GetGlobalID()<<' '<<element->GetNcoeffs();for(int k=0;k<element->GetNcoeffs();++k)coeff<<' '<<exp->GetCoeffs()[exp->GetCoeff_Offset(e)+k];coeff<<'\n';}
  exp->BwdTrans(exp->GetCoeffs(),exp->UpdatePhys());
  std::ofstream report("timing.json");report<<std::setprecision(17)<<"{\"solve_calls\":1,\"setup_seconds\":"<<setup<<",\"first_solve_seconds\":"<<first<<",\"native_l2\":"<<exp->L2(exp->GetPhys(),exact)<<",\"local_coefficients\":"<<exp->GetNcoeffs()<<",\"batch\":"<<batch<<",\"samples_seconds\":[";
  for(size_t i=0;i<times.size();++i)report<<(i?",":"")<<times[i];report<<"]}\n";
  session->Finalise();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
