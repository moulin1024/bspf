// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/cuda_poisson.hpp"
#include "../../curved_poisson/benchmarks/poisson_campaign/native_common.hpp"
#include <cuda_runtime_api.h>
std::vector<double> column_major(const Mat& a){std::vector<double> v(a.size());for(size_t j=0;j<a.c;++j)for(size_t i=0;i<a.r;++i)v[i+a.r*j]=a.v[i*a.c+j];return v;}
int main(int argc,char** argv){
 try{
  if(argc<3||argc>4)throw std::runtime_error("usage: bspf_poisson3d input.bin output_prefix [check]");
  auto p=load(argv[1]);const std::string output=argv[2];bspf::cuda::Poisson3DData d;
  d.n=p.at("B").c;d.q=p.at("B").r;d.basis=column_major(p.at("B"));d.derivative=column_major(p.at("D"));d.eigenvectors=column_major(p.at("V"));d.eigenvalues=p.at("lambda").v;
  const char* names[]={"xx","xy","xz","yy","yz","zz"};for(int i=0;i<6;++i)d.metric[i]=std::move(p.at(names[i]).v);d.measure=std::move(p.at("measure").v);
  auto start=Clock::now();bspf::cuda::GalerkinPoisson3D solver(d);double setup=elapsed(start);
  if(argc==4){
   Bundle out;auto v=solver.apply_host(p.at("probe").v);Mat a(d.n*d.n,d.n);a.v=std::move(v);out["apply"]=a;
   a.v=solver.precondition_host(p.at("probe").v);out["precondition"]=a;save(output+"_check.bin",out);
  }
  solver.set_rhs(p.at("forcing").v,p.at("boundary").v);
  bspf::cuda::Poisson3DOptions options;
  if(p.count("refresh"))options.residual_refresh=int(p.at("refresh").v.at(0));
  if(p.count("max_iterations"))options.max_iterations=int(p.at("max_iterations").v.at(0));
  std::vector<bspf::cuda::Poisson3DStats> stats;
  for(int i=0;i<4;++i)stats.push_back(solver.solve(options));
  Mat u(d.n*d.n,d.n);u.v=solver.solution();save(output+".bin",{{"solution",u}});
  cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,0);
  std::ofstream f(output+".json");f<<std::setprecision(17)<<"{\"gpu\":\""<<prop.name<<"\",\"n\":"<<d.n<<",\"q\":"<<d.q<<",\"dofs\":"<<size_t(d.n)*d.n*d.n<<",\"free_dofs\":"<<size_t(d.n-2)*(d.n-2)*(d.n-2)<<",\"operator_precision\":\"float64\",\"cg_precision\":\"float64\",\"preconditioner_precision\":\"float32\",\"tf32\":false,\"device_bytes\":"<<solver.device_bytes()<<",\"parity_optimized\":"<<(solver.parity_optimized()?"true":"false")<<",\"shared_contractions\":"<<(solver.shared_contractions()?"true":"false")<<",\"fused_tensor_layout\":"<<(solver.fused_tensor_layout()?"true":"false")<<",\"device_setup_s\":"<<setup<<",\"runs\":[";
  for(size_t i=0;i<stats.size();++i){auto&s=stats[i];if(i)f<<",";f<<"{\"iterations\":"<<s.iterations<<",\"residual_replacements\":"<<s.residual_replacements<<",\"relative_residual\":"<<s.relative_residual<<",\"cg_s\":"<<s.cg_seconds<<",\"wall_s\":"<<s.solve_seconds<<"}";}
  f<<"]}\n";if(!f)throw std::runtime_error("result write failed");
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
