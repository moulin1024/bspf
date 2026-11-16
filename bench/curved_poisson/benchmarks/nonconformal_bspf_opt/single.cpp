// Common single-call adapter: no warmup, no repetitions, unweighted RHS inputs.
#include "poisson_solver.hpp"
#include <sys/resource.h>
int main(int argc,char**argv){
  if(argc!=3)return 2;
  setenv("VECLIB_MAXIMUM_THREADS","1",1);setenv("OPENBLAS_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);
  try{
    auto start=Clock::now();Poisson p(argv[1]);double loading=elapsed(start);
    if(!p.p.count("forcing")||!p.p.count("weights"))throw std::runtime_error("Campaign requires unweighted forcing");
    start=Clock::now();
    Mat x=p.solve();double seconds=elapsed(start);
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    save(std::string(argv[2])+".bin",{{"solution",x}});
    std::ofstream f(std::string(argv[2])+".json");
    f<<std::setprecision(17)<<"{\"solve_calls\":1,\"warmups\":0,\"load_seconds\":"<<loading
     <<",\"solve_seconds\":"<<seconds<<",\"iterations\":"<<p.iterations
     <<",\"relative_residual\":"<<p.residual<<",\"preconditioned_defect\":"<<p.preconditioned_defect
     <<",\"refinements\":"<<p.refinements<<",\"peak_rss_bytes\":"<<usage.ru_maxrss
     <<",\"rhs_weighting_inside_timer\":true,\"defect_history\":[";
    for(size_t i=0;i<p.defect_history.size();++i)f<<(i?",":"")<<p.defect_history[i];
    f<<"]}\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
