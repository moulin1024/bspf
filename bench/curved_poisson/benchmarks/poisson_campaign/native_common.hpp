#pragma once
// Matrix/CBLAS/bundle utilities copied from frozen internal-wave C++ baseline v1.
// Standalone float64 Galerkin Boussinesq evolution. No Python or JAX runtime.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Mat {
  size_t r=0,c=0; std::vector<double> v;
  Mat()=default; Mat(size_t r_,size_t c_):r(r_),c(c_),v(r_*c_){}
  size_t size()const{return v.size();}
};
using Bundle=std::map<std::string,Mat>;
template<class T> T read(std::istream& f){T v;f.read(reinterpret_cast<char*>(&v),sizeof(v));if(!f)throw std::runtime_error("Truncated input");return v;}
template<class T> void write(std::ostream& f,T v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
Bundle load(const std::string& path){
  std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);
  if(!f||std::memcmp(magic,"BSPFCPP1",8))throw std::runtime_error("Invalid bundle: "+path);
  uint32_t count=read<uint32_t>(f);if(count>1000)throw std::runtime_error("Invalid array count");Bundle b;
  for(uint32_t i=0;i<count;++i){auto len=read<uint32_t>(f);if(!len||len>200)throw std::runtime_error("Invalid name");std::string name(len,' ');f.read(name.data(),len);
    uint64_t r=read<uint64_t>(f),c=read<uint64_t>(f);if(!r||!c||r>100000||c>100000||r*c>100000000)throw std::runtime_error("Invalid dimensions");
    Mat m(r,c);f.read(reinterpret_cast<char*>(m.v.data()),m.size()*8);if(!f)throw std::runtime_error("Truncated array");
    for(double x:m.v)if(!std::isfinite(x))throw std::runtime_error("Nonfinite input array");
    if(!b.emplace(name,std::move(m)).second)throw std::runtime_error("Duplicate array");}
  return b;
}
void save(const std::string& path,const Bundle& b){
  std::ofstream f(path,std::ios::binary);f.write("BSPFCPP1",8);write<uint32_t>(f,b.size());
  for(auto& [name,m]:b){write<uint32_t>(f,name.size());f.write(name.data(),name.size());write<uint64_t>(f,m.r);write<uint64_t>(f,m.c);f.write(reinterpret_cast<const char*>(m.v.data()),m.size()*8);}
  if(!f)throw std::runtime_error("Could not write "+path);
}
Mat scalar(double value){Mat a(1,1);a.v[0]=value;return a;}
void same(const Mat&a,const Mat&b){if(a.r!=b.r||a.c!=b.c)throw std::runtime_error("Matrix shape mismatch");}
void axpy(Mat&y,const Mat&x,double scale=1){same(y,x);for(size_t i=0;i<y.size();++i)y.v[i]+=scale*x.v[i];}
Mat added(const Mat&a,const Mat&b,double s=1){Mat out=a;axpy(out,b,s);return out;}
bool finite(const Mat&a){return std::all_of(a.v.begin(),a.v.end(),[](double v){return std::isfinite(v);});}
inline std::string blas_library() {
  if (const char* path = std::getenv("BSPF_BLAS_LIBRARY")) return path;
#ifdef __APPLE__
  return "/System/Library/Frameworks/Accelerate.framework/Accelerate";
#else
  return "libblas.so.3";
#endif
}
inline long peak_rss_bytes(long rss) {
#ifdef __APPLE__
  return rss;
#else
  return rss * 1024;
#endif
}
struct Blas {
  using Gemm=void(*)(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
  using Dot=double(*)(int,const double*,int,const double*,int);
  void* handle=nullptr;Gemm gemm;Dot ddot;
  explicit Blas(const std::string& path){handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());
    gemm=reinterpret_cast<Gemm>(dlsym(handle,"cblas_dgemm"));ddot=reinterpret_cast<Dot>(dlsym(handle,"cblas_ddot"));
    if(!gemm||!ddot)throw std::runtime_error("BLAS symbols missing");
    auto set_threads=reinterpret_cast<void(*)(int)>(dlsym(handle,"openblas_set_num_threads"));if(set_threads)set_threads(1);
  }
  ~Blas(){if(handle)dlclose(handle);}
  Mat mm(const Mat&a,const Mat&b,bool ta=false,bool tb=false)const{
    size_t m=ta?a.c:a.r,k=ta?a.r:a.c,kb=tb?b.c:b.r,n=tb?b.r:b.c;if(k!=kb)throw std::runtime_error("GEMM dimensions differ");
    Mat out(m,n);gemm(101,ta?112:111,tb?112:111,m,n,k,1.,a.v.data(),a.c,b.v.data(),b.c,0.,out.v.data(),n);return out;
  }
  double dot(const Mat&a,const Mat&b)const{same(a,b);return ddot(a.size(),a.v.data(),1,b.v.data(),1);}
  double norm(const Mat&a)const{double d=dot(a,a);return d>=0?std::sqrt(d):std::numeric_limits<double>::quiet_NaN();}
};
