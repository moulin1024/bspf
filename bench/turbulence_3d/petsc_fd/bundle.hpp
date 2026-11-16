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
namespace fd_io {
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
}
