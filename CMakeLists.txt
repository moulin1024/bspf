# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.20)
project(bspf VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BSPF_PYTHON "Build the nanobind module bspf_cpp._core" ON)
option(BSPF_TESTS "Build the C++ tests" ON)
option(BSPF_NATIVE "Compile for the host CPU (-march=native / -mcpu=native)" ON)
option(BSPF_BLAS "Use a CBLAS dgemm (Accelerate, OpenBLAS, ...) for the large dense products, if found" ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTW3_LIBRARY fftw3 REQUIRED)

add_library(bspf STATIC src/bspline.cpp src/endpoints.cpp src/fft.cpp src/line.cpp src/op.cpp src/grid.cpp src/galerkin.cpp)
target_include_directories(bspf PUBLIC include ${FFTW3_INCLUDE_DIR})
target_link_libraries(bspf PUBLIC Eigen3::Eigen ${FFTW3_LIBRARY})
target_compile_options(bspf PRIVATE -Wall -Wextra)
if(BSPF_BLAS)
  # Optional: the Galerkin correction's GEMMs run 3-6x faster in an optimized BLAS
  # (on macOS, Accelerate is part of the OS). Without one, Eigen's kernels are used.
  find_package(BLAS)
  if(BLAS_FOUND)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_cxx_source_compiles("
      extern \"C\" void cblas_dgemm(int, int, int, int, int, int, double, const double*, int,
                                     const double*, int, double, double*, int);
      int main() { double a = 2, b = 3, c = 0; cblas_dgemm(102, 111, 111, 1, 1, 1, 1.0, &a, 1, &b, 1, 0.0, &c, 1);
                   return c == 6 ? 0 : 1; }" BSPF_HAVE_CBLAS)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(BSPF_HAVE_CBLAS)
      target_link_libraries(bspf PUBLIC ${BLAS_LIBRARIES})
      target_compile_definitions(bspf PRIVATE BSPF_HAVE_CBLAS)
    endif()
  endif()
endif()
message(STATUS "bspf: CBLAS dgemm ${BSPF_HAVE_CBLAS}")
if(BSPF_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
  check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
  if(HAS_MARCH_NATIVE)
    target_compile_options(bspf PUBLIC -march=native)
  elseif(HAS_MCPU_NATIVE)
    target_compile_options(bspf PUBLIC -mcpu=native)
  endif()
endif()

if(BSPF_TESTS)
  enable_testing()
  add_executable(test_bspf tests/cpp/test_bspf.cpp)
  target_link_libraries(test_bspf PRIVATE bspf)
  add_test(NAME test_bspf COMMAND test_bspf)
  add_executable(bench_bspf bench/bench_bspf.cpp)
  target_link_libraries(bench_bspf PRIVATE bspf)
endif()

if(BSPF_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC src/python_module.cpp)
  target_link_libraries(_core PRIVATE bspf)
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/bspf_cpp)
endif()
