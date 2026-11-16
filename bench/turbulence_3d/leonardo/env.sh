#!/bin/bash
module load gcc/12.2.0 cuda/12.6 cmake/3.27.9 openmpi/4.1.6--gcc--12.2.0-cuda-12.2
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran
export CC=gcc CXX=g++ FC=gfortran
export BSPF_TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.toolchains/leonardo"
export LD_LIBRARY_PATH="$BSPF_TOOLCHAIN_ROOT/src/libceed/lib:$BSPF_TOOLCHAIN_ROOT/install/nekrs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BSPF_NEKRS_PREFIX="$BSPF_TOOLCHAIN_ROOT/install/nekrs"
export BSPF_NATIVE_LIBRARY_PATH="$BSPF_NEKRS_PREFIX/occa/lib:$LD_LIBRARY_PATH"
export BSPF_POISSON_BINARY="$BSPF_TOOLCHAIN_ROOT/build/bspf-poisson/bspf_poisson3d"
