#!/usr/bin/env bash
set -euo pipefail
sdk=${BSPF_HPC_SDK:-/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5}
export CPATH="$sdk/math_libs/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$sdk/math_libs/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export OMPI_CC=gcc OMPI_CXX=g++
export PKG_CONFIG_PATH=/tmp/petsc-3.26.0/arch-bspf-cuda/lib/pkgconfig
cmake -S "$(dirname "$0")" -B /tmp/bspf-petsc-fd8-build \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$sdk/cuda/13.2/bin/nvcc" \
 -DCUDAToolkit_ROOT="$sdk/cuda/13.2" -DMPI_CXX_COMPILER="$sdk/comm_libs/mpi/bin/mpicxx"
cmake --build /tmp/bspf-petsc-fd8-build -j4
