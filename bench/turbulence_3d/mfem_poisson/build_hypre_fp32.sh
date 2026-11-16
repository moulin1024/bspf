#!/usr/bin/env bash
set -euo pipefail
# HYPRE generates CUDA sources in its source tree; keep that copy in /tmp.
if [[ ! -d /tmp/bspf-hypre-fp32-source/src ]]; then
  mkdir -p /tmp/bspf-hypre-fp32-source
  cp -a "${BSPF_HYPRE_SOURCE:-/home/moulin/Workspace/hypre}/src" /tmp/bspf-hypre-fp32-source/
  cp "${BSPF_HYPRE_SOURCE:-/home/moulin/Workspace/hypre}/LICENSE-MIT" "${BSPF_HYPRE_SOURCE:-/home/moulin/Workspace/hypre}/LICENSE-APACHE" "${BSPF_HYPRE_SOURCE:-/home/moulin/Workspace/hypre}/NOTICE" /tmp/bspf-hypre-fp32-source/
fi
sdk=${BSPF_HPC_SDK:-/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5}
export OMPI_CC=gcc OMPI_CXX=g++
export CPATH="$sdk/math_libs/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="/tmp/bspf-libceed/lib:$sdk/math_libs/lib64:$sdk/cuda/13.2/lib64:$sdk/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S /tmp/bspf-hypre-fp32-source/src -B /tmp/bspf-hypre-fp32-build \
  -DCMAKE_INSTALL_PREFIX=/tmp/bspf-hypre-fp32-install -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER="$sdk/cuda/13.2/bin/nvcc" -DCUDAToolkit_ROOT="$sdk/cuda/13.2" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMPI_C_COMPILER="$sdk/comm_libs/mpi/bin/mpicc" -DMPI_CXX_COMPILER="$sdk/comm_libs/mpi/bin/mpicxx" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DHYPRE_ENABLE_SINGLE=ON -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_MPI=ON -DHYPRE_ENABLE_UNIFIED_MEMORY=OFF \
  -DHYPRE_ENABLE_GPU_AWARE_MPI=OFF -DHYPRE_ENABLE_UMPIRE=OFF \
  -DHYPRE_BUILD_TESTS=OFF -DHYPRE_BUILD_EXAMPLES=OFF -DHYPRE_ENABLE_FORTRAN=OFF
cmake --build /tmp/bspf-hypre-fp32-build -j8
cmake --install /tmp/bspf-hypre-fp32-build
# Reuse the existing FP64 MFEM stack; only the private preconditioner is FP32.
cmake -S bench/turbulence_3d/mfem_poisson -B /tmp/bspf-mfem-poisson3d-fp32 \
  -DMFEM_DIR=/tmp/bspf-mfem-hypre86 -DCUDAToolkit_ROOT="$sdk/cuda/13.2" \
  -DCMAKE_BUILD_TYPE=Release -DBSPF_HYPRE_FP32_ROOT=/tmp/bspf-hypre-fp32-install \
  -DMPI_CXX_COMPILER="$sdk/comm_libs/mpi/bin/mpicxx"
cmake --build /tmp/bspf-mfem-poisson3d-fp32 -j4
