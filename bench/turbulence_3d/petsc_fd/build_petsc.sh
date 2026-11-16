#!/usr/bin/env bash
set -euo pipefail
sdk=${BSPF_HPC_SDK:-/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5}
export OMPI_CC=gcc OMPI_CXX=g++ OPENBLAS_NUM_THREADS=1
export PATH="$sdk/cuda/13.2/bin:$sdk/comm_libs/mpi/bin:$PATH"
export CPATH="$sdk/math_libs/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$sdk/math_libs/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$sdk/math_libs/lib64:$sdk/cuda/13.2/lib64:$sdk/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self
export PETSC_DIR=/tmp/petsc-3.26.0 PETSC_ARCH=arch-bspf-cuda
if [[ ! -d "$PETSC_DIR" ]]; then
  curl -fL --retry 2 https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.26.0.tar.gz -o /tmp/bspf-petsc-3.26.0.tar.gz
  tar -xzf /tmp/bspf-petsc-3.26.0.tar.gz -C /tmp
fi
cd "$PETSC_DIR"
./configure PETSC_ARCH="$PETSC_ARCH" --with-debugging=0 --with-shared-libraries=0 \
  --with-cc="$sdk/comm_libs/mpi/bin/mpicc" --with-cxx="$sdk/comm_libs/mpi/bin/mpicxx" --with-fc=0 \
  --with-cuda=1 --with-cuda-dir="$sdk/cuda/13.2" --with-cuda-arch=86 \
  --with-hypre-dir=/tmp/bspf-hypre-gpu-install --with-precision=double \
  --with-blaslapack-lib=-lopenblas --with-mpi=1 --with-mpi-f90module=0 \
  --with-cxx-dialect=C++17 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 CUDAOPTFLAGS=-O3
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" -j8 all
