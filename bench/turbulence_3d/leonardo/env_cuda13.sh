#!/bin/bash
# Load in a fresh shell: the CUDA 12/GCC 12 modules have conflicting dependencies.
module load profile/candidate
module load cuda/13.0
module load gcc/13.4.0
export BSPF_TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.toolchains/leonardo"
export BSPF_POISSON_BINARY="$BSPF_TOOLCHAIN_ROOT/build/bspf-poisson-cuda13/bspf_poisson3d"
# Existing CuPy is built for CUDA 12; retain its runtime for independent audits.
# The native solver links CUDA 13/cuBLAS 13 by their distinct SONAMEs.
export CUDA_PATH=/leonardo/prod/spack/7.3/install/linux-icelake/cuda-12.6.0-fawaeoadqhqzj4i3im3kxugcoiohvstp
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BSPF_NATIVE_LIBRARY_PATH="$LD_LIBRARY_PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
