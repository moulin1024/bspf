#!/usr/bin/env bash
set -euo pipefail
if [[ ! -d /tmp/bspf-nekrs-src ]]; then
  git clone --depth 1 --branch v26.0 https://github.com/Nek5000/nekRS.git /tmp/bspf-nekrs-src
fi
test "$(git -C /tmp/bspf-nekrs-src rev-parse HEAD)" = 96b3cf9e5bacede16568826c04a21bc0fe50dc7d
python - <<'PATCH'
from pathlib import Path
root=Path('/tmp/bspf-nekrs-src')
p=root/'cmake/hypre.cmake'
p.write_text(p.read_text().replace('HYPRE_CUDA_SM=80 90','HYPRE_CUDA_SM=86'))
p=root/'3rd_party/hypre/src/utilities/_hypre_utilities.hpp'
s=p.read_text()
if '#include <thrust/pair.h>' not in s:
    s=s.replace('#include <thrust/count.h>', '#include <thrust/pair.h>\n#include <thrust/iterator/reverse_iterator.h>\n#include <thrust/count.h>')
    p.write_text(s)
p=root/'src/core/elliptic/registerEllipticPreconditionerKernels.cpp'
s=p.read_text()
s=s.replace('if (platform->options.compareArgs(optionsPrefix + "PRECONDITIONER", "SEMFEM")) {\n    const int coarseLevelN',
            'if (platform->options.compareArgs(optionsPrefix + "PRECONDITIONER", "SEMFEM") ||\n      platform->options.compareArgs(optionsPrefix + "MULTIGRID COARSE GRID DISCRETIZATION", "SEMFEM")) {\n    const int coarseLevelN')
p.write_text(s)
PATCH
sdk=${BSPF_HPC_SDK:-/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5}
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran
export PATH="$sdk/comm_libs/mpi/bin:$sdk/cuda/13.2/bin:$PATH"
export CPATH="$sdk/math_libs/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$sdk/math_libs/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$sdk/math_libs/lib64:$sdk/cuda/13.2/lib64:$sdk/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S /tmp/bspf-nekrs-src -B /tmp/bspf-nekrs-build \
 -DCMAKE_INSTALL_PREFIX=/tmp/bspf-nekrs-install \
 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_Fortran_COMPILER=mpifort \
 -DCMAKE_CUDA_COMPILER="$sdk/cuda/13.2/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=86 \
 -DCUDAToolkit_ROOT="$sdk/cuda/13.2" -DCUDA_TOOLKIT_ROOT_DIR="$sdk/cuda/13.2" \
 -DCMAKE_LIBRARY_PATH="$sdk/math_libs/lib64" -DCMAKE_INCLUDE_PATH="$sdk/math_libs/include" \
 -DOCCA_ENABLE_CUDA=ON -DOCCA_ENABLE_HIP=OFF -DOCCA_ENABLE_DPCPP=OFF \
 -DENABLE_HYPRE_GPU=ON -DENABLE_ADIOS=OFF -DNEKRS_BUILD_FLOAT=OFF \
 -DOCCA_CXXFLAGS='-w -O3 -g -march=native' -DOCCA_CUDA_COMPILER_FLAGS='-w -O3 -lineinfo'
cmake --build /tmp/bspf-nekrs-build --target install -j8
