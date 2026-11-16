#!/usr/bin/env bash
set -euo pipefail
sdk=${BSPF_HPC_SDK:-/home/moulin/Workspace/nvidia/hpc_sdk/Linux_x86_64/26.5}
export PATH="$sdk/cuda/13.2/bin:$PATH"
export CPATH="$sdk/math_libs/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$sdk/math_libs/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
cmake -S "$(dirname "$0")" -B /tmp/bspf-poisson3d-cuda-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDAToolkit_ROOT="$sdk/cuda/13.2" \
  -DCMAKE_LIBRARY_PATH="$sdk/math_libs/lib64" -DCMAKE_INCLUDE_PATH="$sdk/math_libs/include"
cmake --build /tmp/bspf-poisson3d-cuda-build -j4
