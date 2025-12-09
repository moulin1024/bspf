#!/bin/bash
# Setup CUDA environment for NVHPC SDK
# Add this to your ~/.bashrc or ~/.zshrc, or source it before running Python

export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo "CUDA environment configured:"
echo "  CUDA_PATH=$CUDA_PATH"
echo "  CUDA_HOME=$CUDA_HOME"


