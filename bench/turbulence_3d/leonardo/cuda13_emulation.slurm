#!/bin/bash
#SBATCH --job-name=bspf-cuda13-emulation
#SBATCH --account=euhpc_e07_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=toolchain-cuda13-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source bench/turbulence_3d/leonardo/env_cuda13.sh
py="$BSPF_TOOLCHAIN_ROOT/python/bin/python"
build="$BSPF_TOOLCHAIN_ROOT/build/bspf-poisson-cuda13"
out="$PWD/bench/turbulence_3d/evidence/cuda13_emulation_${SLURM_JOB_ID}"
case_dir="$PWD/bench/turbulence_3d/evidence/mass_sweep_leonardo_64616443/bspf/n257_q386"
modes="$PWD/bench/turbulence_3d/evidence/nonperiodic_curved_gaussian/modes.npz"
cmake -S bench/turbulence_3d/bspf_poisson -B "$build" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
 -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "$build" -j8
mkdir -p "$out"
module list > "$out/modules.txt" 2>&1
nvidia-smi > "$out/nvidia-smi.txt"
ldd "$BSPF_POISSON_BINARY" > "$out/linked_libraries.txt"
cp src/cuda/poisson3d.cu "$out/candidate.cu"
"$py" -c 'import cupy as cp;print("audit runtime",cp.cuda.runtime.runtimeGetVersion(),"driver",cp.cuda.runtime.driverGetVersion());print(cp.arange(10).sum())'
export BSPF_POISSON3D_EXE="$BSPF_POISSON_BINARY"
for strategy in native performant; do
 export BSPF_POISSON_FP64_EMULATION="$strategy"
 "$py" -m pytest -q tests/python/test_cuda_poisson3d.py > "$out/regression_$strategy.txt" 2>&1
done
"$py" bench/turbulence_3d/bspf_poisson/campaign_257.py \
 --baseline "$BSPF_TOOLCHAIN_ROOT/optimization257/final/bspf_poisson3d" \
 --candidate "$BSPF_POISSON_BINARY" --baseline-emulation native --candidate-emulation native \
 --case "$case_dir" --modes "$modes" --output "$out/cuda12_vs_cuda13" \
 > "$out/cuda12_vs_cuda13.log" 2>&1 || echo "CUDA version comparison failed; inspect log"
for strategy in performant eager; do
 "$py" bench/turbulence_3d/bspf_poisson/campaign_257.py \
  --baseline "$BSPF_POISSON_BINARY" --candidate "$BSPF_POISSON_BINARY" \
  --baseline-emulation native --candidate-emulation "$strategy" \
  --case "$case_dir" --modes "$modes" --output "$out/$strategy" \
  > "$out/$strategy.log" 2>&1 || echo "Emulation $strategy comparison failed; inspect log"
done
for strategy in native performant eager; do
 export BSPF_POISSON_FP64_EMULATION="$strategy"
 nsys profile --sample=none --trace=cuda,cublas --force-overwrite=true \
  --output="$out/profile_$strategy" "$BSPF_POISSON_BINARY" "$case_dir/input.bin" "$out/profile_solution_$strategy"
 nsys stats --report cuda_gpu_kern_sum --format csv "$out/profile_$strategy.nsys-rep" > "$out/profile_${strategy}_kernels.csv"
done
