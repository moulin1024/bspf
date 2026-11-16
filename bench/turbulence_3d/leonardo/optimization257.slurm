#!/bin/bash
#SBATCH --job-name=bspf-opt257
#SBATCH --account=euhpc_e07_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=toolchain-opt257-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source bench/turbulence_3d/leonardo/env.sh
module load python/3.11.7
export CUDA_PATH="$CUDA_HOME"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
py="$BSPF_TOOLCHAIN_ROOT/python/bin/python"
build="$BSPF_TOOLCHAIN_ROOT/optimization257/${BSPF_OPT_VARIANT:-optimized}"
baseline="$BSPF_TOOLCHAIN_ROOT/optimization257/baseline/bspf_poisson3d"
out="$PWD/bench/turbulence_3d/evidence/optimization257_${SLURM_JOB_ID}"
case_dir="$PWD/bench/turbulence_3d/evidence/mass_sweep_leonardo_64616443/bspf/n257_q386"
modes="$PWD/bench/turbulence_3d/evidence/nonperiodic_curved_gaussian/modes.npz"
cmake -S bench/turbulence_3d/bspf_poisson -B "$build" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
 -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "$build" -j8
mkdir -p "$out"
nvidia-smi > "$out/nvidia-smi.txt"
module list > "$out/modules.txt" 2>&1
cp src/cuda/poisson3d.cu "$out/candidate.cu"
export BSPF_POISSON3D_EXE="$build/bspf_poisson3d"
"$py" -m pytest -q tests/python/test_cuda_poisson3d.py > "$out/regression.txt" 2>&1
"$py" bench/turbulence_3d/bspf_poisson/campaign_257.py \
 --baseline "$baseline" --candidate "$build/bspf_poisson3d" \
 --case "$case_dir" --modes "$modes" --output "$out/timing"
if command -v nsys >/dev/null 2>&1; then
 export LD_LIBRARY_PATH="$BSPF_NATIVE_LIBRARY_PATH"
 for variant in baseline candidate; do
  executable="$baseline";if [[ "$variant" == candidate ]];then executable="$build/bspf_poisson3d";fi
  nsys profile --sample=none --trace=cuda,cublas --force-overwrite=true \
   --output="$out/profile_$variant" "$executable" "$case_dir/input.bin" "$out/profile_solution_$variant"
  nsys stats --report cuda_gpu_kern_sum --format csv "$out/profile_$variant.nsys-rep" > "$out/profile_${variant}_kernels.csv"
 done
fi
