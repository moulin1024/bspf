#!/bin/bash
#SBATCH --job-name=chebyshev-cuda13
#SBATCH --account=euhpc_e07_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=toolchain-chebyshev-cuda13-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source bench/turbulence_3d/leonardo/env_cuda13.sh
export BSPF_POISSON_FP64_EMULATION=performant
py="$BSPF_TOOLCHAIN_ROOT/python/bin/python"
modes="$PWD/bench/turbulence_3d/evidence/nonperiodic_curved_gaussian/modes.npz"
out="$PWD/bench/turbulence_3d/evidence/chebyshev_cuda13_${SLURM_JOB_ID}"
mkdir -p "$out/source"
module list > "$out/modules.txt" 2>&1
nvidia-smi > "$out/nvidia-smi.txt"
ldd "$BSPF_POISSON_BINARY" > "$out/linked_libraries.txt"
sha256sum "$BSPF_POISSON_BINARY" "$modes" > "$out/input_hashes.txt"
cp bench/turbulence_3d/chebyshev_poisson/{run.py,axis.py} "$out/source/"
cp bench/turbulence_3d/bspf_poisson/run.py "$out/source/shared_run.py"
cp src/cuda/poisson3d.cu "$out/source/"
"$py" -m pytest -q tests/python/test_chebyshev_poisson.py > "$out/axis_tests.txt" 2>&1
"$py" bench/turbulence_3d/chebyshev_poisson/run.py --modes "$modes" \
 --n 17 --polynomial --check --output "$out/validation"
"$py" bench/turbulence_3d/chebyshev_poisson/run.py --modes "$modes" \
 --n 17 33 65 97 129 193 257 --output "$out/chebyshev"
"$py" bench/turbulence_3d/bspf_poisson/campaign_257.py --basis chebyshev \
 --baseline "$BSPF_TOOLCHAIN_ROOT/optimization257/baseline/bspf_poisson3d" \
 --candidate "$BSPF_POISSON_BINARY" --baseline-emulation native --candidate-emulation performant \
 --case "$PWD/bench/turbulence_3d/evidence/large_retry_leonardo_64611321/chebyshev/n257_q386" \
 --modes "$modes" --output "$out/comparison257" > "$out/comparison257.log" 2>&1
