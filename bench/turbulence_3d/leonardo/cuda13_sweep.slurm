#!/bin/bash
#SBATCH --job-name=bspf-cuda13-sweep
#SBATCH --account=euhpc_e07_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=toolchain-cuda13-sweep-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source bench/turbulence_3d/leonardo/env_cuda13.sh
export BSPF_POISSON_FP64_EMULATION=performant
py="$BSPF_TOOLCHAIN_ROOT/python/bin/python"
modes="$PWD/bench/turbulence_3d/evidence/nonperiodic_curved_gaussian/modes.npz"
out="$PWD/bench/turbulence_3d/evidence/cuda13_sweep_${SLURM_JOB_ID}"
mkdir -p "$out/source"
module list > "$out/modules.txt" 2>&1
nvidia-smi > "$out/nvidia-smi.txt"
ldd "$BSPF_POISSON_BINARY" > "$out/linked_libraries.txt"
sha256sum "$BSPF_POISSON_BINARY" "$modes" > "$out/input_hashes.txt"
cp bench/turbulence_3d/bspf_poisson/{run.py,nodal_forcing.py,nodal_mass.py} "$out/source/"
cp src/cuda/poisson3d.cu "$out/source/"
"$py" bench/turbulence_3d/bspf_poisson/run.py --modes "$modes" \
 --n 17 33 65 97 129 193 257 --forcing-mode nodal-bspf \
 --rhs-assembly nodal-mass --output "$out/bspf"
