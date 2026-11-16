#!/bin/bash
#SBATCH --job-name=galerkin-geosweep
#SBATCH --account=euhpc_e07_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=toolchain-geosweep-cuda13-%j.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source bench/turbulence_3d/leonardo/env_cuda13.sh
export BSPF_POISSON_FP64_EMULATION=performant
py="$BSPF_TOOLCHAIN_ROOT/python/bin/python"
modes="$PWD/bench/turbulence_3d/evidence/nonperiodic_curved_gaussian/modes.npz"
out="$PWD/bench/turbulence_3d/evidence/geosweep_cuda13_${SLURM_JOB_ID}"
mkdir -p "$out/source"
module list > "$out/modules.txt" 2>&1
nvidia-smi > "$out/nvidia-smi.txt"
ldd "$BSPF_POISSON_BINARY" > "$out/linked_libraries.txt"
sha256sum "$BSPF_POISSON_BINARY" "$modes" > "$out/input_hashes.txt"
cp bench/turbulence_3d/bspf_poisson/{run.py,nodal_forcing.py,nodal_mass.py} "$out/source/"
cp src/cuda/poisson3d.cu "$out/source/"
# Round geometric targets to odd sizes; advance ties to keep exactly 41 points.
"$py" - "$out" <<'PYGRID'
import json, math, sys
from pathlib import Path
nodes=[]
for i in range(41):
    target=17*(257/17)**(i/40)
    n=2*int(math.floor((target-1)/2+0.5))+1
    nodes.append(max(n,nodes[-1]+2 if nodes else 17))
assert len(set(nodes))==41 and nodes[0]==17 and nodes[-1]==257
p=Path(sys.argv[1])
(p/'nodes.txt').write_text(' '.join(map(str,nodes))+'\n')
(p/'sweep.json').write_text(json.dumps(dict(nodes=nodes,spacing='geometric targets rounded to distinct odd counts',solvers=['bspf','chebyshev']),indent=2)+'\n')
PYGRID
read -r -a nodes < "$out/nodes.txt"
cp bench/turbulence_3d/chebyshev_poisson/axis.py "$out/source/chebyshev_axis.py"
cp bench/turbulence_3d/leonardo/geosweep_cuda13.slurm "$out/source/"
"$py" bench/turbulence_3d/bspf_poisson/run.py --modes "$modes" \
 --n "${nodes[@]}" --forcing-mode nodal-bspf \
 --rhs-assembly nodal-mass --output "$out/bspf" > "$out/bspf.log" 2>&1
"$py" bench/turbulence_3d/chebyshev_poisson/run.py --modes "$modes" \
 --n "${nodes[@]}" --output "$out/chebyshev" > "$out/chebyshev.log" 2>&1
