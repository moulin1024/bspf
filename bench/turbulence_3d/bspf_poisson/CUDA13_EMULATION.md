# CUDA 13.0.2 FP64 emulation on Leonardo

The `profile/candidate` module `cuda/13.0` installs CUDA 13.0 Update 2 and
cuBLAS 13.1.0.3. Build this solver with GCC 13.4 in a fresh shell:

```bash
source bench/turbulence_3d/leonardo/env_cuda13.sh
cmake -S bench/turbulence_3d/bspf_poisson \
  -B .toolchains/leonardo/build/bspf-poisson-cuda13 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build .toolchains/leonardo/build/bspf-poisson-cuda13 -j8
```

The environment exports `BSPF_POISSON_BINARY` to this separate executable
and loads CUDA 13's driver compatibility library. The CUDA 12.6 production
executable remains preserved. Existing CuPy uses its CUDA 12 runtime and the
CUDA 12.6 toolkit for independent error audits; the native solver links CUDA
13/cuBLAS 13 by distinct SONAMEs.

`BSPF_POISSON_FP64_EMULATION` accepts:

- `native`: original pedantic FP64 arithmetic.
- `performant`: enable emulation when cuBLAS predicts an advantage; default
  for builds with cuBLAS >=13.1.
- `eager`: prefer emulation when possible, for diagnostic comparisons.

Both emulation strategies explicitly use dynamic mantissa control, zero
mantissa-bit offset and the library's standard 79-bit maximum. A larger
required precision causes native fallback rather than loss of FP64 accuracy.
The solver supplies a reusable 512 MiB workspace to avoid repeated internal
allocation across synchronized CG iterations. TF32 and reduced-precision CG
iterations are not enabled. CUDA 12 builds remain native by default.

Results record the loaded cuBLAS version and requested strategy. A requested
strategy is not evidence that every GEMM used emulation: check the separate
Nsight kernel reports for actual dispatch.

Submit `sbatch bench/turbulence_3d/leonardo/cuda13_emulation.slurm`. The job
checks native and automatic-emulation regressions, compares CUDA 12.6 with
CUDA 13 native, then compares native with both emulation strategies on the
fixed n=257,q=386 MMS. Each comparison alternates three pairs of executions,
collects nine warm samples per configuration, compares operator/preconditioner
probes and solutions, and independently audits physical errors at q=394/402.
Profiling runs separately from timing measurements.

Validation job 64625068 passed 60 native and 60 automatic-emulation tests.
The complete CUDA-version comparison measured 0.630252 s (CUDA 12.6) versus
0.629672 s (CUDA 13 native). The automatic-emulation comparison measured
0.629770 s (native) versus 0.629464 s (automatic). Both comparisons produced
bitwise-identical operator, preconditioner and solution probes; physical L2
error remains 3.05428e-12. These differences do not establish a speedup.

The user stopped the remaining comparison/profiling work to run the seven-size
CUDA 13 sweep instead. No conclusion about actual emulation kernel dispatch
is claimed without the unfinished profiling evidence.

Sweep job **64625526** completed successfully in 2m21s with the `performant`
strategy. All seven cases passed the native residual, boundary and independent
physical-error integration checks. Binary SHA256:
`2cca3c32a24e1a58162a45f94b7753630cd9c3b7eec9541b1a5bc5cd18708351`.

| n | q | Median warm solve (s) | Relative physical L2 error |
| ---: | ---: | ---: | ---: |
| 17 | 96 | 0.024805 | 3.03391 |
| 33 | 96 | 0.025810 | 0.255099 |
| 65 | 98 | 0.026897 | 3.94077e-12 |
| 97 | 146 | 0.044133 | 1.90348e-12 |
| 129 | 194 | 0.081038 | 2.53113e-12 |
| 193 | 290 | 0.233461 | 3.03903e-12 |
| 257 | 386 | 0.630135 | 3.05428e-12 |

Per-case results are in `../evidence/cuda13_sweep_64625526/bspf/`.
The comparison CSV and accuracy-versus-DOFs/time plots in
`../evidence/comparison_leonardo_to257/` now use these seven BSPF points.
Reference-solver rows are unchanged. The coarsest BSPF errors reflect the
underresolved full nodal forcing; they are not emulation errors.
