# Nonperiodic 3D incompressible MMS on a curved domain

Current Gaussian MMS campaign: [dependencies, builds, validation and reproduction](REPRODUCING.md). Historical measurements below retain their original configurations.

This dataset defines a steady synthetic velocity and pressure directly in
physical Cartesian coordinates. Continuous random wavevectors have random
radial positions and directions, with no reciprocal lattice or periodic parent
box. No boundary faces are identified. Velocity polarizations are perpendicular
to each wavevector, so the analytic velocity is divergence-free everywhere.
The field is manufactured, not a turbulent-flow simulation.

The physical domain is the image of `[0,1]^3` under the exact Q2 map

```
phi(s) = 4 s (1-s)
X_i(s) = s_i + a_i (1-2 s_i) phi(s_j) phi(s_k), j,k != i
(a_x,a_y,a_z) = (0.08, -0.06, 0.07).
```

All six faces are curved, and opposing faces differ in shape. The map has a
positive Jacobian: the infinity norm of the derivative of its displacement is
bounded by `10 max|a_i| < 1`. Components and derivatives are physical Cartesian
components, not reference-space components. For a curved-domain solver, evaluate
the MMS at mapped physical coordinates; do not simply reinterpret reference
components as physical velocity.

For each continuous wavevector `k_m`, the velocity contains

```
u(x) = u_offset + sum_m [a_m cos(k_m.(x-center)) + b_m sin(k_m.(x-center))]
k_m.a_m = k_m.b_m = 0, center=(0.5,0.5,0.5).
```

Pressure has independent scalar cosine/sine coefficients and a constant gauge.
The radial shell coordinate is `|k|/pi`. The prescribed velocity modal energy
in shell `s` is proportional to `s^(-5/3)`, with random directions and radii
within each shell. Each mode contributes `(|a_m|²+|b_m|²)/4` to that modal energy.
This is a coefficient-space specification, **not an FFT spectrum of nonperiodic
samples**. Cross terms on the finite curved domain mean the sum of modal energies
need not equal its measured physical kinetic energy.

The defaults use seed 20261116, shells 4–32, 64 wavevectors per shell (1,856 total),
and a 97-by-97-by-97 sampling grid including boundaries. Physical volume-weighted
velocity RMS is normalized to 1; pressure RMS is 0.1. Both physical means are
zero, computed with curved-domain Gauss quadrature. The field remains analytically
defined between sample nodes, so future convergence studies can use the same
modes at any mesh or quadrature resolution.

## Generate with the GPU

Requires NumPy, SciPy, Matplotlib, h5py, and a CuPy build matching an installed
CUDA toolkit. There is no CPU generation fallback. In this workspace, CuPy
13.6.0 is the CUDA-12 build, so use the installed CUDA 12.6 toolkit even though
the NVIDIA driver also supports CUDA 13:

```sh
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/lib64:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python bench/turbulence_3d/generate.py \
  --out bench/turbulence_3d/evidence/new_nonperiodic_run --save-gradient
```

Run with GPU device access; the Codex sandbox may hide `/dev/nvidia*`, requiring
execution outside it. The output directory must be empty. `--n`, `--shell-max`,
`--modes-per-shell`, `--quadrature`, and `--viscosity` control resolution, field
complexity, normalization accuracy, and manufactured forcing. Grid sampling is
direct analytic evaluation, with no interpolation or FFT padding. `--n` does
not change the modes. See [CuPy installation guidance](https://docs.cupy.dev/en/stable/install.html)
for other environments. Exact coefficient files and software versions identify
the realization; seeds alone are not a promise of cross-version random-stream
identity.

## Data and equations

- `modes.npz`: compact authoritative wavevectors, transverse velocity coefficients,
  pressure coefficients, offsets, viscosity, seed, shells, and geometry parameters.
- `fields.h5`: mapped coordinates, Jacobian, velocity, pressure, vorticity,
  pressure gradient, vector/pressure Poisson forcing, Stokes forcing, and
  Navier–Stokes forcing. `--save-gradient` adds the full velocity gradient.
- `fields.h5/boundary/{x0,x1,y0,y1,z0,z1}`: six physical boundary traces with
  coordinates, unit outward normals, velocity, and pressure. Face labels refer
  to reference faces; the physical faces are curved.
- `report.json`: checks, modal spectrum, physical statistics, hardware/software,
  hashes, and timings. `preview.png` and `preview.pdf` visualize the field/domain.

Volume datasets have spatial axes `(s,t,r)` followed by physical component axes:
`velocity.shape=(n,n,n,3)`, `velocity_gradient.shape=(n,n,n,3,3)`, and
`velocity_gradient[...,i,j]=du_i/dx_j`. The reference nodes include 0 and 1.
The default viscosity is 0.01 and density is 1. All arithmetic is float64.

```
vector_poisson_forcing = -Delta u
pressure_poisson_forcing = -Delta p
stokes_forcing = -nu Delta u + grad p
navier_stokes_forcing = (u.grad)u + grad p - nu Delta u
div u = 0, du/dt = 0
```

The traces are exact manufactured Dirichlet data, with nonzero local normal
velocity permitted. Their total outward flux is zero. The stored pressure trace
is available for checking; imposing velocity on every boundary in an
incompressible solve generally uses a pressure gauge, not an additional pressure
Dirichlet condition everywhere.

For arbitrary physical points, import `PlaneWaveMMS` from `mms.py` and load the
mode file with `PlaneWaveMMS.load(path, xp=cupy)`, then call `evaluate(points)`.
The NumPy evaluator provides a small-problem validation/reference path. Loading
and evaluating the coefficients preserves nonperiodic boundary traces.

## Validation

The generator checks transverse polarizations; the prescribed modal shell energy;
independent higher-order physical RMS/mean quadrature; GPU sine/cosine evaluation
against independent CPU complex exponentials at off-grid points; fourth-order
finite-difference checks of analytic derivatives and Navier–Stokes forcing;
divergence in the volume; and integrated flux over all six curved boundaries.
Opposite-face trace mismatches are recorded to demonstrate nonperiodic data.

CPU analytic/geometry checks run with:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/python/test_nonperiodic_turbulence_mms.py
```

Saved files can be checked without a GPU:

```sh
OPENBLAS_NUM_THREADS=1 python bench/turbulence_3d/check_saved.py \
  bench/turbulence_3d/evidence/nonperiodic_curved_shell32
```

## Generated dataset

The completed RTX 3070 dataset is in
[`evidence/nonperiodic_curved_shell32`](evidence/nonperiodic_curved_shell32).
It has 912,673 volume samples, 1,856 continuous wavevectors, and full velocity
gradients. Its HDF5 file is approximately 245 MB. Generation and validation took
7.87 s excluding plotting; GPU volume sampling took 1.59 s. These are generation
measurements, not PDE solver timings.

- Physical velocity RMS: 1.0000000000000002; pressure RMS: 0.10000000000000005.
- Prescribed modal spectrum slope: -1.6666666666666672.
- Relative divergence over the volume samples: 6.04e-16.
- Net outward flux over all six faces: 5.76e-16 (relative 1.99e-16).
- Opposite-face velocity mismatch: 1.35, 1.42, and 1.39 relative RMS for x/y/z.
- Geometry Jacobian at audit points: 0.8092 to 1.1198; physical volume: 0.92.
- Independent CPU/GPU analytic evaluation discrepancy: at most 1.13e-15.
- Finite-difference derivative/forcing discrepancies: at most 2.58e-10.

[Preview](evidence/nonperiodic_curved_shell32/preview.png),
[validation report](evidence/nonperiodic_curved_shell32/report.json), and
[saved-file validation](evidence/nonperiodic_curved_shell32/storage_check.json)
record the result. The spectral coefficients and physical boundary traces are
used by the [3D MFEM/libCEED Poisson benchmark](mfem_poisson/README.md).
Its [accuracy–walltime plot](evidence/mfem_poisson_rtx3070/accuracy_walltime.png)
and [measurement report](evidence/mfem_poisson_rtx3070/report.json) record
the scalar GPU solve and independent physical-domain error audits.

The [GPU HYPRE/LORSolver comparison](evidence/mfem_lor_amg_rtx3070/preconditioner_comparison.png)
uses degree 8, starts at one element, and stops at 16³ elements. See the
[build instructions and measurements](mfem_poisson/README.md) for the MPI/CUDA
stack, required degree-8 GPU geometry specialization, setup costs, and memory limits.

## Mixed scalar Poisson MMS: Gaussian bumps and a sharp spherical tanh interface

`generate_mixed.py` augments the original scalar solution with eight reproducible
signed anisotropic Gaussians and one spherical tanh profile. The new dataset is
`evidence/nonperiodic_curved_mixed/modes.npz`; the original plane-wave dataset
and its velocity coefficients remain unchanged. The scalar Poisson unknown is
now **phi = original u_x + localized features**, not a modified velocity
component claimed to remain divergence-free. All original velocity, pressure
and geometry coefficient arrays are preserved exactly.

Default feature seed is 20261117. Gaussian centers are randomly placed in the
interior, standard deviations range from 0.045 to 0.10, and signed amplitudes
have magnitudes 0.5 to 0.9. The tanh sphere has a randomly placed interior center,
radius 0.22, transition-width parameter 0.015 and amplitude 0.8. Its planar
cross-sections are circles. Centers and all parameters are recorded in
`manifest.json` and the NPZ. Exact formulae in physical Cartesian coordinates:

```
g_j = A_j exp[-0.5 sum_i ((X_i-c_ji)/sigma_ji)^2]
-laplacian(g_j) = g_j sum_i [1/sigma_ji^2 - (X_i-c_ji)^2/sigma_ji^4]

r2 = |X-c_t|^2; s = 2 R delta; t = tanh((r2-R^2)/s)
phi_t = A_t/2 (1-t)
-laplacian(phi_t) = A_t (1-t^2) [3/s - 4 r2 t/s^2]
```

Using squared radius avoids a radial singularity at the center; the interface
is sharp but smooth everywhere. The total forcing is the sum of the original
analytic forcing and these contributions. Exact boundary values are sampled
from the composite scalar function on the same six curved faces.

Generate with the existing CuPy CUDA environment:

```sh
python bench/turbulence_3d/generate_mixed.py
```

This writes a 129³ CUDA-sampled `samples.h5`, a component slice preview in
PNG/PDF, slice arrays, metadata and the coefficient NPZ. Solvers evaluate the
analytic MMS directly; they do not interpolate the HDF5 samples. BSPF,
Chebyshev, MFEM, NekRS and FC drivers now all accept the new data through
`--modes bench/turbulence_3d/evidence/nonperiodic_curved_mixed/modes.npz`.
Use separate output directories so the original benchmark results remain intact.
The existing comparison plot still represents the original plane-wave MMS;
a new convergence campaign is required for this sharper mixed dataset.

Seven CPU tests (including the original MMS tests) pass. Independent fourth-order
finite differences validate the feature Laplacian, including points near the
interface and at the center. CUDA/NumPy complete-sampler differences are below
8e-16 relatively. The independent Torch-autodiff Laplacian check gives a relative
forcing discrepancy of 2.17e-16; its complete mixed sampler also agrees with
NumPy below 8e-16. Reproduce that check using
`/tmp/bspf-fc-torch/bin/python bench/turbulence_3d/fc_poisson/validate_mixed.py`.
The sharp profile requires fresh quadrature-convergence checks; the original
plane-wave quadrature study does not establish adequacy for this new field.

### Wave + Gaussian campaign (without tanh)

The active variant omits the tanh sphere while preserving every plane-wave and
Gaussian coefficient from the mixed dataset exactly. Its data are in
`evidence/nonperiodic_curved_gaussian/`; the eight signed Gaussians and seed
20261117 are unchanged. The scalar solution is `phi = original_u_x + sum(g_j)`.
The NPZ stores empty tanh arrays, so all five samplers evaluate zero tanh
contribution. Its SHA256 is
`c9119da1d12d3bbca43497bcddc5f5b92baf6611cdd047c49866e025812c9949`.

Generate and run using the existing CUDA environments:

```sh
python bench/turbulence_3d/generate_mixed.py --no-tanh \
  --out bench/turbulence_3d/evidence/nonperiodic_curved_gaussian
python bench/turbulence_3d/run_mixed_suite.py
python bench/turbulence_3d/fc_poisson/report.py \
  --suite-root bench/turbulence_3d/evidence/gaussian_mms_rtx3070 --max-n 129
python bench/turbulence_3d/plot_mixed_3d.py
```

The suite driver sets its subprocess CUDA library paths and runs jobs sequentially.
The generator and quadrature checker use the CuPy CUDA environment described above.
`--modes` and output-directory options allow separate campaigns without replacing
the original evidence. Physical resolutions are 17, 33, 65, 97, 129 and 145 nodes
per axis. FC retains 25 continuation nodes per side, excluded from its plotted
physical-grid count; their computational cost remains included in solve timings.

Independent error integration refines automatically when localized features need
it; those audits are outside the solve timing. An optional check,
`python bench/turbulence_3d/check_mixed_quadrature.py`, raises
integration order by 50%, computes the physical solution difference, and bounds
the change in relative L2 error using the triangle inequality. A full exact-MMS
audit follows if that bound is inconclusive. Acceptance requires error change
below the larger of 1% of the measured error and 1e-12 (the outer-solver accuracy
scale). The selected source files and checks are saved in
`quadrature_validation.json`. Plotting defaults to the completed main campaign;
`--validated-quadrature` explicitly selects the optional check results. The
comparison CSV records the actual source files and solve timings. MFEM independently checks 11 versus 13 quadrature points per
element direction at the largest mesh.
