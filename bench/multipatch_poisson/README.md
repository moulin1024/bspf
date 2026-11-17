# BSPF multipatch Poisson / TCV-X21 investigation

This is a small CPU correctness experiment using the `HybridAxis` basis from
the current 3D Poisson benchmark. It assembles two curved quadrilateral patches
with common interface unknowns, positive variable diffusivity, and manufactured
Dirichlet data. Patch stiffness contributions and loads are added at shared
unknowns. This enforces the interface flux balance weakly; independently solving
patches and averaging their derivatives would not do so.

Run with NumPy/SciPy and the repository's Python environment:

```
OPENBLAS_NUM_THREADS=1 python bench/multipatch_poisson/check.py
```

`results.json` records a refinement study and independent increased quadrature
checks. This is a dense validation assembly, not the production GPU solver,
not the ghost-overlap `MultiPatch2D` differentiation implementation, and not yet
a TCV mesh test. The separate `test_multipatch.py` / `test_galerkin.py` suite
could not collect in this checkout because `bspf_cpp._core` is not built.

The two-patch check passed on 2026-11-17:

| Nodes per patch | Global unknowns including boundary | Maximum nodal error (higher quadrature) |
|---|---:|---:|
| 11 x 11 | 231 | 6.08e-7 |
| 17 x 17 | 561 | 7.15e-12 |
| 25 x 25 | 1225 | 3.77e-14 |

All six assemblies were positive definite after boundary elimination, with
relative symmetry defects below 7e-17 and solve residuals below 9e-15.
Increasing quadrature changed the maximum-error estimates by less than
7e-14. Errors are measured at solution nodes, not a continuous supremum norm.

## GENE-X field equations inspected locally

Sources under `~/genex` (2026-11-17):

- `parallax/src/helmholtz/helmholtz_operator_m.f90`
- `src/operators/field_solve_operators/op_solve_qn_eq_s.f90`
- `src/operators/field_solve_operators/op_solve_amps_law_s.f90`
- `src/operators/op_mom_maxwells_eq_cpu_s.f90`
- `src/mesh/mesh_5d_s.f90`
- `local-paccx/README.rst`
- `run_genex/params_in.txt` and `run_genex/TCV-X21.nc`
- `runs/tcv-strong-profile-1to256-20261116/inputs/params_in.txt` (matching geometry settings)

The Parallax interior stencil approximates

    L u = -xi div_RZ(c grad_RZ u) + lambda u.

It uses nearest Cartesian R,Z neighbors with arithmetic averages of c on
faces. PAccX supplies FGMRES with a Parallax-specific multigrid preconditioner.
It does not define a different field equation. The experimental OpenMP-target
GENE-X branch instead uses paccx_ext/HYPRE; the OpenACC branch retains
Parallax/PAccX. These backends must not be conflated.

For TCV, the cylindrical Jacobian J=R. In the plasma interior, with
epsilon=rho_ref/L_ref, the inspected normalized equations are:

    -1/J div_RZ(c_QN grad_RZ phi) = b_QN,
    c_QN = epsilon^2 J/B^2 sum_s(m_s n_s),
    b_QN = sum_s(q_s n_s).

The density is the code's velocity-space moment, including its integration
weights. Nonlinear polarization updates c_QN; the linear-polarization path
freezes the first supplied moment-based coefficient. Its initialize routine
also allocates an initial epsilon^2 J/B^2 coefficient.

    -epsilon^2/J div_RZ(J grad_RZ A_parallel) = b_A.

Here b_A is the normalized parallel-current moment assembled by GENE-X,
including beta_ref and species thermal-speed factors. The inspected Ampere
solve has no interior reaction term. Target rows use xi=0, lambda=1. The
Parallax solver factory requests zero Dirichlet conditions for core, wall,
dome, and outer boundaries. Each toroidal plane gets its own 2D field solve.

For a symmetric Galerkin discretization of quasineutrality, multiply by J:
assemble integral(c_QN grad(v).grad(phi)) with load integral(J v b_QN).
For Ampere, the stiffness coefficient is epsilon^2 J and the load is J b_A.
The physical cylindrical J and each patch mapping determinant are distinct.
On a patch with map derivative F, the reference stiffness tensor is
det(F) c F^{-1} F^{-T}. Shared traces and assembled interface equations are
required, as are a global preconditioner and a coarse correction for scaling.

## Geometry and remaining validation

The NetCDF file contains a 500 by 600 equilibrium psi grid, R and Z arrays,
a 513-point divertor polygon, and exclusion/private-flux polygon data. It is
not a quadrilateral patch mesh. The inspected input sets flip_Z=true,
rhomin=0.74, a five-point exclusion polygon override, and RZ spacing 0.00136.
Those transformations and masks must be reproduced before fitting boundaries;
using only the raw divertor polygon would not reproduce the simulation domain.

The magnetic X-point is not a coordinate singularity of this Cartesian RZ
elliptic operator. A flux-aligned patch layout is not required. A geometrical
quadrilateral decomposition can cross the separatrix, provided patch maps
remain nonsingular and material coefficients are represented accurately.

Next validation: extract the exact active domain from these inputs, construct
conforming patches with positive Jacobians, run a manufactured-solution
refinement test on that geometry, then compare against the Parallax discrete
solution with identical coefficients and boundary data. No claim of TCV-X21
solver validation or GPU throughput follows from the two-patch check alone.
