# MAST-U JOREK versus BSPF campaign

The campaign includes smooth and tilted Grad–Shafranov manufactured solutions, a compiled JOREK bicubic element baseline, and BSPF spatial and degree refinement on the MAST-U geometry.

[Solution contours](../../evidence/jorek_comparison/hard_mms_solutions.png) · [Domain-wide refinement](../../evidence/jorek_comparison/domain_refinement.png) · [Degree refinement](../../evidence/jorek_comparison/p_refinement.png) · [Measurements](../../evidence/jorek_comparison/)

### Prepare the reference inputs

Run the following from the release repository root. The Python drivers use NumPy, SciPy, h5py and Matplotlib. The native-element comparison additionally needs gfortran and two source files from the pinned JOREK checkout. An existing checkout can be selected with the environment variable `JOREK_SOURCE`.

```sh
mkdir -p third_party evidence/jorek_comparison/reference
git clone https://github.com/iterorganization/JOREK.git third_party/JOREK
git -C third_party/JOREK checkout --detach eaf299a66774dc6af66a86c9fad6b369cc43a459
archive=grid_with_patches_mastu_303_3f86adfbe97bce34ea359cddd9a83226.tgz
curl -fL "https://jorek.eu/dav_nrt/$archive" -o "$archive"
echo "df9a20ce992d8526cb4fbb3e266f604264afba1b0ffe87efc366a9cc79ce66ea  $archive" | shasum -a 256 -c -
tar -xzf "$archive" -C evidence/jorek_comparison/reference end.h5 eqdsk.dat
```

The reference is the final axisymmetric, bicubic model-303 solution from
`grid_with_patches_mastu_303`. The primary comparison holds the converged current
fixed. JOREK stores reduced current as `j = Δ*ψ`. On JOREK's physical domain,
BSPF solves the equivalent weak equation with stiffness weight 1/R and load
−j/R, imposing the JOREK flux trace on exterior edges. The native interior
flux is used for scoring. The two methods share geometry, current, and
boundary data. BSPF uses an independently assembled tensor spline/Fourier
Galerkin discretization and a sparse direct solve.

## Running

From the workspace root, use Python 3.11+ with NumPy, SciPy and h5py.
Run the analytic restart-format check first.

```sh
python3 bench/jorek_comparison/selfcheck.py
```

With the original reference archive extracted, run all three BSPF resolutions.

```sh
python3 bench/jorek_comparison/campaign.py path/to/end.h5 comparison/replay.json
```

To compute the original equilibrium locally, supply a built one-harmonic
model-303 executable and the regression case's eqdsk.dat. This runner creates
a fresh directory, copies the unchanged JOREK namelist, runs one MPI rank,
checks the convergence log, and consumes its restart directly.

```sh
python3 bench/jorek_comparison/native.py path/to/jorek_model303_1 path/to/eqdsk.dat comparison/native-run
```

The JOREK regression archive selected by this checkout is
[grid_with_patches_mastu_303](https://jorek.eu/dav_nrt/grid_with_patches_mastu_303_3f86adfbe97bce34ea359cddd9a83226.tgz).
Use the end.h5 from that case for a reference-only run. Source hashes identify
the particular snapshot. The HDF5 reader assumes variable indices 1 and 3
are ψ and j as in model 303.

## Measurements and scope

Each native Bézier element is one BSPF patch. Resolutions are 5, 7 and 9
nodal samples per patch axis. Element geometry and field derivatives are
evaluated from the native corner values, tangents, twist, and size factors.
Shared nodes are merged in physical space at 11 decimal places. The campaign
assumes a conforming mesh without hanging interfaces. It uses the same
18 × 18 Gauss rule per element to compare ψ and the physical poloidal field,
using JOREK's sign convention for both solvers. Relative L2 norms use dR dZ.
The flux gauge is the original JOREK gauge. Record the flux error together
with the gauge-independent field error, linear residual, setup time,
factorization-and-solve time, and free unknown count for each resolution.

The native wall-clock measurement includes startup, equilibrium iteration,
grid generation, and restart output. BSPF timings cover a fixed-current
linear reconstruction. They have different scopes and cannot establish an
end-to-end speedup. An independent nonlinear equilibrium comparison requires
matching the profile laws, their flux normalization, initial guess, X-point
selection, and wall trace. A subsequent benchmark should score magnetic-axis,
X-point and strike-point displacement on a common wall at matched solver
tolerance, including repeat timings on the same host.

The analytic check generates a temporary four-element sheared mesh with
ψ = R³ + Z² and j = 3R + 2. It checks signs, current units in the reduced
equation, shared traces and both physical field derivatives. It is separate
from any MAST-U measurement.

### Reproduce the field discrepancy controls

From the repository root, run

```sh
python bench/jorek_comparison/native_space_check.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/native_space_check.json --c0-output evidence/jorek_comparison/c0_bicubic.json
python bench/jorek_comparison/diagnose.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/diagnosis.json
```

The first command solves the fixed-current problem with the restart's shared
Hermite degrees of freedom and tagged wall constraints, then repeats it with a
value-continuous bicubic basis. The second compares BSPF with the restart
field on each element and saves the spatial distribution in diagnosis.png.

### Manufactured-solution comparison

Run the fixed-current accuracy benchmark with gfortran and the campaign's Python dependencies.

```sh
python bench/jorek_comparison/mms.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/mms.json
```

The manufactured flux is ψ = (1 + R²) cos(kZ), with reduced current Δ*ψ = −k²ψ and k = 3 per metre. Both methods evaluate the specified current as a function of position at 16 × 16 quadrature points per element. Each imposes the same physical Dirichlet data in its own trace space. The JOREK harness sets tagged wall derivative degrees of freedom from the analytic gradient. Both are scored at 24 × 24 quadrature against analytic flux and poloidal field using the physical area measure.

The native harness compiles the checkout's unchanged `models/element_matrix_GS.f90` and `elements/mod_basisfunctions.f90`. The small Fortran context supplies the nodal data, quadrature and manufactured profile callbacks. Python assembles the global matrix using the restart indices. Both sides use SciPy's sparse direct solve. The results measure spatial discretization accuracy for the linear GS operator. Production MPI solves, nonlinear iterations and the C++ BSPF direct backend are outside this measurement.

The MAST-U rows hold the physical mesh fixed. The rectangular rows use the same physical domain for every run and double the cells per axis at each refinement. Element counts differ between methods to approximately match the number of free unknowns.

### Physical-coordinate BSPF optimization

`global_bspf.PhysicalBSPF` uses one tensor BSPF expansion in physical R and Z,
restricted to the mesh. The curved MAST-U elements supply integration points and
exterior traces. A weighted least-squares solve enforces Δ*ψ = j in the interior
and ψ = g on the wall. The strong form assumes a smooth flux and source.
`solve(current, wall_flux)` accepts functions of (R,Z).
`evaluate(R,Z)` returns flux and its physical gradient.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/jorek_comparison/optimize_mms.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/optimized_mms.json
```

The benchmark uses degrees 5 in R and 13 in Z, one Fourier pair in each
direction and the existing default period of 1.5 in normalized coordinates.
It repeats the comparison at two source wavenumbers with the same basis.
Both methods run with 4 × 4 and 16 × 16 assembly. BSPF also runs with 2 × 2.
All runs are scored by the same 24 × 24 rule on all 2,715 elements.
Timings include geometry preparation, assembly, factorization and solve.
They exclude reference scoring, compilation and common restart loading. The
native harness additionally reads connectivity and wall tags during setup.

### Tilted near-wall MMS

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/jorek_comparison/challenge.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/tilted_mms.json
```

The flux adds a rotated Gaussian, 0.6 exp(−(u²+v²)/2), to the k=3
background. With x=R−1.30 and y=Z−0.72 (metres), u=(0.8x+0.6y)/0.08 and
v=(−0.6x+0.8y)/0.24. The local field norm integrates over the part of
u²+v² ≤ 4 inside the computational domain and uses the exact field norm
on that same region as its denominator. The operator is Δ*ψ=ψ_RR+ψ_ZZ−ψ_R/R.
The fixed-current harness carries the radial coordinate through the native
linear element's profile argument so its manufactured FFprime callback can
evaluate the Gaussian source at physical quadrature points. The JOREK
element routine is unchanged. The 128-coefficient global BSPF settings,
native wall constraints, and 24 × 24 field scoring follow the smooth test.
The local seven-node BSPF control uses one tensor patch per JOREK element.

### Residual-selected refinement of the global basis

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/jorek_comparison/refine_challenge.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/refined_tilted_mms.json
```

The pilot is the 128-coefficient physical-coordinate BSPF solve. It accumulates
the area-weighted squared strong PDE residual in square bins of side one eighth
of the shorter bounding-box dimension. Bins capturing 90% of that residual
receive compactly supported tensor quintic splines, with a one-bin halo.
The knot spacing is one twenty-fourth of the shorter dimension. A single global
least-squares system refits the original and added coefficients together.
Near-empty supports are discarded before diagonal scaling, and a ridge of
1e−12 stabilizes the scaled normal equations. Timings include the pilot,
selection, enriched assembly and final solve. The MMS ellipse only enters
the local error measurement. The spacing and marking fraction were selected
during development on this tilted benchmark.

### Flux contours and spatial convergence on the hard MMS

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/jorek_comparison/hard_convergence.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison
```

This produces the flux contour and error-versus-unknowns figures, measured
errors in JSON, and the two numerical flux fields plus the analytic field on
one common physical triangulation in `hard_mms_fields.npz`. The triangulation
follows the finest JOREK elements. Error norms are integrated independently
with 24 × 24 Gauss points per integration element; assembly uses 8 × 8.
The global BSPF sweep holds the residual-selected bins fixed and reduces the
knot spacing. Its first point is the original global basis without enrichment.

The native-element sweep uses the original MAST-U grid and 2 × 2 and 3 × 3
subdivisions of every original element. The Python mesh generator retains
the original vertex indices and derivative couplings at inherited corners.
It introduces shared Hermite tangent frames at new edge nodes and preserves
the original flux identification at the collapsed axis. These are harness-
generated grids consumed by the compiled JOREK bicubic element. Interior
parameterizations can change at original interfaces. The driver verifies the
physical trace of every original edge, including the wall, by comparing four
points on every cubic subinterval, and records the domain area at each level.
The flux contours use the finest tested space for each method.

### Large-range refinement with fixed physical coverage

Run `large_refinement.py RESTART OUTPUT --fixed-footprint --order Q --divisors D ... --score S`.
For each divisor D, the spacing is the shorter physical bounding-box dimension
divided by D. All levels keep complete quintic spline supports inside the same
union of padded marked-bin boxes. The physical padding is the support reach
of the two-bin, divisor-32 experiment. Thus the aligned doubling sequences
keep the intended physical footprint. Diagonal screening still discards
weakly sampled supports. The program records pilot and per-level refinement
times separately; scoring is excluded. At large sizes it eliminates the local
spline block before solving the small global Schur complement.

`plot_large_refinement.py evidence/jorek_comparison` reads the
`large_refinement_q*.json` files and plots each quadrature sweep beside the
three native-element measurements. Timing measurements are indicative single
runs on this workstation.

The recorded wide sweep used the following assembly orders, spacing divisors, and
independent scoring orders.

| Assembly | Divisors | Scoring |
| --- | --- | --- |
| 8 | 16, 24, 32, 48, 64, 96, 128 | 24 |
| 16 | 32, 48, 64, 96, 128 | 24 |
| 32 | 64, 96, 128, 192, 256 | 36 |
| 48 | 96, 128, 256 | 36 |

### Knot-resolved assembly

The fixed-region refinement uses composite Gauss quadrature and fixed physical
padding. In each intersecting curved element, Bézier derivative bounds select
the number of integration panels separately along the two parametric axes.
Each panel spans at most one knot spacing per physical coordinate in each
parametric direction. Six Gauss points per direction are used by default.
Wall integration is also subdivided. The global, local and coupling blocks
are integrated consistently and the solve fits a correction to the pilot
field. The scaled diagonal regularization decreases with the knot spacing.
The above historical fixed-quadrature runs can be reproduced by passing
`--points-per-span 0` to `large_refinement.py`.

```sh
python bench/jorek_comparison/check_refined_quadrature.py
python bench/jorek_comparison/large_refinement.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/fixed_refinement.json --fixed-footprint --divisors 16 24 32 48 64 96 128 192 256 --score 36
python bench/jorek_comparison/plot_large_refinement.py evidence/jorek_comparison --fixed evidence/jorek_comparison/fixed_refinement.json
```

### Diagnose the fixed-footprint plateau

`global_degree_study.py` freezes the pilot's marked bins and wall penalty and
varies the vertical degree and local knot spacing. Each `--case P D` sets the
vertical degree P and uses spacing equal to the shorter box dimension divided
by D. Use `--extra-bins 1` to widen the padding on every side by one coarse bin.
Flux and field errors are scored independently of the residual indicator.

The alternative `--remark` option in both `global_degree_study.py` and
`large_refinement.py` computes the strong PDE residual of the complete fitted
field at the pilot quadrature points, marks bins accounting for 90% of that
squared residual, takes the union with the initial bins, and resolves at the
same spacing. Each resolution starts from the same pilot bins. For example:

```sh
python bench/jorek_comparison/large_refinement.py evidence/jorek_comparison/reference/end.h5 /tmp/remark.json --divisors 64 96 128 --score 36 --remark
```

### Domain-wide correction for the fixed-region error floor

The standard `refine(h)` and the large refinement driver now fit a quintic
spline correction across the physical domain. The original global BSPF pilot
is retained as the background. On halving the knot interval the cardinal
two-scale relation transfers the previous correction exactly, including its
first and second derivatives. The solver then fits the residual in the fine
space. A non-dyadic change starts a fresh correction to the pilot. Geometry,
Dirichlet data, and knot-resolved quadrature are unchanged. Explicit
`support_padding` selects local refinement in the solver. Use
`--fixed-footprint` for the earlier fixed-region CLI experiments and `--remark`
for the two-pass local method.

```sh
python bench/jorek_comparison/large_refinement.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/domain_refinement.json --domain --divisors 16 32 64 128 256 --score 36
python bench/jorek_comparison/plot_domain_refinement.py evidence/jorek_comparison
```

The count reported for the domain-wide curve includes the fixed pilot, active
correction unknowns, and inherited nonzero coefficients on screened supports.
Refinement time excludes pilot construction and independent error scoring.

### Fixed-knot degree refinement

Run `p_refinement.py` to vary the degree of the domain-wide cardinal spline
correction independently of the global pilot. The default sweep holds physical
knot spacing at the short bounding-box dimension divided by 32 and uses degrees
3 through 9. Interior knots stay simple, so continuity rises with degree. Each
degree solves afresh for the correction from the same pilot. The plot shows
error against degree, total retained coefficients, and assembly-plus-solve time.
Timings exclude the shared pilot and independent error scoring.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/jorek_comparison/p_refinement.py evidence/jorek_comparison/reference/end.h5 evidence/jorek_comparison/p_refinement.json
python bench/jorek_comparison/plot_p_refinement.py evidence/jorek_comparison/p_refinement.json evidence/jorek_comparison/p_refinement.png
```
