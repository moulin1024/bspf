STOKES-GUIDED CHEBYSHEV ATLAS — GENERAL 2D PROTOTYPE

This module implements a reusable mapping procedure, without a built-in
physical test geometry or prescribed quad layout. It consumes an existing
conforming triangulation of the geometry and an already computed Stokes field.
It does not solve Stokes, infer a domain from the flow, or claim to find the
globally optimal block topology. No old obstacle benchmark is run or imported.

PUBLIC API

From the repository root:

    from experimental.stokes_chebyshev_atlas import (
        Geometry, StokesGuide, Settings, generate, from_arnoldi_solver,
    )

    geometry = Geometry(vertices, triangles, curves=boundary_curves)
    guide = StokesGuide(q=phi_prime, dq=phi_second, scale=q_scale)
    # Alternatively, consume the existing fitted AAA/lightning Stokes solver:
    guide = from_arnoldi_solver(stokes_solution, scale=q_scale)

    result = generate(geometry, guide, Settings(
        degree=8, solution_degree=24,
        max_condition=20, max_scale_variation=30,
        spectral_tolerance=1e-5,
    ))
    result.atlas.save("atlas.npz", result.report)
    atlas = result.require_accepted()

The variable names above are inputs from your application, not a supplied
physical example. vertices is a one-dimensional complex coordinate array;
triangles is an integer (T,3) connectivity array. Input triangles must form
a nonoverlapping planar mesh with conforming edges. Local degeneracy, duplicate
triangles, and inconsistent shared-edge incidence are checked. Full geometric
overlap/intersection detection of arbitrary input triangulations is not done.

curves[(vertex_a, vertex_b)](t) evaluates an analytic CAD boundary segment for
t in [0,1], including its endpoints. Curves must preserve the triangulation's
topology. Unspecified physical boundaries are straight. Holes and disconnected
components are supported by the same incidence construction. Curved geometry
may require a finer input triangulation to keep initial Coons charts regular.
Interior triangle edges and redundant straight boundary vertices do not prescribe
block interfaces. The generator first reconstructs the domain boundary, then
tries triangle pairing and conforming block agglomeration where needed.

The callbacks q and dq accept complex NumPy arrays and return complex arrays
(or constants). They are derivatives with respect to the physical complex
coordinate, not derivatives with respect to a chart parameter.

    Phi = p/mu - i*omega = 4*f'(z)
    q = Phi' = 4*f''(z); dq = Phi'' = 4*f'''(z).

This identity assumes 2D unforced constant-viscosity incompressible Stokes.
For conservative body forcing, remove its pressure potential first. General
body forcing, variable viscosity, and 3D require a different guide. The adapter
differentiates polynomial/rational Arnoldi recurrences analytically through
third order. It includes the existing single-hole solver's logarithmic modes;
additional holes can be supplied through explicit (center, coefficient) pairs
in from_arnoldi_solver(..., log_terms=[...]). No finite-difference derivative
of pressure, vorticity, or a rational fit is used.

ALGORITHM

0. Prefer aligned affine charts when the boundary is rectilinear. Recover a
   common orthogonal frame from boundary directions (rotation is supported),
   discard collinear sampling vertices, and collect true corner coordinate
   levels. Their intersections form candidate rectangles; an even/odd boundary
   test removes solid cells. No square-obstacle branch, block count, or interface
   locations are encoded. A rectangle gives one block, a single aligned
   rectangular hole gives eight, and other orthogonal geometries give other
   counts. If this candidate passes quality and budget checks, preserve its
   affine maps instead of distorting them to reduce the Stokes-guidance energy.

   This is an explicit geometric preference: zero shear and constant metric
   coefficients take priority over fewer skewed blocks. It is not convergence
   of the Stokes energy to a globally optimal topology. No Stokes solve is
   required for recognition; a zero guide can be supplied. With
   prefer_affine_rectilinear=False (CLI --no-affine-preference), use the older
   block-count-first search. A failed aligned candidate falls through to the
   general search. General curved geometry still uses Stokes-guided optimization.

1. Reconstruct oriented boundary cycles from the triangle incidence graph.
   If the domain has one boundary cycle with exactly four geometric corners,
   discard its interior triangulation edges and try a single Coons chart.
   Consecutive straight boundary segments become one affine side. Smooth CAD
   segments can be joined; their representation is checked against the exact
   input curves. No artificial corner is invented on a smooth closed boundary.
   A positive, resolved one-block map is retained immediately: no subdivision
   is needed, even if the Stokes guide has a zero inside the chart.

2. When the single chart is unavailable or fails, enumerate connected triangle
   groups and test their four-sided curved boundary maps. A binary exact-cover
   solve selects a low-block-count cover from admissible groups. This handles
   smooth hole boundaries that pairwise triangle matching cannot quadrangulate.
   Enumeration is bounded to 4000 groups of at most 16 triangles; the cover solve
   has a 10-second limit. Distinct physical arcs cannot share one endpoint-pair
   key, avoiding accidental identification of two sides of a closed wall.
   Also try adjacent-triangle minimum-cost perfect matching, with a geometric
   shape cost. Conforming triangle subdivision remains a fallback.
   Coarsen by merging adjacent chart pairs or vertex stars into four-sided
   regions. Every proposed merge must pass the actual map acceptance checks.
   Vertices required by outside charts are protected; no hanging interfaces
   may be introduced. A region with a hole cannot be merged into one chart.

3. Outside the affine-aligned preference, rank searched layouts by block count,
   subject to acceptance. When needed,
   try both two-way splits of a rejected single chart before four-way refinement.
   Multi-chart refinement is currently uniform four-way refinement to preserve
   conformity. max_refinements and max_blocks bound this search. Refinement is
   never run when it cannot beat an already accepted layout's block count.
   This is a finite heuristic search, not a global minimal-topology algorithm.
   Stokes influences the continuous maps and their feasibility, not the matching
   shape cost or region enumeration. Critical points are diagnostic only; their
   search is not exhaustive.

4. Jointly optimize tensor polynomial maps, using the same physical Stokes field
   across all charts. Shared Lobatto unknowns enforce full polynomial trace
   compatibility under reversal. The unknowns parameterize polynomials directly
   through invertible DCTs; there is no point-grid-to-spline fitting stage.
   Physical boundary values remain fixed. With X_xi=a, X_eta=b, d=det(DX), and
   normalized q=Phi'/q_scale, define:

       r = |q|^2; m = 1 + density_contrast*r/(1+r)
       D = (|a|^2+|b|^2)/(2*d)

       E = mean_integral[
             distortion_weight*(D-1)
             + area_weight/2 * log(m*d/target)^2
             + alignment_weight/2 * (A(a)+A(b))
           ] + spectral_map_regularization

       A(a) = 4 Re(q*a)^2 Im(q*a)^2 / ((1+|q|^2)^2 |a|^4)

   Alignment is four-way and vanishes smoothly at q=0. The positive metric
   M=m*I guides spacing without inverting Phi. The fixed global area target is
   computed from the initial atlas. q_scale and density_contrast are modeling
   choices, not unique consequences of Stokes flow or guarantees of wall layers.
   The modal map penalty has weight max(max(i,j)-1,0)^6, normalized by squared
   domain diameter and block count. Affine/bilinear modes are unpenalized.

5. A feasible L-BFGS/Armijo search must lower E AND pass the quality filter.
   It rejects inversion, unresolved operator tails, and excessive deterioration
   of each block's worst condition, scale variation, or minimum singular value.
   quality_slack=1.05 bounds deterioration relative to the initial candidate,
   not cumulatively from the previous iteration. A candidate that initially
   passes all acceptance checks must keep passing after every accepted step.
   An initially unresolved candidate may improve, but its tail may not worsen.
   A constraint-limited exit is reported as quality_limited, not convergence.

6. Validate on an oversampled Lobatto grid and an independent tensor Gauss grid.
   Report minimum Jacobian, maximum condition, scale variation, CAD boundary
   error, complete interface mismatch, and coefficient tails beyond the intended
   PDE degree. Tail checks cover X, mass det(DX), and the weak Laplace tensor
   det(DX)*inverse(DX)*inverse(DX)^T. Mass and stiffness tails are normalized
   separately because they have different physical units; the three stiffness
   components share one normalization to handle zero off-diagonal entries.
   operator_tail is the larger of the mass and stiffness tails. Thus a change
   of length units cannot hide unresolved stiffness coefficients.
   Map degree and intended PDE degree remain distinct. All optimization map
   derivatives and adjoints use FFT/DCTs, not collocation derivative matrices.
   Independent off-grid diagnostics use polynomial Clenshaw evaluation.

WHAT ACCEPTANCE MEANS

result.report['quality']['accepted'] means only that the specified sampled
geometry and coefficient-resolution criteria passed. It does not certify
global injectivity, locate every complex singularity, prove exponential PDE
convergence, or establish a stable timestep. Analytic polynomial maps alone
do not make inverse metric coefficients harmless: nearby complex zeros of the
Jacobian can still slow spectral convergence. The operator-tail diagnostic is
an empirical resolution check, not an analyticity proof.

If bounds fail, require_accepted() raises and the report names failing blocks
and criteria. Increase the intended solution degree for unresolved smooth
operators, improve the map objective, or refine/restructure the geometry
triangulation. Curved boundary underresolution may require increasing map
degree as well. Topology refinement is bounded and automatic; polynomial degree
adaptation and local nonuniform refinement are not implemented.

Optimization convergence and geometry acceptance are separate report fields.
An accepted finite-iteration candidate need not be stationary or optimal.
For affine aligned maps, optimization.status is affine_aligned_preserved:
iterations=0, history=[], and objective/gradient values are null because the
secondary Stokes objective was not optimized. New per-block diagnostics report
maximum_skew_cosine and relative_metric_variation. The former measures loss of
orthogonality; the latter measures variation of DX^T DX inside each chart.
Conversely, a converged optimizer can still produce a rejected geometry.

Interfaces are exactly C0 polynomial-compatible. The method does not impose
raw equality of chart Jacobians, or claim global C1 maps through extraordinary
vertices. Physical PDE interface coupling remains the solver's responsibility.
An oriented interface penalty now compares opposing unit transverse directions;
sharing positions alone did not prevent visibly kinked lines. A separate wall
penalty encourages normal incidence, and optional interface_spacing_weight
matches transverse scales. interface_weight and wall_orthogonality_weight
default to 1 and 0.1. Set them to zero to recover the previous objective.
These are soft constraints, not continuity guarantees. The smoothness report
gives angle RMS weighted by arclength and (1-s^2)^2, with endpoint angles
reported separately. Smoothness does not currently gate geometry acceptance.
The region cover requires retained smooth curved-wall vertices to have two
incident blocks; it can remove sampled vertices. This is a geometry constraint,
not a topology derived from Stokes. Extraordinary vertices can remain elsewhere.

smooth_atlas(atlas, guide, settings) resumes coordinate optimization on existing
connectivity with the same map degree. The optional operator_barrier_weight
(default zero) adds a log barrier on squared mass and stiffness spectral tails,
normalized separately and differentiated with FFT adjoints. Its threshold is
max(tolerance^2, 1.05*initial_tail^2); the existing hard acceptance guard still
applies. It helps steer around spectral limits, but does not remove shape
constraints or guarantee convergence. quality_limited means optimization
stopped at a guard, not that an optimal smooth mesh was found.
An optional deformation_degree (integer 2 through degree, default None) limits
the polynomial degree of the displacement while preserving the full initial
geometry map. This separates CAD resolution from optimization flexibility.
The displacement is shared across interfaces and zero on physical boundaries;
its lift and exact adjoint use FFT/DCTs. It is a restriction of the current
objective, not a solution-dependent accuracy penalty. To remove high modes
introduced by earlier smoothing, restart from the Coons maps: applying a low
degree correction to an already oscillatory map will retain its high modes.
Lower deformation degree can preserve spectral accuracy, at the cost of less
interface smoothing. It does not guarantee optimality or a PDE error bound.
The usual O(p^-2) Lobatto endpoint spacing is retained, and no RK4 stable step
is inferred. To choose a step, analyze the assembled coupled PDE operator.

CLI (NO EMBEDDED PHYSICAL CASE)

Create an importable application module exporting:

    def build_problem():
        return geometry, guide

Then, from the repository root:

    python3 -m experimental.stokes_chebyshev_atlas my_problem \
        --output /path/to/atlas --degree 8 --solution-degree 24

The command writes atlas.npz and atlas.json. Exit code 0 means the sampled
acceptance checks passed; code 2 means a saved candidate failed those checks.
The NPZ contains complex tensor coefficients (block, xi_mode, eta_mode),
geometry Lobatto nodes, shared node indices, fixed-node flags, and the report.
Sampling the coefficients on a different solution grid preserves shared traces.

VALIDATION

MappedLaplacian(atlas.coefficients, degree=p, flux_oversampling=2) provides
block-local physical Laplacian verification. conservative(values) applies
J^-1 div_reference(J*g^-1*grad_reference(values)) using FFT/DCT transforms.
Values have shape (blocks,p+1,p+1). With oversampling 2 or 3, interpolate the
solution first, evaluate rational metric fluxes on a finer nested Lobatto grid,
then differentiate and return to the solution nodes. No solution DOFs are added.
chain_rule(values, count=None) independently differentiates the interpolant
using analytic polynomial map first/second derivatives. gradient(values)
returns physical x/y derivatives, for example for checking interface fluxes.
No boundary conditions, interface enforcement, or global PDE solve are included.

The circle comparison is in output/circle-rectangle-auto/laplacian_verification.txt
with raw JSON and convergence/error plots. At degree 24 the visually smoothed
maps are LESS accurate for the manufactured functions tested, despite passing
the operator-tail criterion. Higher solution degree recovers accuracy. The
metric tail is not a bound on differentiated error of u composed with X.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest \
        experimental/stokes_chebyshev_atlas -q

Tests exercise polynomial exactness, DCT adjoints, the full objective gradient
including physical guide derivatives, reversed interfaces, curved boundary
preservation, null guides, non-forcing critical points, triangulation-invariant
one-block recovery, rotated eight-block recognition, multiple-hole recognition,
curved-hole region covering, distinct-arc preservation,
conforming merges, conditional splitting,
quality protection, length-unit invariance, actual field dependence, rejected
inversions, acceptance/rejection, export, and exact Goursat derivatives. These are implementation
checks, not a case study or evidence of a publishable performance advantage.

MATHEMATICAL SOURCE

The Goursat representation follows Xue, Waters & Trefethen (2024),
Computation of Two-Dimensional Stokes Flows via Lightning and AAA Rational
Approximation, SIAM J. Scientific Computing 46(2), A1214-A1234,
https://doi.org/10.1137/23M1576876 . The regularized metric, objective weights,
topology search, and acceptance policy here are prototype design choices.
