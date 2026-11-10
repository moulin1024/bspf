# Examples

High-level scripts using the Python bindings (`bspf_cpp`) on pybspf's benchmark
fields (`pybspf/examples/operation`). The fields, domains and settings are
collected in [`basic/benchmarks.py`](basic/benchmarks.py) with their exact
derivatives. Each script prints errors against the exact solution and a
refinement study; `--plot out.png` also saves a figure (needs matplotlib).

```sh
cmake --build build          # builds python/bspf_cpp/_core*.so
PYTHONPATH=python python examples/basic/diff_1d.py --plot diff_1d.png
```

| script | operation | field |
|---|---|---|
| `basic/diff_1d.py` | `Line.derivative`, spline-only comparison, batched fields | sin(x/(1.2 + cos x)) on [0, 2π] |
| `basic/integrate_1d.py` | `Line.quadrature(a, b)`, `Line.primitive` | same, from samples of f′ |
| `basic/interpolate_1d.py` | `Line.evaluate` values and derivatives off the grid | same |
| `basic/diff_2d.py` | `Grid.partial`, `Grid.gradient` | cosine modes + circular tanh front |
| `basic/diff_3d.py` | `Grid.gradient`, `curl`, `divergence`, `laplacian` | shifted Taylor–Green vortex, Chebyshev endpoints |
| `basic/diff_mapped_2d.py` | `MappedGrid2D`: physical first and second derivatives on a region bounded by four spline curves (transfinite map, exact metrics) | analytic f(x, y) on a curved quadrilateral |
| `basic/diff_multipatch_2d.py` | `MultiPatch2D`: physical derivatives on four conforming deformed patches; lines crossing a cut overlap into the neighbour by ghost points (vs one-sided), shared nodes averaged | analytic f(x, y) on a doubly connected ring around a hole |
| `basic/diff_cylinder_rectangle_2d.py` | `MultiPatch2D`: MMS first/second derivatives and Laplacian, including wall nodes, on four patches surrounding a cylinder | exponential boundary-layer factor multiplying sin(2x+0.5) cos(3y-0.2) + 0.5 exp(xy), outside a radius-0.5 disk in [-2,2] × [-1.5,1.5] |
| `basic/compare_cylinder_chebyshev.py` | compares BSPF (overlap 0 and 8) with four-block FFT Chebyshev differentiation at equal node counts | same cylinder boundary-layer MMS and polynomial patch maps |
| `pde/wave_2d.py` | linear wave equation as a first-order system + RK4, Dirichlet walls, with `Grid.weak_laplacian` (energy-stable Galerkin form) and Chebyshev jets | two crossing plane waves, time-dependent wall data |
| `pde/heat_2d.py` | time-dependent heat equation: `Grid.weak_laplacian` (energy-stable Galerkin form) + RK4, Dirichlet walls, Chebyshev jets | decaying mode with non-periodic, time-dependent wall data |
| `pde/burgers_2d.py` | 2D viscous Burgers system: `Grid.partial` advection + `Grid.weak_laplacian` diffusion + RK4, Dirichlet walls; jets chosen by Re | exact traveling-wave front (Fletcher), Re = 100 by default |
| `pde/shock_tube_1d.py` | BSPF Galerkin compressible Navier–Stokes + RK4, physical viscosity and heat conduction, rigid adiabatic walls | regularized Sod states; mass/energy balance, entropy and spatial refinement |
| `pde/laval_nozzle_2d.py` | 2D compressible Euler / Navier–Stokes on one `MappedGrid2D` block with the C++ `CompressibleSolver2D`: Galerkin BSPF weak form, weak inlet/outlet/wall fluxes, Legendre spectral vanishing viscosity, local-time-step RK4 with Anderson acceleration and grid sequencing | planar Laval nozzle, total-condition inlet, supersonic exit; isentropic invariants, mass flow, Re = 2000 boundary layers |
| `pde/laval_nozzle_mms.py` | manufactured-solution check of the nozzle solver's full Navier–Stokes operator on the curved block | smooth subsonic state, sympy source |

The 2D example draws its random mode amplitudes with NumPy (seed 123), so the
field differs slightly from pybspf's JAX-generated one.

Run the physical-viscosity shock tube with:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/pde/shock_tube_1d.py --sizes 65 129 257 --plot shock_tube_bspf.png --json shock_tube_bspf.json --npz shock_tube_bspf.npz
```

This needs SciPy in addition to the usual example dependencies. The model is
1D compressible Navier–Stokes on [0,1], with gamma=1.4, R=1, constant dynamic
viscosity mu=0.01, Pr=0.72 and kappa=mu*cp/Pr. The viscous flux is
`(0, tau, u*tau+kappa*T_x)` with `tau=4*mu*u_x/3`: there is no mass diffusion.
Initial primitive Sod states `(rho,u,p)=(1,0,1)` and `(0.125,0,0.1)` are joined
at x=0.5 with a fixed tanh width of 0.02. Defaults integrate to t=0.2.
`--mu`, `--prandtl`, `--width`, `--final-time`, `--cfl` and `--quadrature`
control the experiment. This is a regularized viscous initial-value problem,
not the discontinuous inviscid Sod Riemann problem.

The state consists of conservative-variable coefficients in the BSPF cardinal
space. Nonlinear physical fluxes are integrated by composite Gauss quadrature
between grid samples and spline knots. The weak residual uses BSPF test-function
derivatives, with mass solves at each RK4 stage. Momentum vanishes at both
walls; natural zero mass and total-energy fluxes impose rigid, insulated ends.
Mass and energy should stay constant; wall forces can change total momentum.
No finite-volume update, artificial viscosity, filtering, clipping or shock
enrichment is used. Nonphysical RK stages are rejected and retried at a smaller
step; persistent failures abort. Positivity is checked at nodes, quadrature
points, and on a dense output grid, not proven throughout the interpolant.

The BSPF settings are degree=9, n_basis=18, constraint_order=4, Chebyshev
endpoint estimation with 16 samples and 12 modes, alpha=1e-12, and **lam=0**.
The fit penalty is disabled to preserve constants in the Galerkin space.
The default `--backend fft` reuses the C++ Galerkin quadrature kernels:
shifted FFTs and spline corrections evaluate the state and its derivatives;
the weighted adjoint projects nonlinear fluxes. Knot-containing cells use
split quadrature with fixed-size corrections. The interior mass inverse is
low-rank; density/energy use that inverse plus a 2x2 boundary Schur complement
for the full mass solve. The diffusion spectral radius is obtained by
matrix-free generalized Lanczos. No dense cardinal/mass/stiffness matrices
are assembled or retained by this solver backend. `--backend dense` retains
the original Cholesky implementation as an independent reference. Both
backends use the same trial space and quadrature rule. The explicit
step estimate combines acoustic and full-space diffusion spectral scales,
and must be checked by time-step refinement. Reports include steps, physical
minima, entropy, mass/energy drift and differences from the finest computed
grid at common output points. These differences are not exact-solution errors.

A sharper physical front can be run with `--mu 0.001 --sizes 129 257` while
retaining the same Prandtl number and initial smoothing width. At t=0.2,
the measured 129-vs-257 L1 differences are approximately 1.54e-6 in density,
5.76e-6 in velocity, 2.30e-6 in pressure, and 2.66e-6 in temperature. Both runs
complete without rejected stages and keep positive density/pressure at the
checked points. Recorded relative mass/energy drift stays below 8e-14, and
the integrated entropy increases. These are physical-viscosity calculations;
no artificial viscosity or limiter is switched on in the sharper case.

Compare the two implementations with
`PYTHONPATH=python python bench/bench_shock_tube_backends.py --sizes 129 257 --final-time 0.2 --json shock_backend_comparison.json`.
This alternates warm RHS timings and optionally runs both backends to the
same final time, reporting final-profile differences, conservation and timings.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` externally for reproducibility.
The default benchmark only times RHS applications at 65,129,257,513 nodes.

Run the cylinder-in-rectangle operator verification with:

```sh
PYTHONPATH=python python examples/basic/diff_cylinder_rectangle_2d.py --sizes 17 33 65 129 --layer-thickness 0.1 --plot cylinder_mms.png --json cylinder_mms.json
```

The rectangle sides are straight; degree-15 polynomial arcs approximate the
circle, with the sampled radius error printed separately. Errors compare
physical derivatives against analytic expressions at the mapped nodes. This
is an operator MMS check, not a Poisson solve. For a manufactured Poisson
problem, use `s = -lap_exact(x, y)` in `-Delta u = s` and the exact field on
both the cylinder and rectangle walls. `--overlap 0` selects one-sided patch
operators for comparison. JSON output includes absolute and relative maximum
errors for each derivative and refinement level.

The manufactured field is `u = (1-exp(-(r-0.5)/delta))*u0`, where `u0` is
the smooth field in the table and `r = sqrt(x*x+y*y)`. It vanishes on the
cylinder and changes rapidly across a layer of thickness `delta` (default
0.1, controlled by `--layer-thickness`). Only the solution is modified;
the mesh is unchanged. Exact first/second derivatives include all derivatives
of the radial factor. Thinner layers need more nodes. With delta=0.1, the
relative maximum Laplacian errors at n=33,65,129 are approximately 1.22e-2,
2.03e-4,8.74e-7, including the wall nodes.

Compare against multiblock Chebyshev-Lobatto operators with:

```sh
PYTHONPATH=python python examples/basic/compare_cylinder_chebyshev.py --repeats 15 --plot cylinder_chebyshev.png --json cylinder_chebyshev.json
```

The Chebyshev implementation uses an even extension and real FFT to obtain
coefficients, the Chebyshev coefficient derivative recurrence, and an inverse
FFT to evaluate derivatives. It uses no dense differentiation matrices.
Both methods use analytic map metrics and average shared interface traces.
Chebyshev differentiates each block independently; the overlap-0 BSPF result
provides the corresponding interface treatment. Overlap-8 BSPF is also shown.
The domains and node counts match, but the node locations differ: Chebyshev
clusters at all block boundaries, while BSPF retains uniform reference nodes.
Errors are maxima at each method's own nodes, including walls.

For delta=0.1, Chebyshev Laplacian errors at n=17,33,65,129 are approximately
2.35e-4, 2.11e-8, 4.90e-10, 7.18e-9; the last increase is consistent with
round-off amplification in second derivatives at clustered endpoints.
BSPF errors are 6.03e-2, 1.22e-2, 2.03e-4, 8.74e-7. The global maxima
are effectively unchanged by overlap in this case, where the cylinder layer
dominates. Timings are warm median Laplacian applications including physical
chain rules and interface averaging, excluding setup and exact-field sampling.
They compare this NumPy FFT implementation with the existing C++ BSPF wrapper;
they are not a comparison of equally optimized kernels or a PDE solver benchmark.

To add a strong concentric shock ring while retaining the wall layer:

```sh
PYTHONPATH=python python examples/basic/compare_cylinder_chebyshev.py --sizes 33 65 129 257 513 --shock-amplitude 10 --shock-radius 1 --shock-width 0.02 --repeats 7 --plot cylinder_shock_chebyshev.png --json cylinder_shock_results.json
```

This uses `u = B*(u0+S)`, `B = 1-exp(-(r-0.5)/0.1)`, and
`S = A/2*(1-tanh((r-rs)/w))`, with A=10, rs=1 and w=0.02. The
shock ring is centered on the cylinder, inside the existing four patches;
neither mesh is adapted to it. The 10–90% transition width is about 0.0439.
This is a smooth, stationary shock-like scalar MMS with exact derivatives,
not a discontinuity or an Euler shock propagation calculation. The field
still vanishes on the cylinder. Set `--shock-amplitude 0` for the earlier case.
The plot includes the exact field, exact Laplacian and a radial profile;
JSON includes all first/second derivative errors, timings and worst Laplacian
error coordinates.

For this ring the coarse grids are under-resolved: maximum Laplacian errors
can exceed 100% and need not decrease monotonically. Relative nodal maximum
Laplacian errors (including walls) are:

| n per block direction | BSPF overlap 8 | Chebyshev FFT |
|---|---:|---:|
| 33 | 1.916 | 1.912 |
| 65 | 2.078 | 3.241 |
| 129 | 1.115 | 2.400 |
| 257 | 1.015e-1 | 1.922e-1 |
| 513 | 8.490e-4 | 1.324e-4 |

Overlap 0 and 8 have effectively identical global maximum errors for these
runs. Chebyshev has the smaller fine-grid error, but BSPF has the smaller
Laplacian error at n=65–257; the boundary-layer-only ranking does not extend
uniformly to an interior shock ring. Errors use each method's own nodes.

### Adaptive tanh enrichment experiment

`pde/shock_tube_enrichment.py` augments the structured FFT BSPF space with
moving tanh modes. Density and pressure gradients detect up to two separated
fronts; local jumps and slopes estimate their widths. No shock trajectory is
provided. The basis is fixed during each RK4 step, then updated at a specified
interval. Each update uses an L2 projection that preserves mass and total
energy (constants belong to both spaces). This projection can nevertheless
introduce numerical damping; it is not an entropy-preserving remap.

The added functions vanish at the walls and are orthogonalized against the
interior BSPF basis. Full and wall-constrained mass inverses use the existing
FFT mass inverse plus a small Schur complement. Physical stress and heat
conduction are unchanged. There is no added viscosity, FVM flux, or coefficient
filter. This is an exploratory method, without a positivity or entropy-stability
guarantee. Positivity failures trigger stage retries or rejection of a remap.

Run from the repository root:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python examples/pde/shock_tube_enrichment.py --n 65 --reference-n 1025 \
  --mu 0.0001 --time 0.1 --width 0.01 \
  --prefix benchmarks/shock_tube/enrichment_n65_mu0001_fine

PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python bench/bench_shock_enrichment.py
```

The second command reuses the saved reference and checks quadrature, time step,
update interval, fixed enrichment, and multiple widths per detected front.
`--scales 0.25 0.5 1` on the first command enables three tanh widths per front.
JSON reports, NPZ profiles, and PNG plots are saved under the selected prefix.

Observed L1 errors against the 1025-node BSPF reference, at mu=1e-4, t=0.1,
65 base nodes, initial tanh width 0.01, and 12-point quadrature:

| Method | Density | Velocity | Pressure | Temperature |
| --- | ---: | ---: | ---: | ---: |
| BSPF | 0.005671 | 0.024755 | 0.007517 | 0.009982 |
| Adaptive, update 0.002 | 0.003836 | 0.016455 | 0.004826 | 0.006370 |
| Adaptive, update 0.001 | 0.002606 | 0.009013 | 0.002814 | 0.003147 |
| Adaptive, update 0.0005 | 0.001847 | 0.005449 | 0.001832 | 0.003571 |
| Fixed initial enrichment | 0.006985 | 0.025719 | 0.008635 | 0.010362 |

For update 0.001, recorded relative mass/energy drift stayed below 1.5e-13,
with no rejected stages or remaps. Ringing remained: velocity Linf error was
0.242 versus 0.265 for the base method. At update 0.002 it was slightly worse
than the base method (0.269). Multiple widths did not consistently improve
accuracy. More frequent remapping is not uniformly better for every variable.

Doubling quadrature changed the default final conservative profiles by less
than 2.7e-8 in max norm; halving CFL changed them by about 5e-5. In contrast,
halving the update interval changed them by 0.044. Thus remapping is a material
part of the method, and these results do not isolate basis approximation from
projection-induced dissipation. The base method initializes by interpolation;
the enriched method uses L2 projection of the same analytic initial profile.
This is a smoothed initial jump, not a discontinuous initial condition.

The 513-to-1025 reference differences in L1 were 6.2e-5 (density), 3.0e-4
(velocity), 8.9e-5 (pressure), and 1.4e-4 (temperature), much smaller than the
coarse errors but not an exact-solution certification. This experiment does not
establish superiority over an adaptively resolved Chebyshev method.

### Retained local spline enrichment

`pde/shock_tube_spline_enrichment.py` replaces the small tanh family with local
quintic B-splines. The same solution-based sensor selects patches around the
fronts. A fixed fine lattice supplies new spline centers, and selected centers
are retained as the fronts move. Existing independent modes are preserved;
only newly added directions undergo a dependency check. Consequently updates
form nested spaces, up to roundoff, rather than repeatedly discarding resolved
information. No front trajectory or exact solution enters the sensor.

The global BSPF FFT evaluation/projection and mass inverse remain active.
Enrichment evaluation and the small mass Schur complement add work proportional
to the number of local modes. The generalized eigenvalue estimate includes the
added modes for explicit time stepping. Patches are selected every 0.005 time
units by default; their halfwidth is two base node spacings. Physical viscosity
and heat conduction remain the only explicit dissipative PDE terms.

After generating the 1025-node reference with the previous section's command:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python examples/pde/shock_tube_spline_enrichment.py --refinement 8 \
  --quadrature 64 --prefix benchmarks/shock_tube/spline_nested8_q64
```

For the same mu=1e-4, t=0.1, initial width 0.01 experiment, the final space has
65 base modes plus 172 enrichment modes per field. Errors against the
1025-node reference were:

| Field | Base BSPF Linf | Tanh, update 0.001 Linf | Local spline Linf | Local spline L1 |
| --- | ---: | ---: | ---: | ---: |
| Density | 0.04561 | 0.03942 | 0.002824 | 0.0001844 |
| Velocity | 0.26491 | 0.24177 | 0.01410 | 0.0006619 |
| Pressure | 0.06184 | 0.04759 | 0.003833 | 0.0002741 |
| Temperature | 0.12742 | 0.09194 | 0.005447 | 0.0002489 |

There were no rejected stages or remaps. Recorded mass/energy drift was below
1.8e-14. The maximum L2 remap error was 2.2e-12, so repeated projection damping
is no longer the mechanism reducing the oscillations. With quadrature 32,
peak errors differed by less than 0.4%; the maximum change in the conservative
profiles was 5.5e-4. Piecewise spline quadrature is not exactly integrated by the
structured rule, so quadrature refinement remains necessary.

The improvement comes from additional local degrees of freedom, not an
oscillation-free 65-mode representation. This experiment retains all selected
centers and does not yet coarsen old patches. The existing fine BSPF solution is
a numerical reference, not an exact solution, and the initial data remain
smoothed. Small residual ringing remains. The unchanged physical model and
retained spaces do not imply a general entropy-stability or positivity theorem.

A further 16x local-refinement check (65 + 341 modes, quadrature 64) reduced
velocity Linf to 0.00428 and pressure Linf to 0.00245, but increased density,
velocity, and pressure L1 errors to 0.000474, 0.000982, and 0.000601. Thus this
study does not establish monotonic convergence in refinement; the 8x case
remains the default. The finer case would need additional quadrature and
conditioning checks before using it to claim a convergence rate. Its report
is `benchmarks/shock_tube/spline_nested16.json`.

Cost comparison for the retained 65 + 172 mode experiment can be reproduced
with:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_enrichment_cost.py
```

Three single-threaded runs of the same mu=1e-4, t=0.1, initial-width=0.01,
CFL=0.35 problem gave the following medians on the development machine.
End-to-end includes setup, evolution, diagnostics and final probe evaluation;
it excludes imports, plotting and file output.

| Configuration | Final modes per field | Steps | Total seconds | Final-space RHS milliseconds |
| --- | ---: | ---: | ---: | ---: |
| Original BSPF, quadrature 12 | 65 | 117 | 0.124 | 0.190 |
| Original BSPF, quadrature 64 | 65 | 117 | 0.460 | 0.702 |
| Enriched BSPF, quadrature 64 | 237 | 897 | 12.729 | 3.297 |

Thus the implemented enrichment costs about 103x the original run, or 27.7x
at matched quadrature, despite a final coefficient count only 3.65x larger.
The local fine modes require 7.67x as many explicit steps. Final-space RHS cost
is 17.3x the original, or 4.70x at matched quadrature. This microbenchmark uses
the final enriched space, whereas that space grows throughout the evolution.
Approximately 67% of the enriched runtime is in RHS calls, 18% in space
rebuilds (including spectral-radius estimation), and 15% in remaining work.
These are implementation timings, not equal-accuracy comparisons or predicted
asymptotic costs. The original coarse solution has substantially larger errors.
Raw timings and environment are saved to
`benchmarks/shock_tube/enrichment_cost.json`.

### Equal final mode count: global 237 versus 65 + 172

`bench/bench_shock_equal_modes.py` repeats both evolutions three times with the
same physical parameters, initial analytic profile, CFL, and 1025-node reference.
It compares original BSPF at 237 nodes using quadrature 12 and 64 with retained
spline enrichment using quadrature 64. It uses the same end-to-end timing scope
as the preceding cost benchmark.

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python bench/bench_shock_equal_modes.py
```

| Quantity | Global BSPF 237, q12 | Global BSPF 237, q64 | BSPF 65 + 172, q64 |
| --- | ---: | ---: | ---: |
| Median runtime (s) | 3.186 | 16.123 | 12.876 |
| Steps | 686 | 686 | 897 |
| Density Linf | 0.01139 | 0.01139 | 0.002824 |
| Velocity Linf | 0.06103 | 0.06103 | 0.01410 |
| Pressure Linf | 0.01558 | 0.01558 | 0.003833 |
| Temperature Linf | 0.03380 | 0.03380 | 0.005447 |
| Density L1 | 0.0007204 | 0.0007204 | 0.0001844 |
| Velocity L1 | 0.003309 | 0.003309 | 0.0006619 |
| Pressure L1 | 0.0009843 | 0.0009843 | 0.0002741 |
| Temperature L1 | 0.001435 | 0.001435 | 0.0002489 |

Original BSPF at 237 nodes and q12 is 4.04x faster than the enriched solver,
with peak errors 4.0--6.2x larger. Increasing the global method's quadrature from
12 to 64 changes its final conservative profiles by only 2.8e-9 in max norm;
therefore q64 adds cost without a meaningful accuracy benefit in this case.
The enrichment uses its coefficients more effectively near the front, but does
not win on runtime at equal final coefficient count. The enriched count grows
during evolution, while the global count is fixed. All runs had zero rejected
stages and recorded relative mass/energy drift below 1.8e-14.

Reports, profiles, and the visually checked comparison plot are saved as
`benchmarks/shock_tube/equal_modes237.{json,npz,png}`. This comparison does not
yet establish which approach is faster at equal error tolerance.

### Monotone Sod profiles with relaxed stabilization constraints

`pde/shock_tube_monotone.py` is a deliberately problem-specific 1D experiment.
It retains the original 65-mode BSPF space and FFT Galerkin RHS, but applies
conservative shape-constrained projections after all three SSPRK3 stages.
Each scalar projection minimizes a mass-matrix norm subject to linear
constraints. Density is made decreasing and bounded by the initial Sod states;
velocity has a single detected maximum and vanishes at the walls; pressure is
made decreasing and bounded by the initial pressures. With `--temperature-shape`,
temperature is allowed one detected trough followed by one peak. Extrema are
detected from the evolving state, not from the reference. These are pre-reflection
Sod shape assumptions, not general Navier--Stokes invariants.

The coupled pressure and temperature constraints become linear in energy after
density and momentum have been limited. Each projection preserves the integral
of the conservative component. Constraints start on a coarser sample set and
are augmented at violations found on an 8193-point global audit grid. These
points add constraints, not trial-space modes or local mesh refinement. An
independent 32769-point audit checks the final profile. None of these sampled
checks proves monotonicity at every point of the continuum.

A completed run with monotone density/pressure and no additional temperature
extrema above prominence 1e-4 uses explicit artificial conservative diffusion
of strength 0.002, in addition to physical mu=1e-4:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python examples/pde/shock_tube_monotone.py --temperature-shape \
  --artificial-diffusion 0.002 --prefix benchmarks/shock_tube/monotone_combined65

PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python bench/bench_shock_monotone.py --input benchmarks/shock_tube/monotone_combined65
```

This adds epsilon times U_xx to all three conservative equations, including
artificial mass diffusion. It is numerical regularization, not a change to the
reported physical viscosity. Projection also introduces numerical dissipation.
The initialized profile is conservatively limited before evolution. There is
no finite-volume blend and no added basis mode. The reference is used only for
post-run error measurements and plotting.

At t=0.1 the completed combined run took 143 steps, about 76.5 seconds, of which
75.3 seconds were spent filtering. Relative mass and total-energy drift were
below 7e-15. Independent sampled total upward variation was 2.2e-11 for density
and 3.7e-10 for pressure. Velocity was nonnegative and unimodal within the sampled
tolerances; temperature had one trough and one peak at prominence 1e-4.

| Metric | Original BSPF 65 | Combined constrained BSPF 65 | Reference 1025 |
| --- | ---: | ---: | ---: |
| Minimum density | 0.11085 | 0.12518 | 0.12500 |
| Minimum pressure | 0.07489 | 0.10000 | 0.09999 |
| Minimum velocity | -0.21979 | approximately 0 | -0.000065 |
| Density Linf error | 0.04561 | 0.05974 | — |
| Velocity Linf error | 0.26491 | 0.37114 | — |
| Pressure Linf error | 0.06184 | 0.08165 | — |
| Temperature Linf error | 0.12742 | 0.18940 | — |
| Pressure-shock 90-to-10 width | ambiguous with ringing | 0.03023 | 0.004032 |

The oscillations are suppressed at the expense of broadening: the shock is
about 7.5 times wider than the reference and peak errors increase. This is a
working shape-preservation experiment, not a more accurate or scalable 3D
solver. Dense SLSQP constraint solves dominate its runtime.

Other attempted configurations are retained for comparison:

- Without artificial diffusion, the default density/pressure/velocity limiter
  completed, but temperature still had additional extrema.
- Adding hard temperature constraints without artificial diffusion encountered
  a numerical SLSQP failure after the last reported time 0.05967. This is not
  a proof of infeasibility; see `monotone_temperature_failed.json`.
- `pde/shock_tube_regularized.py` tests uniform artificial diffusion alone after
  initial limiting. Large coefficients remove visible ringing but smear all
  waves excessively; reports are `regularization.json` and
  `regularization_strong.json` under `benchmarks/shock_tube`.

The use of optimization-based shape filtering is motivated by
[structure-preserving nonlinear filtering](https://epubs.siam.org/doi/10.1137/20M1337223).
Our sequential constraints, shock-tube shape assumptions, and implementation
are experimental and do not inherit a general entropy-stability theorem.

### Selective damping instead of hard monotonicity constraints

`pde/shock_tube_spectral_viscosity.py` tests two weaker stabilizations while
retaining the original BSPF space and initialization:

- Mass-orthonormal eigenmodes of K z = lambda M z are ordered by eigenvalue.
  Modes below a cutoff are untouched; damping increases quadratically above
  that cutoff. Full and wall-constrained spaces use separate bases. Constants
  are undamped, preserving mass and total energy integrals.
- Optional compression viscosity adds a conservative numerical flux
  nu(x) U_x with nu = C h min(h max(-u_x,0), |u|+c). It acts on all conserved
  variables, including density, and is therefore explicitly artificial.

No constrained stage projection, extra modes, or local mesh refinement is used.
The modal eigendecomposition is dense and intended for the 1D experiment;
this is not yet a scalable FFT implementation of the stabilization in 3D.
Neither damping term guarantees positivity or monotonicity for compressible NS.
The selection of the tested parameters uses the reference for assessment;
the evolving stabilization coefficients do not use the reference solution.

Reproduce a useful compromise at 65 modes:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/pde/shock_tube_spectral_viscosity.py --cutoffs .2 --rates 50 \
  --local-strengths .1 --prefix benchmarks/shock_tube/selective_candidate
```

Same problem as above: mu=1e-4, t=0.1, initial width=0.01, quadrature 12,
CFL=0.35, and the 1025-node numerical reference.

| Metric | Original 65 | Hard constraints + diffusion | Selective damping 65 |
| --- | ---: | ---: | ---: |
| Density L1 | 0.005671 | 0.009061 | 0.002311 |
| Velocity L1 | 0.024755 | 0.023016 | 0.009536 |
| Pressure L1 | 0.007517 | 0.010511 | 0.002960 |
| Temperature L1 | 0.009982 | 0.012738 | 0.004836 |
| Density Linf | 0.04561 | 0.05974 | 0.04470 |
| Velocity Linf | 0.26491 | 0.37114 | 0.35617 |
| Pressure Linf | 0.06184 | 0.08165 | 0.06591 |
| Temperature Linf | 0.12742 | 0.18940 | 0.13570 |
| Max velocity error on 0.70 < x < 0.90 | 0.14419 | 0.06298 | 0.02681 |
| Pressure 90-to-10 width near the shock | 0.01134 | 0.03023 | 0.01687 |

For oscillatory profiles, the shock-width diagnostic chooses each descending
level crossing closest to the reference shock midpoint; it is not a monotonicity
measurement. The reference width is 0.004032. The selective case took about
0.33 seconds, 175 steps, zero rejections, and had mass/energy drift below 1e-15.
These are single-run timings. The hard-constraint run took about 76.5 seconds.
Doubling quadrature or halving CFL changed the final conservative profiles by
less than 1.7e-6 in max norm. Eleven relevant Python tests passed.

The selective case is substantially more accurate in L1 than the hard limiter
and the original 65-node run, but it still rings and increases peak velocity
error relative to the original. It is a compromise, not elimination of Gibbs.
Sweeps of cutoff (0.2, 0.4, 0.6, 0.8), damping rate (100, 300, 1000), compression
strength (0.05, 0.1, 0.2, 0.5), and additional 129/237-node runs did not establish
a setting that both eliminates ringing and preserves all peak errors.

Artifacts: `spectral_viscosity.json`, `compression_viscosity.json`,
`selective_resolution.json`, `selective_validation.json`, and
`selective_comparison.{json,png}` under `benchmarks/shock_tube`.
The approach is motivated by selective and localized dissipation, as in
[adaptive spectral viscosity](https://www.math.umd.edu/~tadmor/pub/spectral/Tadmor-Waagan%20ASV%20SISC12.pdf);
our BSPF formulation and tests do not inherit that paper's guarantees.

### Faster enrichment with sparse compact-support evaluation

The spline example now defaults to `--backend sparse`; `--backend dense`
retains the earlier sampling-matrix implementation. The trial space, detector,
retention policy, mass Schur complement, and PDE are unchanged. The Python
`LocalRefinement` factory still defaults to `SplineSpace` for reference
benchmarks; pass `space_type=SparseSplineSpace` to select the optimized path.

The orthogonalized enrichment is H=(S-Phi C)T, where S is the compact-support
spline sampling matrix, C its interior BSPF projection, and T the accepted basis
transformation. Instead of storing and multiplying dense H and H_x, the new
backend evaluates Phi(b-C T a)+S T a. Its weighted adjoint projection is
T^T(S^T W f-C^T Phi^T W f). CSR matrices retain the local support of the raw
quintic splines, while the BSPF term uses the existing structured FFT operations.
Previously computed columns of C are reused when the space grows. The previous
largest-eigenvalue vector seeds the next spectral-radius estimate. Dense H/H_x
are still used during setup and then discarded; peak setup memory is therefore
not reduced by the same factor as retained sampling-array storage.

Three matched, single-threaded full runs at mu=1e-4, t=0.1, and 65 + 172 final
modes gave these medians:

| Backend | Quadrature | Runtime (s) | Speedup | Final-space RHS (ms) | Sampling arrays (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense enrichment | 64 | 13.97 | 1x | 3.436 | 12.09 |
| Sparse enrichment | 64 | 7.10 | 1.97x | 1.234 | 0.251 |
| Sparse enrichment | 32 | 4.61 | 3.03x | 0.784 | 0.125 |

Sampling-array storage counts H/H_x versus CSR S/S_x only, not the full solver.
The final mode count remains 172 in both sparse variants. The q64 variant uses
897 steps, the q32 variant 893; neither rejects stages or remaps. Sparse q64
final conservative profiles differ from dense q64 by at most 1.4e-6. Sparse q32
differs by 5.5e-4, principally due to the quadrature change already observed
with dense enrichment. All four primitive peak errors change by less than
0.6%, and L1 errors by less than 2.6%, when comparing sparse q32 with dense q64.
Relative mass/energy drift stays below 4e-12. Algebraically equivalent factored
actions change floating-point cancellation in the nearly dependent enrichment
basis, so results are not bitwise identical.

A narrower patch (halfwidth one base spacing rather than two) reduced the count
to 149, but approximately doubled density and pressure L1 errors. It is not the
recommended default. The demonstrated improvement is in implementation and
quadrature cost, rather than a lower-dimensional representation. The dense
small transforms and Schur solves, explicit fine-scale time-step restriction,
and accumulated retained modes remain scaling limits.

```bash
# Preserve the previous quadrature accuracy:
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python examples/pde/shock_tube_spline_enrichment.py --backend sparse \
  --quadrature 64 --prefix benchmarks/shock_tube/spline_sparse64

# Faster option, checked against quadrature 64:
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python examples/pde/shock_tube_spline_enrichment.py --backend sparse \
  --quadrature 32 --prefix benchmarks/shock_tube/spline_sparse32

# Reproduce matched timings, errors and retained sampling storage:
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_sparse_enrichment.py
```

The report and profiles are `benchmarks/shock_tube/sparse_enrichment_cost.json`
and `.npz`. Sixteen relevant Python tests passed, including dense/sparse
operator and RHS agreement through successive basis expansions.


### Retained-basis transfer and larger time steps

The sparse spline solver now injects old coefficients exactly into its retained
basis, avoiding repeated L2 remapping. It shares enrichment transforms between
value/gradient evaluation, avoids an unused full momentum mass solve, reuses
unchanged spaces, and skips adaptation at the final time. The latter removes
five unused modes; this is not a reduction of the space used during evolution.
No artificial damping is added.

A new matched three-repeat benchmark (single-threaded, q32, mu=1e-4, t=0.1)
gave:

| Variant | Median seconds | Steps | Final enrichment modes |
| --- | ---: | ---: | ---: |
| Previous, CFL 0.35 | 2.955 | 893 | 172 |
| Updated, CFL 0.35 | 2.806 | 893 | 167 |
| Updated, CFL 0.70 | 1.883 | 455 | 167 |

The implementation alone is 1.05x faster; with CFL 0.70 it is 1.57x faster.
These are matched timings from this run, not ratios against earlier timings
under different machine load. All primitive peak and L1 errors change by less
than 0.04% relative to the previous q32 run. Maximum conservative-profile
change is 3.42e-6. Relative mass/energy drift is below 2.6e-15, versus 3.5e-12
previously. No stages or remaps were rejected. Errors use the saved 1025-node
numerical reference, not an exact solution.

CFL 0.70 is validated for this case by comparison with CFL 0.35; it is not a
stability guarantee for other flows. Conservative CLI defaults remain intact.

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
python examples/pde/shock_tube_spline_enrichment.py --backend sparse \
  --quadrature 32 --cfl .7 --prefix benchmarks/shock_tube/enrichment_v2_cfl07

PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_enrichment_v2.py

PYTHONPATH=python MPLBACKEND=Agg python bench/plot_shock_enrichment_v2.py
```

The benchmark requires the saved pre-change Python sources under
`benchmarks/shock_tube/enrichment_before_v2`. Reports/profiles are
`enrichment_v2_benchmark.{json,npz}` and plots are
`enrichment_v2_results.{png,pdf}` in the same output directory.
All 18 relevant Python tests pass, including retained-coefficient transfer and
absence of unused final-time adaptation.

### Support-only spline sampling and cached probes

A further implementation optimization evaluates raw splines only inside their
compact supports, reuses quadrature samples during setup, and caches sparse
transposes. Front detection/remap checks share a single BSPF probe matrix across
spaces and a sparse probe matrix per space. Cached grids are copied and checked
by value, so changing the probe array invalidates the cache. Accepted RK4 trial
samples also supply the next step's CFL calculation, avoiding repeat evaluation.

Three fresh-process, single-threaded comparisons against commit `6866d47` at
q32, CFL 0.70, mu=1e-4 and t=0.1 gave median times **1.912 -> 1.396 seconds**:
**1.37x speedup, or 27% less runtime**. Both use 455 steps and 65 + 167 modes.
Final sampled conservative profiles are bit-for-bit identical in this benchmark;
all reported errors and conservation diagnostics are unchanged. No damping,
quadrature change, mode reduction, or time-step increase contributes to this gain.

The cache retains one 2001-by-65 base probe matrix (about 0.99 MiB), plus a sparse
probe matrix and copied coordinates. Dense basis setup and small mass transforms
still remain; this is not a demonstration of 3D scaling. Nineteen focused Python
tests pass, including unsorted/duplicate probes and cache invalidation.

To reproduce the comparison using the existing saved numerical reference:

```bash
mkdir -p benchmarks/shock_tube/enrichment_before_v3
for module in shock_tube_enrichment shock_tube_spline_enrichment; do
  git show "6866d47:examples/pde/${module}.py" > "benchmarks/shock_tube/enrichment_before_v3/${module}.py"
done
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_enrichment_v2.py \
  --baseline-dir benchmarks/shock_tube/enrichment_before_v3 --baseline-cfl .7 \
  --prefix benchmarks/shock_tube/enrichment_v3_benchmark
```

Results are `benchmarks/shock_tube/enrichment_v3_benchmark.{json,npz}`.

### Faster stability-radius setup for small bases

The sparse spline setup now applies the enriched mass/stiffness operators in
blocks during generalized Lanczos. It reuses the existing mass blocks, builds
temporary stiffness coupling blocks, and caches the two base matrices only for
`n <= 128`. At n=65 this permanent cache costs 66 KiB; its limit is 256 KiB.
Larger bases use native matrix-free base products. The stiffness coupling blocks
are released after setup. Time stepping still uses sparse enrichment sampling,
structured FFT evaluation/projection, and the coupled mass inverse.

Three matched fresh-process runs against the previous support-only implementation
at the same q32/CFL 0.70 settings gave **1.407 -> 1.265 seconds** median runtime:
**1.11x speedup, 10.1% less runtime**. Both use 455 steps and 167 added modes.
Maximum final conservative-profile difference is 4.9e-9; all four peak errors
change by less than 0.00001%. The difference comes from rounding in the equivalent
stability eigenproblem, which affects step sizes slightly. There is no added
damping or change to the spatial approximation. Twenty-two focused tests pass,
including block-product agreement on both sides of the cache-size threshold and
the dense-backend fallback. This small-base optimization does not establish 3D
scalability.

The final report is `benchmarks/shock_tube/enrichment_v4_benchmark.json`, with
profiles in the matching `.npz`. It uses saved pre-change Python modules and
backend under `benchmarks/shock_tube/enrichment_before_v4`:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_enrichment_v2.py \
  --baseline-dir benchmarks/shock_tube/enrichment_before_v4 --baseline-cfl .7 \
  --prefix benchmarks/shock_tube/enrichment_v4_benchmark
```

### Combined mass solve and retained Gram blocks

The sparse RHS now solves the full coupled mass system for all three fields in
one call. Momentum then receives a rank-two correction enforcing the two zero
wall values. With B selecting the wall degrees of freedom, the precomputed lift
is L=M^-1 B (B^T M^-1 B)^-1, so the momentum result is z-L B^T z. This is the
same constrained mass solve, not a penalty or damping term. The wall lift adds
two columns of storage (about 3.6 KiB for 232 total modes).

When adding enrichment directions, setup reuses the retained mass Gram block
and the already evaluated old basis columns. Only new mass coupling blocks
are formed. The old/new blocks are checked against direct quadrature through
successive basis expansions.

Three matched fresh-process timings at q32/CFL 0.70 gave **1.326 -> 1.191 seconds**
median runtime: **1.11x speedup, 10.2% less runtime** versus the previous version.
Both use 455 steps and 167 added modes. Maximum conservative-profile difference
is 1.63e-8; all primitive peak errors change by less than 0.00001%. Relative mass
and energy drift remain below 2.8e-15, with no stage or remap rejections.
Twenty-three focused tests pass, including direct residual checks of the combined
mass solve and exactly zero wall momentum rates. The small numerical changes
reflect reassociation of equivalent algebra; no quadrature, CFL, or basis-size
change contributes to the speedup.

Report/profiles: `benchmarks/shock_tube/enrichment_v5_benchmark.{json,npz}`.
The pre-change modules are saved in `benchmarks/shock_tube/enrichment_before_v5`.

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_enrichment_v2.py \
  --baseline-dir benchmarks/shock_tube/enrichment_before_v5 --baseline-cfl .7 \
  --prefix benchmarks/shock_tube/enrichment_v5_benchmark
```

### Ordinary BSPF at matched ringing error

A resolution sweep of ordinary (unenriched) FFT BSPF used 237, 321, 385, 432,
449, 450, 480, 500, 512, and 513 modes. All cases use mu=1e-4, t=0.1, analytic
initial width 0.01, and CFL 0.70. Ordinary quadrature is 12; enriched quadrature
is 32. Ordinary initial data are interpolated, while enrichment uses L2
projection. Timings include setup, evolution, diagnostics, and final probes.

Ringing is measured by the maximum error relative to the saved 1025-node
numerical reference in two fixed bands, [0.64,0.67] and [0.69,0.73], beside the
shock. This is a reproducible error proxy, not a decomposition of pure Gibbs
amplitude. A stricter comparison also uses whole-domain peak error. The ratios
below are the worst ratio across density, velocity, pressure, and temperature;
ratios at or below one mean each field matches or improves its enriched error.

| Method | Final modes/field | Steps | Median seconds (3 runs) | Beside-shock error ratio | Whole-domain peak error ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Enriched BSPF | 65 + 167 | 455 | 1.166 | 1.000 | 1.000 |
| Ordinary BSPF | 237 | 343 | 1.155 | 6.108 | 6.210 |
| Ordinary BSPF | 450 | 955 | 2.812 | 0.944 | 1.274 |
| Ordinary BSPF | 480 | 1065 | 3.297 | 0.816 | 0.944 |

Ordinary 450 matches the beside-shock error and runs 2.41x slower. Its density,
velocity, and pressure peak errors are already lower, but temperature peak error
is 27% higher. Ordinary 480 matches or improves all four peak errors and all four
L1 errors, and runs 2.83x slower. Thus this experiment supports approximately
450--480 ordinary modes for comparable error, versus 232 final enriched modes.
This is a bracket from tested resolutions, not a proven minimum. Ordinary 449
also passes the beside-shock criterion but was slower in the initial sweep;
FFT size affects runtime non-monotonically.

Doubling ordinary quadrature to 24 changes primitive profiles by at most 1.3e-9;
halving CFL changes them by at most 2.5e-6. These checks do not change the
comparison. Absolute errors still depend on the numerical reference's accuracy.
These are measurements of the current implementations on this machine, not a
hardware-independent speedup or 3D scaling result.

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_matched_error.py --modes 237 450 480 --repeats 3 \
  --prefix benchmarks/shock_tube/matched_error_timing
PYTHONPATH=python MPLBACKEND=Agg python bench/plot_shock_matched_error.py
```

Reports/profiles/plots: `benchmarks/shock_tube/matched_error_timing.{json,npz,png,pdf}`.
Sweeps: `matched_error.{json,npz}` and `matched_error_refined.{json,npz}`.
Validation: `matched_error_validation.json`.

### Optimizing ordinary BSPF before comparing speed

The ordinary solver now reuses accepted quadrature samples for the CFL estimate,
first RK stage, and diagnostics. Its three fields share one full mass solve,
followed by the same rank-two zero-wall momentum correction used by enrichment.
The sample reuse preserves overridden RHS/timestep methods in the regularization
examples. All physical state checks remain in place.

Matched fresh-process before/after medians (three runs each) were:

| Ordinary modes | Previous seconds | Updated seconds | Speedup |
| --- | ---: | ---: | ---: |
| 237 | 1.040 | 0.925 | 1.12x |
| 450 | 2.622 | 2.286 | 1.15x |
| 480 | 3.204 | 2.841 | 1.13x |

Maximum final conservative-profile change is below 5.7e-11. Thirty-three focused
tests pass, including the regularization subclasses, cached sampling, and direct
agreement with the previous separate full/interior mass solves.

A separate interleaved three-repeat comparison of the current implementations
supersedes the earlier speed advantage:

| Method | Median seconds | Time / enriched | Error status |
| --- | ---: | ---: | --- |
| Enriched 65 + 167 | 1.229 | 1.00 | Target |
| Ordinary 237 | 1.022 | 0.83 | Up to 6.2x larger peak error |
| Ordinary 450 | 2.438 | 1.98 | Matches beside-shock error; temperature peak still 27% higher |
| Ordinary 480 | 2.995 | 2.44 | Improves all four peak field errors |

The evidence now supports approximately **2.0--2.4x** speed advantage for the
current enriched implementation at comparable error. Do not use the older
2.4--2.8x claim without noting its less optimized ordinary baseline. Neither
implementation is claimed to be optimally tuned; timing varies with machine
load. The numerical parameters, error bands, and reference are unchanged.

Before/after report: `benchmarks/shock_tube/ordinary_optimization.json` (requires
`ordinary_before_opt/shock_tube_1d.py` to rerun `bench/bench_shock_ordinary_optimization.py`).
Updated comparison: `matched_error_optimized_ordinary.{json,npz,png,pdf}`.

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_shock_matched_error.py --modes 237 450 480 --repeats 3 \
  --prefix benchmarks/shock_tube/matched_error_optimized_ordinary
PYTHONPATH=python MPLBACKEND=Agg python bench/plot_shock_matched_error.py \
  --prefix benchmarks/shock_tube/matched_error_optimized_ordinary
```

### Exact viscous traveling-shock verification

`examples/pde/viscous_traveling_shock.py` implements a source-free Becker shock
for a perfect gas with R=1, gamma=1.4, constant dynamic viscosity, and Pr=3/4.
This intentionally changes the earlier shock-tube Pr=0.72 to obtain an exact
reference. The profile follows the constant-transport Becker relation in
[B. M. Johnson, Closed-form shock solutions, Eq. (3.3)](https://arxiv.org/abs/1403.6754).
A compression ratio of two permits an analytic quadratic inversion.

Let xi=x-x0-s*t, and let v=u-s denote shock-frame velocity. The upstream state
is rho0=T0=p0=1, v0=sqrt(3.5), and the downstream state has rho1=2, v1=v0/2,
T1=1.375 and p1=2.75. Define

```text
cp = gamma/(gamma-1), m = rho0*v0, H = cp*T0 + v0^2/2
w = 8*mu*gamma / (3*m*(gamma+1))
A = exp(xi/w)/2
q = 2 / (2 + A + sqrt(A*A + 4*A))
v = v1 + (v0-v1)*q
rho = m/v, T = (H-v*v/2)/cp, u = v+s, p = rho*T
U = (rho, rho*u, rho*(T/(gamma-1) + u*u/2))
```

The code uses an equivalent log/asinh expression to avoid exponential overflow.
The derivative is analytic: q_x=-q*(1-q)/(w*(1+q)), and U_t=-s*U_x.
Substitution gives spatially constant `F(U)-F_visc(U,U_x)-s*U`, which is checked
numerically to about 1e-14. No manufactured forcing or artificial viscosity is
added. The test also checks the exact derivative and weak PDE balance independently.

Default benchmark: mu=1e-3, s=0.5, x0=0.4, t_final=0.05. The front moves 0.025;
its density 10--90% thickness is approximately 0.0050264. All three endpoint
states are prescribed from the exact solution at every RK stage. These are
open-flow verification boundaries, replacing the closed, insulated shock-tube
walls; the total mass and energy are not constant on this domain. The report
therefore gives integral errors relative to the exact solution, not closed-tube
conservation drift.

Both methods use the same constrained L2 initialization, exact endpoint values,
RK4/CFL 0.35, and physical fluxes. The weak equation uses interior test functions;
a full mass solve followed by the rank-two boundary lift enforces U_t at the
endpoints. The omitted endpoint flux loads cancel under that constraint.
Enrichment uses the existing gradient detector on computed density/pressure,
one detected front, quintic splines at h/8, patches of halfwidth 2h, and retained
modes, updating every 0.005. The detector never receives the exact front position.
Errors are sampled at 4001 common points. Timings from this study are single
runs of the verification driver, not a matched performance benchmark.

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_traveling_shock.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/bench_traveling_shock.py --ordinary 385 513 --enriched 193 \
  --prefix benchmarks/traveling_shock/convergence_fine
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/validate_traveling_shock.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python bench/validate_traveling_shock_fine.py
```

The convergence study gives the following representative final errors. Ordinary
uses q32; the enriched entries below use q64 through n=97 and q128 thereafter.
This extra quadrature is necessary to distinguish spatial error from integration
error in the enriched mass/stiffness blocks.

| Method | Base / final modes | Density Linf | Density L2 | Pressure Linf |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 65 / 65 | 2.716e-1 | 5.276e-2 | 4.421e-1 |
| Ordinary | 129 / 129 | 1.389e-1 | 1.836e-2 | 2.282e-1 |
| Ordinary | 257 / 257 | 3.966e-2 | 3.794e-3 | 6.626e-2 |
| Ordinary | 513 / 513 | 3.952e-3 | 3.117e-4 | 6.750e-3 |
| Enriched | 33 / 73 | 5.576e-2 | 5.894e-3 | 9.104e-2 |
| Enriched | 49 / 92 | 1.183e-2 | 1.335e-3 | 2.275e-2 |
| Enriched | 65 / 110 | 4.836e-3 | 5.473e-4 | 8.617e-3 |
| Enriched | 97 / 149 | 3.133e-4 | 5.204e-5 | 3.259e-4 |
| Enriched | 129 / 186 | 1.274e-4 | 1.222e-5 | 2.234e-4 |
| Enriched | 193 / 262 | 1.039e-5 | 2.240e-6 | 1.637e-5 |

The enriched density L2 error decreases with observed slopes 5.80, 5.04 and
4.18 over base refinements 65->97->129->193, using h=1/(n-1). These are empirical
rates for adaptive spaces, not a proof of a fixed convergence order. Peak-error
slopes fluctuate more. Ordinary BSPF also converges, and its density L2 slope
increases to 4.20 over 385->513; the low-resolution results reflect an unresolved
thin layer, not a claim that the underlying BSPF scheme has low formal order.
At 513 ordinary modes the initial density projection error is 4.10e-3 and the
final error 3.95e-3, illustrating the spatial-resolution limitation.

The exact reference exposes a quadrature problem hidden by coarse comparisons:
q32 enriched density errors at n=97,129,193 are 3.95e-4, 4.62e-4, 1.68e-4.
At n=193, q64 gives 1.067e-5 and q128 gives 1.039e-5. All four peak errors change
by less than 3% on q64->q128 at n=129 and n=193; this is a resolution check, not
machine-precision quadrature convergence. Halving CFL at n=129/q32 changes all
primitive profiles by less than 4.0e-10. Ordinary q32->q64 differences at n=65
and n=257 are below 4.9e-10; halving ordinary CFL at n=257 changes profiles by
less than 3.0e-9. Quadrature error, rather than time error, caused the earlier
nonmonotone enriched density trend.

Doubling the enrichment patch halfwidth at n=129/q64 increases final modes from
186 to 218. It reduces density L2 from 1.239e-5 to 8.973e-6, but changes the peak
only from 1.305e-4 to 1.284e-4. It does not explain the q32 error plateau.
Ringing is reduced, not eliminated: the finest enriched density overshoot is
1.039e-5. This benchmark does not claim monotonicity or a speed advantage for
all tolerances. It also uses a different viscosity and Prandtl number from the
previous Sod experiments, so their errors/timings are not interchangeable.

`bench/plot_traveling_shock.py` produces `benchmarks/traveling_shock/convergence.png`
and `.pdf`, plus `summary.json` with all four fields, initial/final norms,
overshoot, mode counts, and observed density rates. Raw profiles and quadrature
checks are retained in the same directory. All figures use the analytic profile
as the reference; there is no fine-grid numerical reference in this study.

### Extending the traveling-shock study to high accuracy

The high-accuracy branch keeps the same source-free Becker shock, mu=1e-3,
Pr=3/4, speed 0.5 and final time 0.05. Merely increasing N with the old settings
is insufficient: the fixed endpoint basis loses numerical accuracy and retained
enrichment transformations can become extremely ill-conditioned.

`BSPFNavierStokes(..., line_options={...})` now permits explicit endpoint basis
settings while preserving the old defaults. The large ordinary run uses N=2305
and 137 endpoint spline basis functions, whose knots align with the sample grid.
The fit amplification is about 1.54e4, versus 2.07e7 for N=1537 with the old
18-function endpoint basis. These endpoint functions are internal to the N-node
BSPF cardinal construction, not extra solution degrees of freedom.

The higher-order enrichment branch uses degree-13 splines, spacing h/4 and
patch halfwidth 12h. Centers still come only from the computed density/pressure
sensor. Selected centers are retained, but `recondition=True` rebuilds the
orthogonalized directions at each expansion and projects the old numerical
solution into the new space with the prescribed endpoint values. Reports retain
the remapping errors and maximum transformation coefficients. This is a new
h/p-refinement branch, not a continuation of the fixed quintic trial spaces.

The un-reconditioned N=769 degree-13 run started near 1e-8 but ended with pressure
error 7.61e-3. Its transformation reached 4.12e13 and raw evaluation coefficients
8.21e10, producing catastrophic cancellation. Those failed results are preserved
as `high_resolution/enriched769_p13.{json,npz}` and are not counted as convergence.
Lower Gram cutoffs and SVD trials did not improve accuracy; those exploratory
options were removed, and the established rank cutoff remains unchanged.

The native quadrature kernels now store endpoint spline samples sparsely,
retaining every nonzero entry. This avoids dense work on known zeros as the
endpoint basis grows. Both ordinary and enriched solvers use the same kernels.
The C++ and Python operator tests compare against dense cardinal operators.

High-accuracy driver (run from the repository root):

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python bench/bench_traveling_shock_high.py --n 2305 --basis 137 --quadrature 12 \
  --prefix benchmarks/traveling_shock/high_resolution/ordinary2305
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python bench/bench_traveling_shock_high.py --n 769 --basis 57 --enriched --recondition \
  --prefix benchmarks/traveling_shock/high_resolution/enriched769_p13_reconditioned
```

The driver saves coefficients, centers, and transformations for independent
resampling. `bench/check_traveling_shock_high.py PREFIX --quadrature Q` checks
64,001 output points, reproduces the saved output, compares one RK4 step against
two half steps, and repeats that step with higher quadrature while holding the
trial functions fixed. These last checks are local checks at the final state;
they are not full time-step-halved or quadrature-refined re-integrations.
Runs overlapped in execution, so their wall times are not a matched speed study.

Both completed integrations reach peak absolute error below 1e-8 in every
primitive field, checked independently on 64,001 points at t=0.05:

| Method | Final coefficients per field | Density | Velocity | Pressure | Temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary BSPF | 2305 | 9.67e-10 | 6.47e-10 | 1.66e-9 | 3.26e-10 |
| Degree-13, reconditioned enrichment | 769 + 138 = 907 | 2.92e-9 | 5.66e-9 | 6.20e-9 | 3.42e-9 |

The preceding enriched run (513 base, 633 total coefficients) has maximum
primitive error 5.97e-8. Increasing to 907 coefficients reduces this to 6.20e-9.
The enriched target run uses 2.54 times fewer coefficients than the ordinary
target run, but this is not a measured speedup or an equal-error comparison.
No artificial viscosity or forcing was added. Reconditioning is an L2 remap,
not an exact coefficient transfer: its largest sampled conservative-field
change is 8.98e-9; the existing numerical-dependence cutoff remains active.

Local step-halving differences in conservative variables are 9.41e-14
(ordinary) and 1.51e-12 (enriched). Local quadrature doubling, q12->24 and
q64->128 respectively, gives differences 4.05e-13 and 3.20e-11. These support
the final-state checks but do not bound accumulated time or quadrature error.

Run `python bench/plot_traveling_shock_high.py` to regenerate
`benchmarks/traveling_shock/high_resolution/convergence_1e8.png` (also PDF)
and `summary_1e8.json`. The figure separates the earlier quintic study from
the new degree-13 branch; endpoint conditioning also changes in the finest
ordinary run, so the connected curves should not be read as fixed-parameter
asymptotic convergence orders.

### Six-resolution convergence check

`bench/run_traveling_shock_convergence.py` extends the degree-13 branch with base
N=193,257,385,513,641,769. Every enriched run uses q64, CFL=0.7, degree 13,
spacing h/4, halfwidth 12h, reconditioning, and the same computed-state sensor
and adaptation interval (0.005). The endpoint count follows (N-1)/16+9, retaining
aligned knots while controlling fit conditioning. The physical problem and
final time are unchanged. Completed compatible runs are reused.

```bash
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/run_traveling_shock_convergence.py sweep
python bench/run_traveling_shock_convergence.py ordinary
python bench/run_traveling_shock_convergence.py validation
MPLCONFIGDIR=/tmp/bspf-mpl python bench/analyze_traveling_shock_convergence.py
```

The ordinary comparison uses the same endpoint-count rule at
N=193,257,385,513,641,769,1025,1537 with q12 and CFL=0.7. Its final N=2305 point
reuses the earlier 137-endpoint-function run; that point uses 128 aligned knot
intervals. Full-evolution sensitivity runs are performed for the enriched
N=513 and N=769 cases: CFL=0.35 at q64, and q128 at CFL=0.7. These rerun
initialization, all time steps, and all adaptations, unlike the earlier local
one-step checks. The ordinary comparison retains the earlier local checks at
its finest resolution; it has not received equivalent full-evolution checks.

The analyzer evaluates every final profile on the same 64,001-point grid,
computes absolute L2 and sampled Linf errors against the analytic shock,
calculates adjacent observed rates using h=1/(N-1), and compares full-rerun
profiles with their baselines. It writes `full_convergence.json`,
`full_convergence.txt`, and `full_convergence.png`/`.pdf` under
`benchmarks/traveling_shock/high_resolution/`. The JSON retains individual-field
errors, rates, adaptation metadata and sensitivity differences. Adaptation and
remapping sensitivity are included in the full-rerun differences, so those
values should not be interpreted as isolated truncation errors.

Dense-grid results for the enriched branch:

| Base N | Total modes | Maximum primitive Linf error | Density L2 error | Density L2 rate |
| ---: | ---: | ---: | ---: | ---: |
| 193 | 292 | 3.919e-04 | 3.690e-05 | - |
| 257 | 360 | 9.585e-05 | 4.896e-06 | 7.02 |
| 385 | 497 | 1.677e-06 | 6.343e-08 | 10.72 |
| 513 | 633 | 5.976e-08 | 2.866e-09 | 10.77 |
| 641 | 770 | 1.584e-08 | 9.160e-10 | 5.11 |
| 769 | 907 | 6.199e-09 | 2.663e-10 | 6.78 |

All four field errors decrease at every resolution in this sweep. The rate
variation reflects the adaptive approximation sequence; it is not a claim
of a fixed formal order. Reconditioning errors are nonzero and recorded:
the maximum sampled conservative remap error in the finest baseline is
8.98e-9, so basis transfer remains relevant at this accuracy.

The full-evolution checks confirm the enriched 1e-8 target on the dense grid:

| Base N | Check | Maximum primitive Linf error | Maximum profile change from baseline |
| ---: | --- | ---: | ---: |
| 513 | CFL 0.7 -> 0.35 | 5.961e-8 | 1.199e-9 |
| 513 | q64 -> q128 | 5.947e-8 | 2.092e-9 |
| 769 | CFL 0.7 -> 0.35 | 6.215e-9 | 3.293e-10 |
| 769 | q64 -> q128 | 6.132e-9 | 8.937e-10 |

Both finest checks remain below 1e-8 in all four fields. The differences are
small compared with the target, but are not zero and do not establish accuracy
at 1e-10. All enriched sweep and sensitivity runs completed without rejected
stages. The ordinary nine-point comparison also decreases in all four fields,
reaching a worst-field error of 1.661e-9 at 2305 modes. The enriched baseline
reaches 6.199e-9 with 907 final coefficients per field. These mode counts compare
target attainment, not equal errors or measured runtime speedups.

### Classical MUSCL finite-volume comparison

`examples/pde/muscl_traveling_shock.py` implements a uniform-grid, second-order
finite-volume comparator: primitive-variable MUSCL reconstruction with the
monotonized-central (MC) limiter, HLLC inviscid flux, centered physical viscous
flux, and SSPRK2 time stepping. MC and HLLC are conventional selectable
[MUSCL limiter/flux options](https://mflowcode.github.io/documentation/case.html).
This is a method-of-lines MUSCL variant, not a MUSCL-Hancock predictor.
Its HLLC flux supplies inherent numerical dissipation; no separate artificial
viscosity is added. Numba compiles the integration loop.

The physical problem matches BSPF: gamma=1.4, mu=0.001, Pr=0.75, shock speed
0.5, [0,1], and t=0.05. Initialization uses 16-point Gaussian conservative
cell averages. Constant ghost states equal the analytic far-field states to
floating-point precision throughout this test; both boundaries are hundreds
of shock widths from the moving front. Convective and physical-diffusion
restrictions enter the explicit time step with CFL=0.4.

Pointwise errors use the method's limited piecewise-linear primitive
reconstruction on the same 64,001-point grid as BSPF. Conservative cell-average
errors are also recorded against Gaussian averages of the analytic solution;
cell averages are not treated as exact point values. The uniform-state,
contact-flux, global flux-balance, initial quadrature, and refinement checks
are in `tests/python/test_muscl_traveling_shock.py`.

```bash
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/bench_muscl_traveling_shock.py
# Run this separately, without concurrent benchmark jobs:
python bench/compare_muscl_timing.py
MPLCONFIGDIR=/tmp/bspf-mpl python bench/plot_muscl_comparison.py
```

The sweep uses 256,512,1024,2048,4096,8192 cells plus a full 4096-cell CFL=0.2
rerun. The separate timing comparison warms JIT/native code, then measures
serial end-to-end solves including initialization, integration, and common
dense error evaluation. It compares configurations at a 5e-4 peak-error
threshold, not exactly equal errors. These are single implementation-specific
wall measurements, not a general algorithmic speedup or 3-D scaling result.
Outputs are in `benchmarks/traveling_shock/muscl/`: `comparison.png`/`.pdf`,
`comparison.json`, `comparison.txt`, raw profiles, and `serial_timing.json`.

Measured MUSCL convergence (worst peak error among rho,u,p,T):

| Cells | Worst peak error | Density cell-average peak error |
| ---: | ---: | ---: |
| 256 | 1.916e-1 | 7.087e-2 |
| 512 | 1.382e-1 | 2.738e-2 |
| 1024 | 2.720e-2 | 8.812e-3 |
| 2048 | 6.743e-3 | 1.979e-3 |
| 4096 | 1.390e-3 | 4.106e-4 |
| 8192 | 3.311e-4 | 8.904e-5 |

The final pressure peak-error rate is 2.07. Halving CFL at 4096 cells changes
the reconstructed fields by at most 3.125e-7, versus a baseline worst error
of 1.390e-3. This supports spatial error dominating at that resolution.

Fresh serial measurements at a common 5e-4 worst-field threshold:

| Method | Unknowns per field | Worst peak error | End-to-end seconds |
| --- | ---: | ---: | ---: |
| MUSCL-MC/HLLC | 8192 | 3.311e-4 | 114.12 |
| Ordinary BSPF | 897 | 2.481e-4 | 39.76 |
| Enriched BSPF | 193 + 99 = 292 | 3.919e-4 | 15.48 |

For this threshold and these implementations, enrichment uses about 28 times
fewer coefficients and 7.4 times less wall time than MUSCL. Errors are not
identical. These results concern an analytic, smooth viscous shock and do not
establish performance for discontinuous Euler shocks or general 3-D flows.
The MUSCL sweep does not reach 1e-8; no runtime or cell count at that tolerance
has been measured. BSPF's separately verified finest results remain
1.661e-9 (ordinary, 2305 modes) and 6.199e-9 (enriched, 907 modes).

### Characteristic WENO-Z5 comparison

`examples/pde/wenoz_traveling_shock.py` implements conservative finite-difference
WENO-Z5 with nodal conservative unknowns. At each face it uses a Roe-averaged
characteristic basis, global Lax-Friedrichs flux splitting, and WENO-Z weights
`d_k * (1 + (tau5/(beta_k + epsilon))**2)` with epsilon=1e-40 and ideal weights
(0.1,0.6,0.3). See the [WENO-Z paper](https://scholars.hkbu.edu.hk/en/publications/an-improved-weighted-essentially-non-oscillatory-scheme-for-hyper-2/)
and the [NASA discussion of WENO-Z parameters](https://ntrs.nasa.gov/api/citations/20080043618/downloads/20080043618.pdf).
This comparator uses characteristic reconstruction rather than separately
weighting each conservative component. Lax-Friedrichs splitting contributes
numerical dissipation; no separate artificial-viscosity term is added.

Both the velocity/temperature gradients and divergence of the physical viscous
flux use sixth-order centered differences. Classical RK4 advances the solution;
CFL=0.7 is applied to a combined advective/diffusive bound. Thus the second-order
viscous discretization from the MUSCL comparator does not limit this scheme.
The same analytic traveling shock, mu=0.001, Pr=0.75 and t=0.05 are used.
Initial data are nodal values, and prescribed constant endpoint/ghost states
agree with the analytic far field to floating-point precision in this problem.

The sweep uses N=257,513,1025,2049,4097,8193, plus full CFL-halving runs at
N=2049 and N=4097. Dense output uses a degree-five spline (sixth-order accuracy
for smooth data) purely for diagnostics; it does not enter evolution. Reports
include nodal norms and the interpolation error of exact sampled profiles to
separate this diagnostic from evolution error. Dense norms use the same 64,001
points as the BSPF comparison. Dedicated tests verify uniform preservation,
fifth-order contact-advection derivatives, sixth-order physical diffusion, and
open-domain conservative flux balance.

```bash
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/bench_wenoz_traveling_shock.py
# Run separately, without concurrent benchmark jobs:
python bench/compare_wenoz_timing.py
MPLCONFIGDIR=/tmp/bspf-mpl python bench/plot_wenoz_comparison.py
```

Results, coefficient/profile arrays, full-step checks and plots are written to
`benchmarks/traveling_shock/wenoz/`. Timing compares end-to-end warm solves at
a common 5e-4 peak-error threshold, with JIT compilation excluded and setup,
integration, and dense error evaluation included. These are single serial
implementation-specific measurements, not an exactly equal-error comparison.

Measured WENO-Z5 convergence:

| Nodes | Worst dense peak error | Worst nodal error | Pressure peak-error rate |
| ---: | ---: | ---: | ---: |
| 257 | 3.160e-1 | 2.373e-1 | - |
| 513 | 1.015e-1 | 7.869e-2 | 1.64 |
| 1025 | 1.607e-2 | 1.573e-2 | 2.66 |
| 2049 | 6.399e-4 | 6.204e-4 | 4.65 |
| 4097 | 2.613e-5 | 2.575e-5 | 4.61 |
| 8193 | 8.558e-7 | 8.482e-7 | 4.93 |

Full CFL-halving runs change the primitive profiles by at most 1.500e-10 at
2049 nodes and 3.388e-12 at 4097 nodes. The exact-profile diagnostic
interpolation error at 8193 nodes is 2.086e-10, well below the numerical error.
The derivative tests independently recover fifth-order convection and
sixth-order diffusion; the shock sequence approaches fifth-order convergence
once the physical layer is resolved.

Fresh serial end-to-end timings at the common 5e-4 threshold:

| Method | Unknowns per field | Worst peak error | Seconds |
| --- | ---: | ---: | ---: |
| WENO-Z5 | 2305 | 3.872e-4 | 5.36 |
| Ordinary BSPF | 897 | 2.481e-4 | 41.80 |
| Enriched BSPF | 193 + 99 = 292 | 3.919e-4 | 15.56 |

At this moderate tolerance, WENO-Z5 is about 2.9 times faster than enriched
BSPF for nearly equal errors, while enrichment uses about 7.9 times fewer
coefficients. Thus coefficient efficiency does not imply a runtime advantage.
The WENO-Z5 sweep has not reached 1e-8; the separately verified BSPF results
remain 1.661e-9 with 2305 ordinary modes and 6.199e-9 with 907 enriched modes.
No equal-error runtime comparison at 1e-8 or general 3-D speedup is claimed.

### Serial runtime at a 1e-6 error threshold

`bench/compare_wenoz_timing_1e6.py` repeats the comparison at a tighter sampled
peak primitive-error threshold. The physical problem, final time, 64,001-point
error evaluation and warm end-to-end timing definition are unchanged. Jobs run
serially with one BLAS thread; JIT warmup and result-file writes are excluded.

| Method | Unknowns per field | Worst primitive peak error | End-to-end seconds |
| --- | ---: | ---: | ---: |
| WENO-Z5 | 8193 | 8.558e-7 | 146.01 |
| Ordinary BSPF | 1601 | 6.256e-7 | 197.34 |
| Enriched BSPF | 417 + 113 = 530 | 7.101e-7 | 83.72 |

Enriched BSPF takes 1.74 times less wall time than WENO-Z5 and uses 15.46 times
fewer coefficients for these passing configurations. Ordinary BSPF takes
1.35 times the WENO runtime here. These are single fresh measurements, not
repeated-trial statistics; use this table's times together rather than mixing
in the earlier sweep timings. Errors are comparable but not identical, and
the tested resolutions are not claimed to be globally optimal. The previous
385-base enriched and 1537-node ordinary cases exceeded 1e-6, motivating the
slightly higher trial resolutions. Both passed on the first new candidate.

Run from the repository root:

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python bench/compare_wenoz_timing_1e6.py
```

Full reports and profiles are saved under `benchmarks/traveling_shock/wenoz/`,
with summary `serial_timing_1e6.json` and the measured table in
`serial_timing_1e6.txt`.

### Modal discontinuous Galerkin comparison

`examples/pde/dg_traveling_shock.py` implements overintegrated modal Legendre DG
with HLLC inviscid interface fluxes, central BR1 traces for auxiliary velocity
and temperature gradients and physical viscous fluxes, and classical RK4.
The BR1 approach originates with [Bassi and Rebay](https://www.sciencedirect.com/science/article/pii/S0021999196955722).
This is a conventional modal DG formulation, not the entropy-stable split-form
DGSEM variant. No modal filter, artificial viscosity, or subcell blending is
used. All elements have the same width; there is no local mesh refinement.

Degrees p=3,5,7 use p+1 Legendre coefficients per conserved variable per element
and 2p+3 Gauss points for volume terms. The diagonal mass matrix is exact.
Auxiliary gradients are computed by the weak BR1 equations, not by simply
differentiating each discontinuous element independently. Boundary traces use
the same constant analytic far-field states as the previous comparators, with
zero exterior viscous flux. These match this benchmark's endpoints to floating
point precision. Initial coefficients are L2 projections of the analytic shock.

The explicit time step combines a convective and a diffusive bound using
sampled Fourier-symbol spectral radii of the degree-specific scalar upwind DG
and central BR1 operators, scaled to the element width and physical wave and
diffusion speeds. The safety factor is 0.7. This bound is not claimed to be the
largest possible stable time step for the nonlinear problem.

Errors are measured on the common 64,001-point grid and on both traces at every
element interface. The reported worst error includes both checks, so a jump
cannot be hidden by selecting only one side at an interface. Timings exclude
JIT warmup, include one operator construction, initialization, integration,
and error evaluation, and count all E*(p+1) coefficients per field. DG's
auxiliary gradient arrays are temporary, not additional evolved unknowns.

Reproduction commands:

```bash
export PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/bench_dg_traveling_shock.py
python bench/check_dg_traveling_shock.py time --elements 576
python bench/check_dg_traveling_shock.py quadrature --elements 576
MPLCONFIGDIR=/tmp/bspf-mpl python bench/plot_dg_comparison.py
python bench/summarize_dg_comparison.py
```

The sweep saves and reuses completed runs. It tests degrees 3 and 5 through
1024 elements and degree 7 through a finer sequence near the threshold. A
1024-element degree-7 run was stopped after the 512-element result narrowly
missed the tolerance; its unfinished state is not used. The subsequent
576-element run passes. Degrees 3 and 5 did not reach 1e-6 within their tested
resolution ranges. This is not a claim that they cannot reach that accuracy.

The passing case uses 576 elements at p=7 (4608 coefficients per field), with
sampled-grid error 5.357e-7 and two-sided interface peak error 8.216e-7. Its
serial warm end-to-end time is 144.92 seconds. Reference runs from the preceding
serial benchmark took 146.01 seconds for WENO-Z5 (8193 unknowns, error 8.558e-7),
197.34 seconds for ordinary BSPF (1601 unknowns, error 6.256e-7), and 83.72
seconds for enriched BSPF (530 unknowns, error 7.101e-7). The roughly 1% timing
difference between DG and WENO is not meaningful evidence of a speed advantage
from single measurements. Enriched BSPF is about 1.73 times faster than this DG
implementation, with 8.69 times fewer coefficients at the common threshold.

Full sensitivity reruns halve the CFL and double the volume quadrature at the
passing DG resolution. Their wall times are excluded from performance claims
because those checks ran concurrently. Results and plots are saved under
`benchmarks/traveling_shock/dg/`; `threshold_comparison.json`/`.txt` preserve the
selected case, the earlier reference timings, and both full-evolution checks.

The full DG sensitivity runs confirm the threshold: halving the time step
changes any primitive value (including both interface traces) by at most
9.943e-12, and doubling quadrature changes it by at most 2.586e-12. Their worst
errors remain 8.2163e-7 and 8.2162e-7 respectively. Thus neither temporal nor
volume-quadrature error explains the measured 1e-6-level result. The combined
Python regression suite passes 96 tests, including five new DG checks.

## Graded spline BSPF versus FC-Gram at 1e-8

The [graded spline experiment](../docs/research/graded_spline_fc_1e8.md) uses a
spline step and compact graded degree-13 B-splines to carry the viscous layer,
leaving only 25 fixed BSPF base coefficients plus 45 enrichment coefficients.
Five alternating warm runs give 1.048 seconds median and 2.93e-9 peak error,
versus optimized FC-Gram's 1.758 seconds and 5.48e-9 error. Setup, adaptation and
dense error evaluation are included; one-time JIT preparation is excluded.

```sh
python bench/fetch_fc_data.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/bench_graded_spline_fc.py --repeats 5
MPLCONFIGDIR=/tmp/bspf-mpl python bench/plot_graded_spline_fc.py
```

The benchmark saves repeated timings, time-step/quadrature controls, nearby
passing/failing resolutions, execution-path ablations and profiles under
`benchmarks/traveling_shock/graded_spline_fc_1e8/`. The focused regression suite
now passes 178 tests. Small dense operator compilation is a 1D optimization;
the result does not establish 3D scalability or performance on interacting fronts.

## Fixed-dictionary enrichment without a front detector

The [dictionary experiment](../docs/research/dictionary_spline_fc_1e8.md)
selects a spline block from 1295 fixed position/scale choices using a
state-approximation residual. Seven alternating warm comparisons give
1.359 seconds median and 1.12e-9 error with 70 coefficients, versus a retuned
FC-Gram baseline at 2.171 seconds and 6.68e-9 error with 2120 physical samples.
Setup, selection, integration and dense error assessment are included.
Time-step, quadrature, shifted-profile and reversed-motion controls pass.

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 bench/compare_dictionary_fc.py --repeats 7 --fc-n 2120 --interval .0025 --quadrature 7 --outer-quadrature 6 --output benchmarks/traveling_shock/dictionary_fc_final
MPLCONFIGDIR=/tmp/bspf-mpl python3 bench/plot_dictionary_fc.py
```

This is a one-block 1D prototype, not a certified PDE-residual estimator or a
demonstrated 3D directional-frame solver. The focused regression suite passes
188 tests. The report records failed configurations and the remaining 3D work.

The subsequent [viscous shock-tube test](../docs/research/dictionary_shock_tube.md)
extends selection to multiple blocks and uses the numerical state plus its
physical NS rate. With a 129-point background and up to four blocks it reaches
5.07e-7 discrepancy against a fine FC reference, using at most 361 coefficients
per field. It takes 273.71 seconds; FC-Gram reaches 5.51e-10 in 15.62 seconds.
The single-shock speed advantage therefore does not transfer to this prototype.
The report includes the failed smaller configuration, conservation diagnostics,
cost breakdown and a common-grid comparison plot. The regression suite now
passes 192 tests.

## Laval nozzle on a mapped block (C++ CompressibleSolver2D, 2026-11-10)

First step towards a mapped multi-block compressible solver. The numerics are
`bspf.CompressibleSolver2D` (`include/bspf/compressible.hpp`); the Python
scripts only build the geometry, boundary conditions and initial state, call
`solve`, and post-process. Sextic wall (`--shape sextic --length 4`, the
default): area ratio 1.5, throat curvature radius 5.3, inlet M = 0.43,
quasi-1D exit M = 1.85. Results in `bench/results/nozzle/`.

How the method got here (each step measured by linearized-operator eigenvalues):
* collocation BSPF with strong boundary values has boundary-localized growing
  modes (even 1D acoustics between walls), so it blows up;
* the Galerkin derivative D = M^-1 C is summation-by-parts and as accurate, but
  with nodal fluxes and the dense norm M the variable-coefficient system keeps
  growing modes whose rate grows with N (1D linearized Euler reproduces it);
* the Galerkin weak form, with fluxes formed at Gauss points and weak face
  fluxes, removes them in 1D and nearly in 2D (6-10 weak modes left); a
  Legendre-basis spectral vanishing viscosity, spectrally small on resolved
  fields, makes the 2D operator stable;
* a free-stream correction (the quadrature's metric-identity defect times the
  nodal fluxes) keeps uniform flow exact; 5 Gauss points per cell are needed
  for full accuracy (4 leaves an error floor near 4e-8).

Manufactured Navier–Stokes solution (Re = 100), relative steady-state error:

| Gauss points per cell | 65×23 | 97×33 |
|---|---|---|
| 4 | 7.0e-8 | 4.2e-8 |
| 5 (default) | 1.4e-8 | 1.5e-9 |
| 6 | 1.4e-8 | 1.5e-9 |

Euler nozzle, 97×33 (exactly isentropic flow): Cd = 0.99924113, max entropy
error 2.3e-6, max enthalpy error 1.6e-6, mass-flow variation between sections
2.0e-9.

Steady-state cost, 97×33 Euler on one Ryzen 5800X (16 threads available):

| | steps | time |
|---|---|---|
| Python prototype, plain RK4 | ~35 000 | ~90 min (default SVV diverged) |
| C++, plain RK4 (measured to step 6000, residual 6e-5) | ~35 000 | ~20 min (estimated) |
| C++, Anderson depth 8 | 6 500 | 4 min |
| C++, Anderson + grid sequencing from 49×17, 1 thread | 2 600 | 88 s |
| same, `OMP_NUM_THREADS=8` | 2 600 | 65 s (fine stage 46 s) |

Navier–Stokes, Re = 2000, 97×49 with wall clustering 8 (sequenced from 49×25,
8 threads, Cd = 0.95383 in both columns):

| | explicit viscous | line-implicit wall-normal viscous |
|---|---|---|
| right-hand side | 16.7 ms | 6.3 ms (viscous divergence as nodal tensor products) |
| one step | 68.5 ms | 26.2 ms (incl. per-line LU) |
| steps, 49×25 + 97×49 | 5 100 + 8 250 | 4 250 + 5 450 |
| time | 655 s | 216 s |

The implicit wall-normal viscous terms raise the near-wall time step 1.4×
(49×25) to 1.8× (97×49); after that the near-wall step equals the Euler one, so
the acoustic wall-normal limit binds. Making the wall-normal inviscid terms
line-implicit too is the next step for clustered viscous grids.

One right-hand side at 97×33 costs 9.1 ms on one thread and 3.2 ms on eight;
it is bound by the dense quadrature-point GEMMs (BSPF trial functions are
global along each line), about 160 MFLOP per evaluation. More than 8 threads
is slower, and small grids (49×17) gain nothing from threads.

    OMP_NUM_THREADS=8 PYTHONPATH=python python examples/pde/laval_nozzle_2d.py \
        --n 97 33 --sequence 49 17
    OMP_NUM_THREADS=8 PYTHONPATH=python python examples/pde/laval_nozzle_2d.py \
        --n 97 49 --sequence 49 25 --re 2000 --cluster 8 --cfl 0.45 --tol 1e-9

### FP32 GPU solver (`bspf.cuda.CompressibleSolver2D`, 2026-11-10)

Built from a CPU solver (`bspf.cuda.CompressibleSolver2D(cpu_solver, refresh=10)`),
Euler flows only so far. Every step runs on the device: the BSPF trial field is
evaluated at the Gauss points and projected back with GalerkinLine's fast
structure (banded spline + batched cuFFT, `bspf.cuda.GalerkinAxis`), fluxes and
face states in fused kernels, mass solve and SVV with cuBLAS, and the
line-implicit preconditioner along eta lines as batched cuBLAS LU (refactored
every `refresh` steps). FP32 needs a small fit amplification, hence n_basis ~ N/3
(at least 24) per axis.

Euler nozzle on an RTX 3070, line-implicit with `refresh=10`, no Anderson:

| grid | explicit | line-implicit |
|---|---|---|
| 97×33, to residual 1e-4 | 6 050 steps, 18.6 s (3.1 ms/step) | 1 150 steps, 4.5 s |
| 193×65, to residual 2e-4 | not reached in 8 000 steps | 2 950 steps, 27.5 s; Cd 0.999237, entropy error 8e-5 |

Stall check on 193×65 (2026-11-10). With SVV 0.1 and Anderson the residual
levelled off at 2-3e-4 from step 5 000 on. It was not the FP32 floor: the exact
(double) residual of that state was 3.2e-4 with the device right-hand side
accurate to 1e-5 in the interior. Without Anderson the same run diverged (the
CPU solver too, so not an FP32 effect): left preconditioning pushed the weakly
damped modes of the operator (stability margin ~0 at SVV 0.1) to growth, and
Anderson only masked it. SVV 0.2, now the default, makes the iteration stable
and costs no accuracy (Euler MMS 97×33: 6.3e-10 vs 9.2e-10). The remaining
plateau near 1e-4 is the FP32 floor: there the double residual is 9.8e-5
overall but 1.7e-5 in the interior, and the device/double difference is 7e-5,
concentrated at the boundary. Anderson acceleration does not help on the device
(FP32 normal equations): 5 900 instead of 2 950 steps on 193×65 and a Cd off by
1e-4, so use `anderson=0` there. A second gap, found with the manufactured
solution: the preconditioner linearized only wall penalties on the eta faces,
so exterior (fixed-state) faces diverged; their penalty is now included (MMS
65×23: 300 steps, error 1.2e-9, as explicit).

At 97×33 the device steady state matches the double-precision CPU one to 1e-5.
The FP32 floor grows with N (face rounding amplified by the mass inverse at the
corners): residuals stall near 3e-5 on 97×33, and entropy errors are 5e-5 there
and 7e-4 on 193×65 (CPU: 2e-6).

### FAS multigrid with h- and p-coarsening (`bspf.multigrid_solve`, 2026-11-10)

C++ (`include/bspf/multigrid.hpp`), for CPU and GPU levels alike: injected
state, full-weighting residual restriction, BSPF prolongation of the
correction, damping omega = 0.5 (halved automatically if a correction would be
nonphysical), V- or W-cycles. Levels are nested per axis (n or (n + 1) / 2), so
p-levels (same nodes, lower-order trial space: degree 5 or 3 with
finite-difference jets) and semi-coarsening mix with h-levels.
`--multigrid 0` builds the deepest hierarchy, e.g. for 193×65:
193×65 high, 97×33 high, 49×17 high, 49×17 medium, 25×9 medium, 25×9 low,
13×5 low, 7×5 low.

| case | cycles | fine-step work | notes |
|---|---|---|---|
| GPU 97×33, 2 levels / deep, to 1e-4 | 11 / 11 | 442 / 427 | 2.7 s / 4.9 s (small GPU levels are latency bound) |
| GPU 193×65, 3 levels / deep, to 2e-4 | 14 / 13 | 457 / 424 | 6.4 s / 8.7 s; single grid 2 950 steps, 27.5 s |
| CPU double 97×33, 2 levels / deep, to 1e-8 | 70 / 57 | 2 543 / 2 004 | late rate 0.90 / 0.86 per cycle |
| CPU double 193×65, 3 levels / deep | stall at 3e-7 | | Cd 0.99924113 |

Open: the line-implicit smoother's stability margin shrinks with N. 97×33
converges fully in double, 193×65 stalls at 3e-7 (a neutral mode that no
coarse level removes), and 385×129 diverges even on a single grid and at CFL
0.4 (explicit pseudo-time stays stable there). Multigrid cannot cure a
non-damping smoother; the preconditioner (nodal SBP form of the eta operator)
has to match the weak-form residual better, or be used inside a Krylov solver.

### Weak-form line operator, and multigrid without it (2026-11-10)

The line-implicit preconditioner now linearizes the weak-form eta operator
(Gauss-point Jacobians, quadrature mass, numerical face fluxes) instead of the
nodal form; 193×65 then converges on a single grid. It still diverges once the
eta lines get long (193×129, 97×129 after ~800 steps), so the multigrid
smoother defaults to explicit RK4 (`--line-implicit none` with `--multigrid`),
with a smoother-aware h/p hierarchy (`order_valid(..., implicit=False)`).

Explicit-smoothed deep FAS on the GPU (FP32), cycles to the FP32 tolerance:
97×33 16, 193×65 22, 193×129 18, 385×129 21 (fine-step work 550-700): grid
independent, including grids where line-implicit smoothing diverges.
In double precision (CPU) both grids reach ~1e-5 in ~40 cycles with Cd correct
to 1e-7, but the tail is slow (97×33: 8.5e-8 after 200 cycles) or stalls
(193×65: 4-5e-6). The weak-form operator itself has weakly growing modes
without SVV (49×17: 8 eigenvalues with Re > 0, max 0.256), identical with 5 or
8 Gauss points per cell, so not quadrature aliasing: a formulation property
(Galerkin in conservative variables is not energy stable for variable-
coefficient Euler), which pseudo-time iterations cannot converge past.

### Newton-Krylov with FAS preconditioning (`bspf.newton_krylov_solve`, 2026-11-10)

Newton on R(U) = 0 with FGMRES; Jacobian-vector products by directional finite
differences (step ~ sqrt(machine epsilon)), right preconditioner one FAS cycle of
the deep h/p hierarchy (explicit smoothing) applied to a small perturbation
(linearized by the same kind of difference); inexact-Newton tolerances and a
line search. Driver: `--multigrid 0 --newton`. CPU, double precision:

| grid | Newton steps | Krylov iterations | residual | time | Cd | entropy error |
|---|---|---|---|---|---|---|
| 97×33 | 7 | 71 | 4.9e-12 | 53 s | 0.99924113 | 4.9e-5 |
| 193×65 | 7 | 125 | 1.6e-11 | 228 s | 0.99924113 | 7.0e-8 |

Pseudo-time methods stalled on 193×65 (4e-6 explicit-smoothed multigrid, 3e-7
line-implicit); Newton only needs an invertible Jacobian. On the GPU in FP32 it
cannot go below the FP32 floor (~1e-4 at 193×65): the finite-difference
Jacobian products are rounding noise there.

Entropy-variable trial field (`variables="entropy"`): implemented, but the 49×17
Jacobian (SVV off) then has 68 growing modes (max Re 0.455) against 8 (0.256)
in conservative variables. Entropy stability needs the matching A0-weighted
mass and exact integration of U(V_h); pseudo-time with the tensor mass in U
does not inherit it. Off by default.

### FP64 device solver, FP32 multigrid preconditioner (2026-11-10)

The device code is templated on precision (cuBLAS S/D, cuFFT R2C/D2Z):
`bspf.cuda.CompressibleSolver2D(cpu, precision="double")` is the default and
matches the CPU residual to 1e-11 (97×33); `precision="float"` keeps the FP32
path. `bspf.cuda.newton_krylov_solve(fp32_levels, U, residual=fp64_solver)` runs
Newton, FGMRES and the Jacobian products in FP64 and the FAS preconditioner on
FP32 levels (driver: `--multigrid 0 --newton --device cuda`):

| grid | Newton steps | Krylov iterations | residual | GPU mixed | CPU FP64 | Cd | entropy error |
|---|---|---|---|---|---|---|---|
| 97×33 | 7 | 75 | 7.8e-12 | 29 s | 53 s | 0.99924113 | 4.9e-5 |
| 193×65 | 7 | 162 | 4.9e-12 | 90 s | 228 s | 0.99924113 | 7.0e-8 |

FP64 steps on the RTX 3070 cost ~10× FP32 (1/32 FP64 rate): 150 ms vs 16 ms per
line-implicit step on 193×65, so the FP64 solver is used for residuals only.

Device Krylov (`device_krylov=True`, default with an FP64 `residual`): Newton,
FGMRES (CGS2 by cuBLAS gemv), Jacobian products and the line search keep their
vectors on the GPU; only the preconditioner's input and output cross to the
host. Same convergence (193×65: 7 Newton, 164 Krylov), same time (91 s): the
breakdown is 82 s preconditioner, 3.2 s Jacobian products, 0.05 s Krylov
algebra. The host-driven FAS cycle (per-level solve calls, host restriction and
CPU prolongation) is the bottleneck, ~0.5 s per cycle against ~0.2 s of GPU work.

### Reference: PyFR 2.1 on the same nozzle (RTX 3070, 2026-11-10)

`bench/pyfr_nozzle_case.py NX NY ORDER PREFIX --steady` writes the gmsh mesh
(exact degree-6 Bezier walls, curved quads) and the PyFR ini: Euler, Rusanov,
`char-riem-inv` inlet from the quasi-1D state (PyFR's Euler system has no total-
condition inlet, so Cd differs; the entropy invariant is exact in both),
`sup-out-fn`, `slp-adia-wall`; steady by dual time (one backward-Euler step,
dt = 1e8) with local pseudo-time RK45 and p-multigrid 4-3-2-1-0.

Per right-hand side, ~12.7k DOF: PyFR 45 µs (FP64) / 27 µs (FP32); BSPF GPU
18.5 ms (FP64) / 1.67 ms (FP32).

Steady solve to residual ~1e-10/1e-11, L2 entropy error:

| | DOF | time | L2 entropy error |
|---|---|---|---|
| PyFR p4, 16×8 | 3 200 | 10.0 s | 1.1e-6 |
| PyFR p4, 32×16 | 12 800 | 13.1 s | 5.1e-8 |
| PyFR p4, 64×32 | 51 200 | 35.3 s | 3.2e-9 |
| BSPF 97×33 | 3 201 | 17.8 s | 2.0e-6 |
| BSPF 193×65 | 12 545 | 61.7 s | 3.2e-9 |

BSPF converges faster with resolution (×600 per 4× DOF vs ×20) and is 16× more
accurate at equal DOF, but each residual is 60-400× more expensive, so PyFR
reaches 3.2e-9 in about half the time with 4× the DOF.
