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
