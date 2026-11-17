# Single-mesh 65³ diagnostic movies

- [3D diagnostic MP4](single_mesh_65_diagnostic_t30.mp4)
- [Central y=2 diagnostic MP4](single_mesh_65_diagnostic_y2_t30.mp4)
- [Comparison with the Gauss and background-corrected controls](comparison.png)

These show the completed **damping-strength 0.02** experiment, simulation job
64704714: 65³ FP64 evolution, dt=0.02, 1500 steps, t=0 through 30. Each MP4
contains 100 saved frames, encoded at 1280×720, 10 fps, lasting 10 seconds.
The y=2 view evaluates the full BSPF interpolant on a 511×511 display grid;
this is visualization sampling, not added simulation resolution. The 3D view
uses nodal isosurface sampling and vertical exaggeration 4. No temporal
interpolation or smoothing is applied.

The evolving fields, nonlinear products, coefficient/nodal sensor and adaptive
numerical diffusion use one 65³ solution mesh. Fixed linear operators and the
sharp background-stratification coupling are preintegrated. Quadrature is
used for setup and independent audits. The artificial stiffness is positive;
its background term uses derivatives of the represented nodal background.
Periodic y and the prior molecular total-buoyancy natural bed condition remain.

This is an experimental diagnostic, **not an accepted replacement**. At t=30,
perturbation buoyancy variance is 6.53% above the original Gauss control;
kinetic energy differs by -0.88%. Final maximum audited buoyancy bound error
is 0.420% of the buoyancy range, and the largest sampled violation during the
run is 1.73%. Median step time is 0.378 s; median velocity/scalar iterations
are 10/9, with maxima 27/14. Coupled and scalar residuals stay below 1e-10.

For context, correcting only the background-stratification load while keeping
the previous fine sensor and quadrature damping reduces the earlier +28.4%
final variance discrepancy to +0.20%. The single-mesh experiment changes the
numerical damping model as well. The unchanged nodal momentum discretization
has not passed its earlier smooth-load accuracy gate. Neither these movies nor
the quadrature audits establish spatial convergence of wave breaking.

The stronger 0.1-damping trial is a separate experiment and is not shown here.

Render job: 64705105. SHA-256 checksums:

- `single_mesh_65_diagnostic_t30.mp4`: `91b2a52aa7f41bca3421b1f8e2b743319ee06e21ff954652fa49a453cb459f35`

- `single_mesh_65_diagnostic_y2_t30.mp4`: `8ab5dafc1dd60f6ff99aef5182b21ee2f75013e2503a1644064966cb29f03342`
