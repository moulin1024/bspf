# Stabilized 127³ internal-wave movies

- [Central y=2 cross-section MP4](internal_wave_stabilized_y2_t30.mp4)
- [3D buoyancy-isosurface MP4](internal_wave_stabilized_t30.mp4)

Both movies contain 100 saved snapshots covering t=0 through t=30, encoded
as 1280×720 H.264 at 10 fps (10 seconds). The central section evaluates the
full BSPF interpolant on a 511×511 display grid; this adds no simulation
resolution or temporal interpolation. The 3D view uses vertical exaggeration 4.

This periodic-y, 127³ FP64 case uses dt=0.01, viscosity 1e-4 and diffusivity
1e-5. The initial KdV elevation wave has amplitude +0.04 and interface
z=-0.725, halfway between the deep bottom and shelf top.

The updated formulation adds implicit high-mode damping (cutoff 0.6,
power 4, velocity/buoyancy rates 50/250), grad-div coefficient 0.01,
skew-symmetric momentum advection, and a natural no-flux condition on total
buoyancy at the bottom. Top and streamwise-end buoyancy perturbations remain
zero; velocity is no-slip on x/z faces. The background may diffuse.

Simulation jobs 64699161 and 64700291 completed 3,000 steps, with a checkpoint
continuation at t=28.46. Render job 64700307 produced both movies and verified
100 encoded frames each. Median throughput was 0.5063 s/step, with median
11 velocity iterations (maximum 24). True coupled residuals stayed below
1e-10. Independent q=199/207 audits agree on energy, variance and divergence.
Final relative strong divergence is 0.04242%, versus 3.7549% in the previous
case; the absolute divergence L2 norm is 0.000748508.

These remain exploratory, underresolved breaking-wave results. Localized
buoyancy overshoot remains about 1.2% of the unit buoyancy jump. The bottom
condition is imposed weakly; the pointwise bed flux is not converged to zero
(final molecular diffusive-flux L2 norm 1.27803e-4). The initial stratification
is incompatible pointwise with an insulating inclined bed. Neither the movies
nor quadrature agreement establish spatial convergence or resolved mixing.

SHA-256:

- 3D: `6d88dca13c848a9a17a86e0ac959420d12c98f800bed1fe9d0b38493758885fb`
- y=2: `59f90c0765085714364b8ebea6beb76b579e3e4b3ee4496c478811faad85f44b`
