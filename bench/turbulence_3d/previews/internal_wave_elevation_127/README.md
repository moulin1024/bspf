# 127³ mid-shelf elevation-wave movies

- [Central y=2 cross-section MP4](internal_wave_elevation_y2_t30.mp4)
- [3D buoyancy-isosurface MP4](internal_wave_elevation_t30.mp4)

Both movies contain 100 actual snapshots from t=0 through t=30, encoded as
1280×720 H.264 at 10 fps (10 seconds). Decoded frame counts were verified.
The cross-section shows total buoyancy and pycnocline displacement; the 3D
view shows the background-pycnocline isosurface with vertical exaggeration 4.
The interface ends at the solid shelf, rather than being extrapolated through it.

The 127³ FP64 BSPF/Fourier prototype uses periodic y, dt=0.01, viscosity 1e-4 and
diffusivity 1e-5. The background interface is z=-0.725, halfway between the deep
bottom (-1) and shelf top (-0.45), intersecting the ramp at x=15. The initial
packet is an approximate KdV elevation wave with nominal amplitude +0.04,
centered at x=6, with a 1% periodic spanwise seed.

Simulation job 64696007 completed 3,000 steps with a median 0.285109 s/step.
Rendering job 64696023 produced both movies. Independent audit jobs 64696376
and64696672 examined the final field.

These are exploratory shoaling/overturning results, not validated resolved
breaking or mixing. The final strong-divergence L2 norm is 0.0647454, about 3.75%
of the velocity-gradient L2 norm, despite the weak continuity solve meeting its
algebraic tolerance. The bottom still imposes zero buoyancy perturbation,
which pins the background isopycnal contact; this affects interpretation near
the bed. Spatial refinement and a review of the buoyancy wall treatment are
needed before making quantitative breaking claims.
