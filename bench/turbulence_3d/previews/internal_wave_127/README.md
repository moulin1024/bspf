# 127³ internal-wave propagation preview

![Internal depression wave propagating toward a tanh continental shelf](overview.png)

Mid-channel total buoyancy and pycnocline displacement at nondimensional times
**t=0** and **t=3**. The wave travels from the deep side toward the shelf
transition centered at x=15. This preview shows initial propagation; the wave
has not yet reached the shelf transition.

The GPU Boussinesq prototype uses the accepted mixed BSPF formulation with
127³ velocity nodes, 125³ pressure nodes, 191³ Gauss points, FP64 arithmetic,
and timestep 0.01. The domain has length 24, width 4, deep depth 1 and shelf
depth 0.45. The initial depression amplitude is -0.04, centered at x=6.
Stationary no-slip walls and a streamwise end sponge bound the tank; the
background stratification is maintained. The KdV-based initial pulse includes
wall tapers and a small transverse perturbation, and is not an exact DJL wave.

Source: Leonardo job 64685983, saved step-300 section, compared with the initial
section from validation job 64685639. Rendered on 2026-11-17. The preview is
not evidence of a completed shelf interaction or spatially converged mixing.
