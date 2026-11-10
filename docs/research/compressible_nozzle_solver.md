# BSPF compressible solver on a mapped block: Laval nozzle findings (2026-11-10)

Steady Euler (and Re = 2000 Navier-Stokes) in a planar Laval nozzle on one
`MappedGrid2D` block: sextic walls h = 1 + 0.5 (1 - (1 - (x/4)^2)^3), area ratio
1.5, total-condition inlet, supersonic exit. Code: `include/bspf/compressible.hpp`
(CPU), `include/bspf/cuda_compressible.hpp` (GPU), `include/bspf/multigrid.hpp`
(FAS multigrid, Newton-Krylov); driver `examples/pde/laval_nozzle_2d.py`, MMS
`examples/pde/laval_nozzle_mms.py`; numbers in `examples/README.md`.

## Discretization

1. Collocation BSPF derivatives with strongly imposed boundary values are not
   stable: boundary-localized growing modes even for 1D acoustics between walls
   (max Re(lambda) ~ 0.3 N for degree-6 FD jets). The nozzle blows up in a few
   hundred steps.
2. The Galerkin BSPF derivative D = M^-1 C (C + C^T = B to 1e-14) is SBP and as
   accurate. With nodal fluxes the dense norm still meets variable coefficients:
   growing modes whose rate grows with N, reproduced by 1D linearized Euler.
3. The weak form (fluxes at Gauss points, weak face fluxes) fixes 1D and nearly
   2D; a spectral vanishing viscosity in a Legendre-weighted Galerkin eigenbasis
   (spectrally small on resolved fields up to the boundary; the natural
   stiffness basis gives an O(1) wall error) removes the rest. SVV 0.2.
4. A free-stream correction (metric-identity defect of the quadrature times the
   nodal fluxes) keeps uniform flow exact; 5 Gauss points per cell (4 leaves a
   ~4e-8 error floor).
5. Chebyshev endpoint jets with n_basis ~ N/3 per axis (at least 24): smooth-field
   derivative errors ~1e-13 and a fit amplification max|F| ~ 10, which FP32 needs
   (fixed n_basis lets it grow like N^(q-1) into the 1e5 range).
6. Without SVV the weak form still has 8 weakly growing modes at 49×17 (max Re
   0.256), independent of the quadrature order: not aliasing. An entropy-variable
   trial field made it worse (68 modes, max 0.455) without the matching
   A0-weighted mass; not pursued.

## Steady solvers

- Explicit local-time-step RK4: ~35 000 steps on 97×33.
- Line-implicit along eta (nodal, then weak-form operator): fixes 193×65 on a
  single grid, still diverges once eta lines are long (193×129).
- FAS multigrid with h- and p-coarsening (degree 5/3 FD-jet trial spaces on the
  coarse levels, semi-coarsening), full-weighting restriction, BSPF
  prolongation, damping 0.5: grid-independent cycles to ~1e-5 (16-22 cycles,
  97×33 to 385×129 with explicit smoothing), but the double-precision tail stalls
  on 193×65 (pseudo-time cannot converge past the weakly damped modes).
- Newton-Krylov (FGMRES, finite-difference Jacobian products) preconditioned by
  one FAS cycle: 7 Newton steps on 97×33 and 193×65 to ~1e-11.

## GPU

- FP32 and FP64 device solvers (templated: cuBLAS S/D, cuFFT R2C/D2Z), GalerkinLine's
  fast evaluation/projection (banded spline + cuFFT) at the Gauss points.
- Mixed precision: FP64 Newton/FGMRES/residual on the GPU, FP32 FAS preconditioner
  with a device-resident cycle and CUDA-graph smoothing.
  193×65: 7 Newton, 162 Krylov, 60 s (CPU FP64 228 s); 97×33: 17 s.

## Comparison with PyFR 2.1 (RTX 3070, same nozzle)

PyFR p4 curved quads, Rusanov, Riemann-invariant inlet (PyFR's Euler system has
no total-condition inlet, so Cd differs; the entropy invariant is exact in both),
dual time with p-multigrid. Case: `bench/pyfr_nozzle_case.py`.

| | DOF | steady time | L2 entropy error |
|---|---|---|---|
| PyFR p4 32×16 | 12 800 | 13.1 s | 5.1e-8 |
| PyFR p4 64×32 | 51 200 | 35.3 s | 3.2e-9 |
| BSPF 97×33 | 3 201 | 17.8 s | 2.0e-6 |
| BSPF 193×65 | 12 545 | 61.7 s | 3.2e-9 |

Per residual (~12.7k DOF): PyFR 45 µs FP64 / 27 µs FP32, BSPF 18.5 ms / 1.67 ms.

Conclusion: BSPF converges much faster with resolution (×600 vs ×20 per 4× DOF;
16× more accurate at equal DOF, 4× fewer DOF for 3.2e-9) and its Newton-multigrid
solve is grid independent, but each residual is 60-400× more expensive (dense
line-global BSPF operators, ~300 kernel launches per residual, FP64 on a
consumer GPU), so PyFR still reaches equal accuracy ~1.7× faster. The residual
evaluation, not the solver, is the next target: a ~10× cheaper residual (fused
kernels, FFT-based application of the line operators instead of dense products)
would put BSPF ahead at equal accuracy.

## Open

- Weakly growing modes of the weak form without SVV (formulation, see 6).
- Viscous terms and line-implicit preconditioning on the GPU.
- Multi-block coupling (interface fluxes through the same SAT/weak-face machinery).
