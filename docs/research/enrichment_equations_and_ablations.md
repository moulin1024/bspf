# Enrichment equations and controlled ablations

6 November 2026. This comparison concerns the implemented BSPF space and its viscous traveling-shock benchmark. The experiments distinguish enlargement of the approximation space, numerical coordinates, rank truncation, and selection of local basis functions.

## Relation to enriched spectral methods

Chen and Shen construct a boundary-compatible enriched spectral space, orthogonalize enrichment functions, and solve its Galerkin block system using a Schur complement: Eqs. (2.5)–(2.10). Their ESG-II instead estimates enrichment coefficients using decay of the smooth remainder's spectral coefficients: Eqs. (2.12)–(2.17). These are distinct strategies. Our mass solve corresponds structurally to ESG-I, not ESG-II. Their coercive elliptic setting does not supply a stability theorem for our nonlinear NS evolution. [Chen–Shen, 2018](https://www.math.purdue.edu/~shen7/pub/She.S19.pdf)

Herremans and Huybrechs Eq. (16) defines enrichment minus its approximation in a conventional basis. Section 4.1 gives an FFT-based example; Section 5 relates an enriched Galerkin solver to Schur elimination. This is direct precedent for the algebraic structure of our projection residual, rather than merely a broad analogy. [Herremans–Huybrechs, 2023](https://arxiv.org/pdf/2308.05652)

The following derivation uses our discrete quadrature and boundary convention. Let \(W\) be the positive diagonal quadrature matrix, \(\Phi\) the base evaluation matrix, and \(\Phi_I\) its interior columns. Endpoint coefficients are prescribed, so the diagnostic fits subtract the same endpoint lift first. Let \(S\) contain the raw compact splines.

\[
M_I=\Phi_I^T W\Phi_I,\quad B=\Phi_I^T WS,\quad
C_I=M_I^{-1}B,\quad R=S-\Phi_I C_I.
\]

In exact arithmetic,

\[
\Phi_I^TWR=0,\qquad G=R^TWR=S^TWS-B^TM_I^{-1}B.
\]

Thus the residual Gram matrix is itself the raw-space mass Schur complement. With \(G=V\Lambda V^T\), the implementation retains eigenvalues above

\[
\tau=\max(10^{-15},10^{-12}\lambda_{\max}),\qquad
T=V_K\Lambda_K^{-1/2},\qquad H=RT.
\]

Ideally \(H^TWH=I\). For the full base, including endpoint columns, \(Q=\Phi^TWH\) need not vanish. The production solve therefore uses

\[
\begin{bmatrix}M&Q\\Q^T&D\end{bmatrix}
\begin{bmatrix}c\\a\end{bmatrix}
=\begin{bmatrix}b\\d\end{bmatrix},\quad
\Sigma=D-Q^TM^{-1}Q,
\]

\[
a=\Sigma^{-1}(d-Q^TM^{-1}b),\qquad c=M^{-1}(b-Qa),
\]

followed by the boundary constraint through the existing lift. This is standard block elimination. The application-specific work lies in the BSPF operators and how the enriched space is constructed and updated.

For an invertible transform, residualization and conditioning preserve the span:

\[
[\Phi_I,R]=[\Phi_I,S]
\begin{bmatrix}I&-C_I\\0&I\end{bmatrix}.
\]

Rank truncation is different: it can remove approximation directions. To avoid confounding these mechanisms, the experiment compares \([\Phi_I,SV_K]\), \([\Phi_I,RV_K]\), and \([\Phi_I,RT]\) with the **same retained eigenvectors**. All three span the same space in exact arithmetic.

## Evolution and changing spaces

For fixed basis \(\Psi=[\Phi,H]\), each conservative field satisfies a Galerkin evolution of the form

\[
M_\Psi\dot z=\Psi_x^T W(F(U_h)-F_v(U_h,U_{h,x}))+b_{\partial},
\qquad U_h=\Psi z.
\]

Here the boundary contribution is enforced by the prescribed-state lift. The solver freezes the basis during RK4 steps. It changes the space only at update events, using either retained-coordinate injection or a constrained discrete L2 fit. This differs from a continuously moving basis, which would also produce a term \(\Psi^TW\dot\Psi z\). No claim of continuous moving-basis evolution is made.

The sparse implementation evaluates

\[
U_h=\Phi(c-CTa)+S(Ta),\quad
H^TWf=T^T(S^TWf-C^T\Phi^TWf).
\]

These identities retain FFT base actions and sparse raw spline actions without materializing \(H\) during time stepping. They do not make \(C\) or \(T\) sparse, and the dense enrichment operations still matter for scaling.

## Approximation and conditioning experiments

Reproduce with [ablate_traveling_shock.py](../../bench/ablate_traveling_shock.py). All fits use the exact conservative Becker state at t = 0, mu = 0.001, degree-13 splines, spacing h/4 and patch halfwidth 12h. Dense weighted SVD is an independent fitting diagnostic, not a proposed replacement for the production FFT solver. Boundary values are held fixed and peak primitive error is evaluated on 16,001 independent uniform points. The SVD relative cutoff is 1e-14; its effective rank is recorded.

| Base coefficients | Base peak error | Retained enriched peak error | Enriched coefficients |
|---:|---:|---:|---:|
| 65 | 0.446470 | 0.0730781 | 152 |
| 129 | 0.238636 | 0.00535445 | 214 |
| 193 | 0.130701 | 0.000730882 | 278 |

Ordinary BSPF fits using approximately the same total counts (153, 215 and 279, at most one coefficient more) give errors 0.190358, 0.106329 and 0.0586779. Thus, at these resolutions, concentrating extra directions near the front is more effective than increasing the global base by the same count.

At base size 129, the same retained subspace gives:

| Coordinates | Condition number of weighted design matrix | Maximum conservative prediction difference from whitened fit |
|---|---:|---:|
| Raw retained | 3.575e6 | 1.67e-12 |
| Residual retained | 3.266e6 | 1.66e-12 |
| Whitened retained | 12.764 | 0 |

The measured conditioning improvement is about 280,000 times relative to raw retained coordinates, while the represented solution agrees to roundoff-scale accuracy. The mass matrix condition number is the square of the weighted design condition in exact arithmetic; that square is an estimate, not a reliable direct floating-point condition measurement of a nearly singular Gram matrix.

The untruncated raw dictionary has design condition approximately 5.8e13. Its effective SVD rank depends on the numerical cutoff, and normal-equation fits emit conditioning warnings. It is therefore unsuitable as an unqualified reference for the approximation space. At size 129, normal-equation and SVD predictions differ by about 1.4e-5 for the full raw dictionary, versus 1.6e-12 for whitened retained coordinates. This diagnostic is deliberately different from the production Schur solve.

For the retained space, doubling quadrature from 64 to 128 changes the peak error by less than 2e-13. The full ill-conditioned raw fit is substantially less reproducible under that change. Interior orthogonality defects are around 1e-14, but Gram whitening defects reach around 1e-5, consistent with the difficulty of whitening eigenvalues near the cutoff. Do not describe the floating-point basis as exactly orthonormal.

Raising the relative Gram cutoff from 1e-12 to 1e-8 removes three more directions at size 129. The peak primitive error barely changes, but the conservative profile changes by about 4e-6. A nearly unchanged scalar peak error does not prove truncation is harmless for tighter tolerances or derivative accuracy.

## Adaptive selection experiments

These runs use a deliberately less resolved case than the earlier 1e-6 benchmark, to expose mechanism differences: base 129, mu = 0.001, final time 0.1, speed 0.5, degree-9 splines, h/4 spacing, halfwidth 4h, quadrature 48, updates every 0.005, and CFL 0.7. They are not a replacement for the high-accuracy convergence results.

| Treatment | Initial to final coefficients | Final worst primitive error |
|---|---:|---:|
| Ordinary BSPF | 129 to 129 | 0.230406 |
| Ordinary BSPF at matched final count | 183 to 183 | 0.134401 |
| Fixed initial enrichment patch | 163 to 163 | 0.140989 |
| Detected, reconditioned updates | 163 to 183 | 0.00984012 |
| Oracle exact-snapshot detector | 163 to 183 | 0.00984012 |
| Fixed union of detected patches | 183 to 183 | 0.0101603 |
| Detected with retained old directions | 163 to 187 | 0.00977446 |

The fixed-union control is constructed retrospectively from the detected run's patch centers and therefore uses future information. It supplies all those splines from initialization. Its similar final accuracy shows that localization along the traveled path explains most of the improvement; adaptive selection reaches that accuracy while starting with fewer coefficients. This is not evidence that adaptation increases the best-approximation power of the same final space.

Detected and oracle controls give identical results here. This means the detector selected the same patches on this single front at this probe resolution; it does not establish arbitrary-front reliability. The ordinary 183-coefficient control is approximately 13.7 times less accurate than the detected enrichment run, so the improvement is not simply the final coefficient count.

Halving CFL changes the detected primitive profile by 2.46e-7; doubling quadrature changes it by 1.36e-8. Both are small compared with its 9.84e-3 spatial error. No stages were rejected in these controls. These checks support the mechanism comparison at this error level, not 1e-8 accuracy.

Reconditioning produces a largest sampled conservative remap change of 3.75e-3 in this deliberately coarse case. Consequently, differences between fixed-union and adaptive results cannot be assigned purely to selection: representation history, rank selection and remapping also differ. Retaining old directions avoids the L2 remap and gives a similar 0.00977446 error, with four additional modes. This is evidence that the main improvement survives without reconditioning/remap, but not a strictly equal-space comparison because the retained rank differs.

## Artifacts and interpretation

- [Projection data](../../benchmarks/traveling_shock/ablations/projection.json): ranks, condition estimates, coefficient norms, quadrature checks, cutoff sweeps and ordinary fits at approximately matched total coefficients.
- [Evolution data](../../benchmarks/traveling_shock/ablations/evolution.json): all configuration and error details.
- [Selection plot](../../benchmarks/traveling_shock/ablations/selection.png) and [PDF](../../benchmarks/traveling_shock/ablations/selection.pdf).

The JSON and plots are local benchmark artifacts under the repository's ignored benchmarks directory. The script and this report are reproducible source artifacts.

Run from the repository root:

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl python bench/ablate_traveling_shock.py
```

The evidence supports three separate statements: added local spline directions improve approximation; conditioning makes the representation numerically usable; selection follows the front without prescribing its trajectory. It does not establish novelty of residualization or Schur elimination, nonlinear stability, or a general advantage over all enriched spectral methods. No Chen–Shen solver has been benchmarked here; that comparison is equation-level.

Validation: 24 targeted traveling-shock/enrichment tests passed, including new checks for fixed-space evolution and exact-data oracle selection. The broader unit regression passed 167 tests. Two legacy parity/accuracy modules require unavailable `pybspf`; the long-running example-execution module was excluded from the completed regression (an earlier broader run was interrupted after 30 passing tests). All nine ablation evolution cases completed independently.
