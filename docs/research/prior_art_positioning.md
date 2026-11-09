# Prior art and positioning of adaptive enriched BSPF

Latest numerical update: the [graded physical-spline comparison at 1e-8](graded_spline_fc_1e8.md) reaches 2.93e-9 error with 70 coefficients in 1.048 seconds median, versus optimized FC-Gram's 5.48e-9 with 2162 physical samples in 1.758 seconds. Five alternating runs support a 1.68-times runtime advantage for this 1D implementation. This supersedes the earlier runtime conclusion for the newly optimized configuration, while preserving the earlier measurements below. The gain combines spline representation and small-operator compilation; it does not establish general superiority, 3D scalability or novelty.

Research snapshot: 6 November 2026. Initial investigation with a subsequent [equation-level comparison and executed ablations](enrichment_equations_and_ablations.md). It is not an exhaustive novelty determination.

Update, 7 November 2026: the [executed ESG-I adaptation and FC-Gram comparison](prior_art_solver_comparison.md) materially changes the performance positioning. At the 1e-6 tolerance, enriched BSPF uses 530 coefficients and takes 89.6 seconds; FC-Gram uses 1793 physical samples plus 25 computed continuation values and takes 2.51 seconds. These single-run results support fewer unknowns for BSPF, but contradict a runtime advantage over FC on this smooth traveling shock. BSPF is faster than our explicit dense Legendre ESG-I adaptation at comparable error; that result does not establish superiority over optimized or implicit enriched spectral implementations.

The subsequent [cost investigation](bspf_fc_cost_gap.md) reduces BSPF quadrature from 64 to 12, retaining 7.08e-7 accuracy in 21.6 seconds. The best measured gap is consequently about 8.6 times in FC's favor, rather than the initial 35.7 times. Defaults remain unchanged pending validation beyond this benchmark.

The promising contribution is an adaptive, conditioned BSPF–Galerkin discretization for resolving physical viscous shock structure with structured FFT operators. Compressible Navier–Stokes simulation, shock enrichment, residualized enrichment, and fast block solves each have substantial precedents. The study should establish what the specific construction contributes beyond those precedents.

## The formulation being assessed

The implementation represents each conservative field using

\[
U_h=\Phi c+Ha,\qquad H=(S-\Phi C)T.
\]

Here \(\Phi\) is the BSPF base, \(S\) a selected dictionary of compact splines, \(C\) the discrete projection onto the **interior** base space, and \(T\) a Gram-based conditioning and rank-selection transform. This is not full orthogonality against every boundary degree of freedom. The remaining mass coupling is handled by a Schur complement. Raw spline evaluation is sparse; subtracting the base projection generally makes the resulting functions global.

Relevant implementation: [enrichment construction](../../examples/pde/shock_tube_enrichment.py), [structured spline actions](../../examples/pde/shock_tube_spline_enrichment.py), [traveling-shock solver](../../examples/pde/viscous_traveling_shock.py), and [Galerkin operators](../../src/galerkin.cpp).

The traveling-shock driver selects one detected front on a fixed 4001-point probe grid. Initialization uses the exact initial condition; subsequent detection uses the numerical solution. Exact data also supplies boundary conditions and error evaluation. This supports a solution-driven single-front demonstration, not yet automatic arbitrary-front discovery. Enrichment changes can entail L2 remapping and rank truncation; absence of artificial viscosity does not mean absence of numerical error or dissipation.

## Closest literature and what it establishes

Read status distinguishes primary full-text passages (F), primary abstract or publisher description (A), and bibliographic leads requiring substantive inspection (L). An unverified feature is not recorded as absent.

| Reference | Status | Established overlap | Distinction or unresolved issue |
|---|---|---|---|
| Li, Lin, Cao and Chen, BSPF (2025), [arXiv](https://arxiv.org/abs/2512.06631) | A | B-spline boundary correction with Fourier residual; Burgers and shallow water | **Our own foundational work.** Separate the new Galerkin, enrichment, and compressible-NS contributions from the original method. |
| Brill and Ihme (2020), [enriched DG](https://web.stanford.edu/group/ihmegroup/cgi-bin/MatthiasIhme/wp-content/papercite-data/pdf/brill2020enrichment.pdf), DOI 10.2514/6.2020-1752 | F, Sections II.C–III | Non-polynomial enrichment for compressible shocks and boundary layers | Discontinuous enrichments are explicitly used only for Euler. Heaviside functions sit at element centers; stationary and moving Sod tests are included. RANS wall tests are separate. No resolved Becker-shock test established here. |
| Albin and Bruno (2011), [FC compressible NS](https://www.math.k-state.edu/~albin/pubs/albin_bruno_2011_sfc.pdf), DOI 10.1016/j.jcp.2011.04.023 | F, Sections 4.3 and 5.5 | Nonperiodic Fourier solver for full compressible NS; steep viscous acoustic fronts | Section 5.5 uses modified continuity/momentum equations and a nonlinear pressure–density closure, not the full energy-equation Becker benchmark. Exponential filtering is part of the solver. |
| Del Rey Fernández et al. (2019), [entropy-stable p-nonconforming NS](https://arxiv.org/pdf/1909.12546) | F, Section 5.1 | Actual full compressible-NS viscous traveling-shock verification at Pr = 3/4 | Closest verified physical benchmark: Mach 2.5, Reynolds 10, perturbed 3D grids. It uses SBP discretization, not BSPF enrichment. |
| Herremans and Huybrechs (2023), [efficient enriched approximation](https://arxiv.org/pdf/2308.05652) | F, Sections 3–4 | Fast base inverse, residualized additional functions, FFT example, enriched Galerkin Schur solve | A particularly close algebraic precedent. Approximation and a 2D boundary-value example do not establish the present adaptive compressible-NS integrator. |
| Babuška and Banerjee (2011/2012), [SGFEM](https://arxiv.org/abs/1104.0960) | A | Modification of enrichment to control severe conditioning | Component precedent; no compressible viscous-shock application established by the inspected abstract. |
| Kergrene, Babuška and Banerjee (2016), [SGFEM and iterative schemes](https://arxiv.org/abs/1603.08571) | A | Separation of base and enrichment spaces, conditioning, solver efficiency | Interface problems; compare the space-angle argument with our discrete Gram construction. |
| Shahbazi et al. (2011), [FC–WENO](https://www.sciencedirect.com/science/article/pii/S0021999111005110), DOI 10.1016/j.jcp.2011.08.024 | A | Fourier continuation in smooth regions and WENO near discontinuities | A competing shock treatment, not a globally enriched Galerkin trial space. Do not infer full-NS tests from the introduction's motivation. |
| Shahbazi (2016), [3D compressible NS FC–WENO](https://doi.org/10.2514/6.2016-3353) | L | Title explicitly targets 3D compressible NS | DOI and authorship verified through Crossref; full text and benchmark equations remain unresolved. |
| Paolucci, Zikoski and Wirasaet (2014), [WAMR Part I](https://www.sciencedirect.com/science/article/abs/pii/S0021999114000527), DOI 10.1016/j.jcp.2014.01.025 | A | Adaptive sparse representation for unsteady compressible reacting flow with diffusive transport | Important precedent for automatic multiscale compression. Uses adaptive collocation grids; not our fixed base-grid enrichment. |
| Bruno, Hesthaven and Leibovici (2021 preprint), [FC–SDNN](https://arxiv.org/abs/2111.01315) | A | Automatic shock detection with nonperiodic spectral computation | Uses localized artificial viscosity; stated tests are advection, Burgers and Euler. |
| Bruno and Leibovici (2025), [general-domain FC–SDNN](https://arxiv.org/abs/2506.16076) | A | Parallel spectral shock computation around obstacles | Artificial-viscosity approach; stated tests are 2D Euler. Relevant to future geometry and scalability claims. |
| Chen and Shen (2018), [enriched spectral methods](https://www.math.purdue.edu/~shen7/pub/She.S19.pdf), JSC 77, 1468–1489 | F, Section 2 | Boundary-compatible enrichment, orthogonalization and ESG-I Schur solves | Direct equation-level audit is now in the linked ablation report. Elliptic theory does not supply nonlinear NS stability. |

## Algebraic overlap that the paper must address

Herremans–Huybrechs Section 4.1 reduces a Fourier-plus-enrichment approximation to a smaller system with matrix \(A_{12}-A_{11}Z_{11}^{*}A_{12}\). Section 5 discusses Schur-complement solution of enriched Galerkin systems. These are close to our subtraction of the base projection and reuse of the base solver. A claim to have invented residualization or fast enrichment solves would therefore be inappropriate. [Primary text](https://arxiv.org/pdf/2308.05652)

Our possible distinction lies in the particular boundary-compatible BSPF space, selected spline dictionary, evolution and remapping of that space, and efficient physical-flux evaluation for compressible NS. Establish this by a side-by-side derivation, not merely different terminology.

Without rank truncation, subtracting an element of the existing base space does not change the combined span:

\[
\operatorname{span}(\Phi,S-\Phi C)=\operatorname{span}(\Phi,S).
\]

Consequently, residualization principally changes conditioning and coordinates. Approximation gains come from the added space; truncation can alter that space. Experiments should separate these effects. Nor does sparsity of \(S\) imply sparsity of \(H\), \(C\), or \(T\). The cost of these dense transforms and the enrichment solve must appear in any scalability argument.

## Claims and their current standing

| Proposed claim | Assessment | Evidence needed or wording to use |
|---|---|---|
| First high-order simulation of a compressible viscous shock | Already precedented | Cite the SBP traveling-shock verification above. |
| First enrichment for compressible shocks | Already precedented | Discuss enriched DG explicitly. |
| First FFT-compatible enrichment or Schur reduction | Already precedented at the component level | Explain the specific BSPF operator structure and boundary treatment. |
| Adaptive enriched BSPF for resolved compressible viscous fronts | Plausible specific contribution; priority unresolved | Complete closest-paper audit and compare formulations. |
| No added artificial viscosity in the demonstrated solver | Supported implementation description | Also disclose quadrature, remapping, rank cutoff and step rejection. |
| Automatic front representation | Demonstrated only for a detected single front | Test multiple fronts, creation, interaction and sensitivity to probe spacing. |
| Gibbs-free or monotone solution | Not established | Use quantified ringing and error curves; do not claim a theorem. |
| Better accuracy per coefficient at stringent tolerances | Supported for the tested benchmark | Preserve parameters, norm, tolerance and reference method configuration. |
| Faster than WENO or DG in general | Not supported | Report tolerance-dependent performance and repeated timings. |
| Scalable in 3D | Not demonstrated by the current 1D results | Measure memory, setup, transforms and solve cost as enriched regions become surfaces. |

The physical benchmark is smooth at positive viscosity. Its sharp-front oscillations reflect insufficient resolution of a smooth layer; they are not proof that the method handles an inviscid discontinuity or satisfies an entropy condition.

## What the existing experiments support

The [recorded threshold comparison](../../benchmarks/traveling_shock/dg/threshold_comparison.txt) uses the same Becker shock with mu = 0.001, Pr = 0.75 and final time 0.05.

| Method | Coefficients per field | Worst primitive error | Recorded seconds |
|---|---:|---:|---:|
| Enriched BSPF | 530 | 7.1006e-7 | 83.717 |
| Ordinary BSPF | 1601 | 6.2564e-7 | 197.339 |
| WENO-Z5 | 8193 | 8.5584e-7 | 146.006 |
| Modal DG-BR1, p = 7 | 4608 | 8.2162e-7 | 144.920 |

These are single serial warm measurements, including setup and error evaluation, excluding JIT warmup and file writing. They meet a common tolerance rather than identical errors. DG includes both interface traces in its peak error. The implementations are local research baselines, not optimized production representatives of every method. The moderate-tolerance WENO results are faster than enriched BSPF; retain that crossover in the paper. See [experiment documentation](../../examples/README.md).

A useful manuscript sentence is: “For the tested viscous traveling shock, adaptive spline enrichment reduces the BSPF degrees of freedom required to reach a prescribed primitive-variable error, while retaining structured FFT evaluation and projection.” Attach runtime claims to the specific table, rather than incorporating universal superiority into the method description.

## Campaign priorities and completion criteria

### 1. Resolve the closest formulation precedents

Audit Chen–Shen, Herremans–Huybrechs, and SGFEM equations against our projection, conditioning and mass solve. Follow their references and forward citations for local dictionaries, evolving bases, spline–Fourier spaces and conservative remapping. Retrieve the 2016 FC–WENO full paper. Trace later shock-enrichment work citing Brill–Ihme; separate wall-modeling descendants from actual shock developments.

Deliverable: an equation-level comparison with exact section references and a list of remaining inaccessible papers. Completion requires explicit evidence for every claimed distinction; a missing keyword is insufficient.

### 2. Isolate the method contribution

Compare ordinary BSPF, raw enrichment, residualized enrichment, and conditioned/rank-selected enrichment. Keep quadrature and error evaluation consistent. Report approximation error before time integration, Gram conditioning, retained rank, remap perturbation, rejected steps and final evolution error.

Include fixed versus adaptive center selection and a clearly labeled oracle-center control. Test alternative dictionary widths/degrees at matched total coefficients and matched cost. This establishes whether benefits arise from added resolution, numerical conditioning, or effective selection.

### 3. Test a problem class beyond one favorable exact solution

Sweep viscosity and shock strength; report physical shock thickness relative to base spacing and spline support. Vary shock position relative to the fixed knot lattice and propagation speed. Add interacting viscous fronts and a smooth small-amplitude wave crossing a steep layer. For non-Becker Prandtl numbers, use an independently validated numerical traveling-wave reference or a converged independent solver.

Maintain the user's fixed-base-grid, no-extra-damping scope. Adaptive-grid methods can be discussed as prior art without changing our solver into an AMR method. Do not call local basis enrichment “no local resolution enhancement.”

### 4. Establish comparative efficiency fairly

Repeat serial timings with uncertainty summaries. Separate setup, adaptation, residual evaluation, mass solve, integration and diagnostics. Compare equal wall time and equal error as well as coefficients. Ensure time and quadrature errors are below the target for every method. Add an entropy-stable DGSEM/SBP baseline and a faithful FC baseline where feasible; the current conventional DG-BR1 result does not represent all DG variants.

### 5. Earn multidimensional claims

Begin with a translated and rotated planar viscous shock in 2D, then a curved front with an independent reference. Measure how required enrichment and dense transform storage grow with front length. A 3D surface test should establish actual memory and runtime scaling before claiming 3D scalability. Include conservation/boundary-flux balance and entropy diagnostics; successful runs alone do not establish nonlinear stability.

## Suggested manuscript positioning

Working title: **Adaptive spline enrichment of BSPF Galerkin methods for resolved viscous shocks**.

Organize related work around four questions: nonperiodic Fourier discretization; enriched approximation and conditioning; physical viscous-shock verification; and adaptive computational cost. Present the original BSPF publication as our foundation. Position the extension as a particular algorithm with demonstrated accuracy and cost benefits, subject to the closest-method audit above.

The most important unresolved issue is whether the adaptive BSPF combination supplies a substantive advance beyond applying established enrichment algebra to a new base. The ablations, conservation/remapping analysis and robustness across front configurations should answer that question.
