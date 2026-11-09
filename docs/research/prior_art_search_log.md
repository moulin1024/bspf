# Search log for enriched BSPF prior art

Date: 6 November 2026. Scope: scientific method positioning, not a patent opinion. This pass used web search, primary author PDFs, arXiv records and publisher/institutional records. It did not exhaust Scopus/Web of Science citation graphs or inaccessible proceedings.

## Search families executed

- Fourier spectral local enrichment shock compressible Navier Stokes B spline adaptive
- "Fourier" "enrichment" "shock"
- "spectral" "enriched" "viscous shock"
- "Fourier continuation" "Navier" "shock"
- "High-Order Hybrid Fourier Continuation-WENO" Navier Stokes
- "partition of unity" "compressible" "shock" enrichment
- "Fourier" "B-spline" "shock" Galerkin
- "wavelet adaptive" "compressible" "Paolucci" 2014
- "WAMR" "Part I" 2014 sciencedirect
- "enriched spectral" Fourier local basis
- "Fourier" "splines" "enrichment" numerical
- "shock" "enrichment" "Navier-Stokes" numerical method
- "Brill" "Ihme" "enriched" shock 2021 2022 2023
- "spectral FC solver" "shock" Albin Bruno
- "enriched spectral-Galerkin" "Chen" "Shen"

Search snippets were used to locate primary sources. Irrelevant aggregators and automatically generated literature pages were excluded as evidence. Failed retrievals were not treated as evidence of absent content.

## Critical findings and source locations

| Source | Locator | Consequence |
|---|---|---|
| [Brill–Ihme 2020](https://web.stanford.edu/group/ihmegroup/cgi-bin/MatthiasIhme/wp-content/papercite-data/pdf/brill2020enrichment.pdf) | PDF p. 5, after Eq. 36; Section III | Distinguish Euler shock enrichment from viscous wall modeling. |
| [Albin–Bruno 2011](https://www.math.k-state.edu/~albin/pubs/albin_bruno_2011_sfc.pdf) | Sections 4.3 and 5.5, Eqs. 39–41, Fig. 10 | Inspect the acoustic closure rather than inferring benchmark physics from the paper title. |
| [Herremans–Huybrechs 2023](https://arxiv.org/pdf/2308.05652) | Sections 3, 4.1 and 5; reference 6 | Direct algebraic overlap and a route to Chen–Shen. |
| [Del Rey Fernández et al. 2019](https://arxiv.org/pdf/1909.12546) | Section 5.1, Eqs. 51–53 | Verified full-NS physical viscous-shock precedent. |

## Open retrievals and version checks

1. Chen–Shen (2018), JSC 77, 1468–1489: author PDF located but retrieval failed. Bibliography and Schur discussion verified in Herremans–Huybrechs; do not label the original paper fully read.
2. Shahbazi (2016), DOI [10.2514/6.2016-3353](https://doi.org/10.2514/6.2016-3353): metadata verified via [Crossref](https://crossmark.crossref.org/dialog?cm_version=v2.0&doi=10.2514%2F6.2016-3353&domain=arc.aiaa.org&uri_scheme=https%3A); full text unresolved. Conference-program extraction contains inconsistent numbering, so use the DOI record.
3. Original BSPF: [arXiv 2512.06631](https://arxiv.org/abs/2512.06631) is the group's own 2025 paper. A same-author [SSRN version titled A flexible B-spline-Fourier pseudo-spectral method with applications to nonlinear waves](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6366409) surfaced with a March 2026 posting. Reconcile versions before finalizing the new-versus-published contribution boundary.
4. Brill's subsequent spectral-element wall-enrichment work, including [arXiv 2403.05670](https://arxiv.org/abs/2403.05670), is a forward-search lead, not yet an equation-level shock precedent in this report.
5. WAMR Part II and earlier viscous-reactive-wave papers: inspect primary full text for physical front resolution, error controls and parallel scaling before quantitative comparison.
6. Broaden subsequent searches to Eckhoff discontinuity reconstruction, Gegenbauer recovery, VMS/orthogonal subscales, wavelet Galerkin and adaptive frames. Classify reconstruction-only methods separately from PDE evolution.

## Screening template for subsequent passes

For each candidate record: exact bibliographic version; source URL; inspected sections; PDE and constitutive laws; physical viscosity/heat conduction; basis and projection; stabilization; sensor and prior front information; adaptation/remapping; conservation and stability results; fast operators; cost dependence on enrichment count; dimension; benchmark norms; overlap and distinguishing evidence.

A novelty conclusion remains provisional until the highest-overlap papers and their relevant backward/forward citations have been checked. This log records an initial campaign milestone, not an ongoing scheduled background search.

## Subsequent equation audit

Chen–Shen full-text retrieval succeeded on 6 November 2026. Section 2 and Eqs. (2.5)–(2.17) were inspected directly. The initial retrieval gap above is resolved. See [equations and ablations](enrichment_equations_and_ablations.md) for the comparison and executed controls.
