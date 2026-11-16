# Fine resolution sweep: curved-domain scalar Poisson

Campaign status at plot generation: complete. Plots use the available recorded results; pending cases are not inferred.

Same fixed nonconformal four-cubic-edge Coons domain and synthetic turbulence MMS, shells 4–128, seed 20261115. Exact nonzero Dirichlet boundaries; no time stepping. Solver kernels, tolerances, setup exclusion and audit criteria are unchanged.

## Sweep and timing

- BSPF starts at n=96 and increases by 8 to n=288, with q=2n. Both direct and nodal RHS policies are included.
- Nektar++ uses E=8 elements/axis with p=12,14,...,40; E=16 with p=6,8,...,32 plus odd degrees 21,23,...,31. This starts near BSPF’s lowest nominal resolution and resolves the tighter-error transition.
- Existing matching runs are reused. Every new case has one native solve, one CPU thread, no warmup, no repeated timing. All timed jobs run sequentially.
- Analytic sampling and setup are excluded; Nektar++ static-condensation factorization is prepared before timing.
- BSPF direct and Nektar direct sample the same analytic forcing at their own quadrature points. BSPF nodal samples forcing at BSPF nodes and includes interpolation to quadrature inside timing. The nodal policy has different input information.

## Fastest measured configurations reaching each target

Only validated points qualify. Independent quadrature estimates must agree within 5% of the requested target. Values are selected from measured configurations; no log-linear interpolation.

| Error target | BSPF direct ms (configuration) | BSPF nodal ms (configuration) | Nektar++ direct ms (configuration) |
|---:|---:|---:|---:|
| 1e-02 | 170.737 (n=232, q=464) | 205.188 (n=240, q=480) | 24.948 (E=16, p=16) |
| 1e-03 | 220.479 (n=240, q=480) | 212.699 (n=248, q=496) | 34.224 (E=16, p=20) |
| 1e-04 | 236.264 (n=248, q=496) | 212.699 (n=248, q=496) | 77.545 (E=16, p=22) |
| 1e-05 | 236.264 (n=248, q=496) | 235.033 (n=264, q=528) | 138.517 (E=16, p=25) |
| 1e-06 | 286.763 (n=264, q=528) | 235.033 (n=264, q=528) | 212.229 (E=16, p=26) |
| 1e-07 | 286.763 (n=264, q=528) | 235.033 (n=264, q=528) | 302.365 (E=16, p=29) |
| 1e-08 | 286.763 (n=264, q=528) | 235.033 (n=264, q=528) | 401.606 (E=16, p=30) |
| 1e-10 | 309.162 (n=272, q=544) | 285.890 (n=272, q=544) | not reached |
| 1e-12 | 309.162 (n=272, q=544) | 341.860 (n=280, q=560) | not reached |

## Qualification and interpretation

The wall-time plot uses a best-measured staircase: with a given time budget, it reports only solutions actually measured within that budget. Vertical jumps correspond to newly affordable measured configurations; they do not imply intermediate accuracies were measured. Faint markers retain dominated measurements. Failed audits are excluded from target selection and shown as crosses.

The resolution plot connects sampled resolutions only to make convergence trends readable; it is not used to infer solve times. Nektar E p + 1 and BSPF n label nominal resolution, not equal forcing-sample counts or equal computational effort.

Finer sampling can change the ranking implied by a sparse ladder. Even this finer sweep only establishes the fastest tested configurations, not a global optimum or universal method ranking. Reused and new timings are single observations from different moments; small timing differences should not be interpreted as statistically established gains.

## Recorded cases

### BSPF direct RHS

| Configuration | Solve ms | Relative L2 | Status | Reused |
|---|---:|---:|---|---|
| n=96, q=192 | 17.766 | 2.216335e-01 | audit_failed | True |
| n=104, q=208 | 19.534 | 2.041743e-01 | audit_failed | False |
| n=112, q=224 | 26.087 | 1.849531e-01 | validated | False |
| n=120, q=240 | 30.782 | 1.714742e-01 | validated | False |
| n=128, q=256 | 39.285 | 1.564367e-01 | validated | True |
| n=136, q=272 | 42.812 | 1.395354e-01 | validated | False |
| n=144, q=288 | 60.792 | 1.252407e-01 | validated | False |
| n=152, q=304 | 62.361 | 1.108266e-01 | validated | False |
| n=160, q=320 | 72.810 | 9.886279e-02 | validated | True |
| n=168, q=336 | 65.271 | 8.334454e-02 | validated | False |
| n=176, q=352 | 79.039 | 7.174417e-02 | validated | False |
| n=184, q=368 | 79.676 | 5.948853e-02 | validated | False |
| n=192, q=384 | 130.416 | 4.963855e-02 | validated | True |
| n=200, q=400 | 104.812 | 3.591112e-02 | validated | False |
| n=208, q=416 | 112.001 | 2.685076e-02 | validated | False |
| n=216, q=432 | 135.439 | 1.709005e-02 | validated | False |
| n=224, q=448 | 200.958 | 9.597033e-03 | validated | True |
| n=232, q=464 | 170.737 | 2.748691e-03 | validated | False |
| n=240, q=480 | 220.479 | 2.921297e-04 | validated | False |
| n=248, q=496 | 236.264 | 7.541351e-06 | validated | False |
| n=256, q=512 | 328.509 | 8.119526e-08 | validated | True |
| n=264, q=528 | 286.763 | 4.399323e-10 | validated | False |
| n=272, q=544 | 309.162 | 6.824540e-13 | validated | False |
| n=280, q=560 | 318.482 | 5.133980e-13 | validated | False |
| n=288, q=576 | 421.599 | 2.290314e-13 | validated | True |

### BSPF nodal RHS

| Configuration | Solve ms | Relative L2 | Status | Reused |
|---|---:|---:|---|---|
| n=96, q=192 | 14.082 | 6.678103e+01 | validated | True |
| n=104, q=208 | 18.297 | 2.858710e+01 | validated | False |
| n=112, q=224 | 20.242 | 1.364434e+01 | validated | False |
| n=120, q=240 | 24.541 | 3.246848e+00 | validated | False |
| n=128, q=256 | 40.885 | 1.013988e+00 | validated | True |
| n=136, q=272 | 44.489 | 1.128485e+00 | validated | False |
| n=144, q=288 | 50.454 | 7.835104e-01 | validated | False |
| n=152, q=304 | 54.077 | 6.510850e-01 | validated | False |
| n=160, q=320 | 73.243 | 6.521906e-01 | validated | True |
| n=168, q=336 | 71.056 | 7.546045e-01 | validated | False |
| n=176, q=352 | 93.883 | 7.787693e-01 | validated | False |
| n=184, q=368 | 115.368 | 6.226671e-01 | validated | False |
| n=192, q=384 | 114.963 | 4.064402e-01 | validated | True |
| n=200, q=400 | 113.531 | 1.692153e-01 | validated | False |
| n=208, q=416 | 109.507 | 1.307659e-01 | validated | False |
| n=216, q=432 | 127.708 | 2.105191e-01 | validated | False |
| n=224, q=448 | 204.654 | 3.813167e-02 | validated | True |
| n=232, q=464 | 261.726 | 1.406463e-02 | validated | False |
| n=240, q=480 | 205.188 | 8.255715e-03 | validated | False |
| n=248, q=496 | 212.699 | 3.109560e-05 | validated | False |
| n=256, q=512 | 320.291 | 1.924359e-06 | validated | True |
| n=264, q=528 | 235.033 | 4.328403e-09 | validated | False |
| n=272, q=544 | 285.890 | 9.572101e-12 | validated | False |
| n=280, q=560 | 341.860 | 4.895905e-13 | validated | False |
| n=288, q=576 | 392.782 | 5.942969e-13 | validated | True |

### Nektar++ direct RHS

| Configuration | Solve ms | Relative L2 | Status | Reused |
|---|---:|---:|---|---|
| E=8, p=12 | 2.297 | 2.268786e-01 | validated | True |
| E=16, p=6 | 2.113 | 2.207349e-01 | validated | False |
| E=8, p=14 | 4.597 | 1.914154e-01 | validated | False |
| E=8, p=16 | 5.140 | 1.592274e-01 | validated | True |
| E=16, p=8 | 3.459 | 1.538659e-01 | validated | False |
| E=8, p=18 | 6.088 | 1.295952e-01 | validated | False |
| E=8, p=20 | 16.099 | 1.033592e-01 | validated | False |
| E=16, p=10 | 5.129 | 9.793199e-02 | validated | False |
| E=8, p=22 | 13.041 | 7.959158e-02 | validated | False |
| E=8, p=24 | 16.183 | 5.747258e-02 | validated | True |
| E=16, p=12 | 10.632 | 5.219965e-02 | validated | True |
| E=8, p=26 | 34.343 | 3.795199e-02 | validated | False |
| E=8, p=28 | 72.155 | 2.203946e-02 | validated | False |
| E=16, p=14 | 12.687 | 2.145880e-02 | validated | False |
| E=8, p=30 | 52.879 | 1.084905e-02 | validated | False |
| E=8, p=32 | 85.696 | 4.467146e-03 | validated | False |
| E=16, p=16 | 24.948 | 6.658595e-03 | validated | True |
| E=8, p=34 | 110.775 | 1.545115e-03 | validated | False |
| E=8, p=36 | 155.647 | 4.539027e-04 | validated | False |
| E=16, p=18 | 26.026 | 1.585176e-03 | validated | False |
| E=8, p=38 | 290.793 | 1.147591e-04 | validated | False |
| E=8, p=40 | 459.597 | 2.526077e-05 | validated | False |
| E=16, p=20 | 34.224 | 2.969376e-04 | validated | False |
| E=16, p=21 | 45.823 | 1.040150e-04 | validated | False |
| E=16, p=22 | 77.545 | 4.484634e-05 | validated | False |
| E=16, p=23 | 117.207 | 1.395219e-05 | validated | False |
| E=16, p=24 | 151.256 | 5.577143e-06 | validated | True |
| E=16, p=25 | 138.517 | 1.564306e-06 | validated | False |
| E=16, p=26 | 212.229 | 5.814172e-07 | validated | False |
| E=16, p=27 | 422.013 | 1.489750e-07 | validated | False |
| E=16, p=28 | 305.852 | 5.158533e-08 | validated | False |
| E=16, p=29 | 302.365 | 1.221207e-08 | validated | False |
| E=16, p=30 | 401.606 | 3.945640e-09 | validated | False |
| E=16, p=31 | 450.378 | 8.713494e-10 | validated | False |
| E=16, p=32 | 685.890 | 2.630615e-10 | validated | True |
