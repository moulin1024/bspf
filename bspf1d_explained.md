# How `bspf1d.py` Works

This document explains the implementation in [`bspf1d.py`](./bspf1d.py).

## Short Version

`bspf1d.py` implements a 1D numerical operator for sampled data on a **uniform grid**. Its main job is to:

- fit a smooth **B-spline representation** to the data,
- enforce endpoint information through boundary constraints,
- compute derivatives from the spline,
- correct the spline error with an **FFT-based residual correction**,
- and provide related operations such as interpolation, antiderivatives, and piecewise treatment of discontinuities.

Inference from the code: the name `bspf` most likely refers to a **B-spline + spectral/Fourier** method, because the implementation combines B-spline fitting with spectral correction of the residual.

## Main Idea

The module does not differentiate the raw samples directly.

Instead, it splits the signal into two parts:

1. A smooth part represented by a B-spline basis.
2. A residual `r = f - f_spline`.

Then it computes derivatives as:

`d^k f / dx^k = d^k f_spline / dx^k + d^k r / dx^k`

where:

- the spline derivative comes from differentiating the B-spline basis,
- the residual derivative is computed spectrally with FFT.

That is the core design of the file.

## File Structure

### 1. Backend abstraction

The `_Backend` class switches between:

- NumPy/SciPy on CPU,
- CuPy/CuPyX on GPU.

It is strict about device consistency. The code deliberately avoids implicit NumPy <-> CuPy transfers and raises errors when array types do not match the `use_gpu` setting.

### 2. Uniform grid

`Grid1D` stores:

- the sample points `x`,
- spacing `dx`,
- Fourier frequencies `omega`,
- trapezoidal-rule weights `trap`.

The grid must be uniformly spaced. Many later operations assume that.

### 3. Knot generation

`_Knot` either:

- accepts explicit knots, or
- generates a knot vector automatically from the spline degree and desired number of basis functions.

It also supports optional knot clustering using a `tanh` mapping.

### 4. B-spline basis

`BSplineBasis1D` builds the basis functions and caches:

- `B0`: basis values on the grid,
- `BT0`: transpose of `B0`,
- `BkT(k)`: transpose of the `k`th derivative basis matrix.

This gives the module a matrix form for evaluating the spline and its derivatives.

### 5. Endpoint operators

`EndpointOps1D` builds matrices that encode endpoint information.

There are two important objects:

- `C`: maps spline coefficients to endpoint derivatives,
- `BND`: estimates endpoint derivatives from sample values near the boundary.

This is what lets the spline fit respect boundary behavior instead of only matching interior samples.

## The Core Solve

The main class is `bspf1d`.

When a `bspf1d` object is created, it precomputes:

- the basis matrices,
- trapezoid-weighted matrix `BW`,
- Gram-like matrix `Q = BW @ B0.T`,
- endpoint operators,
- LU factorizations of a KKT system, cached by `lam`.

The solve is a constrained least-squares problem with optional Tikhonov regularization.

Conceptually, it solves for spline coefficients `P` such that:

- the spline matches the sampled data in a weighted least-squares sense,
- endpoint derivative constraints are satisfied,
- and an optional regularization parameter `lam` stabilizes the fit.

The KKT matrix combines:

- the fitting term,
- the regularization term,
- and the endpoint constraints.

## Differentiation Pipeline

The most important methods are:

- `differentiate`
- `differentiate_1_2`
- `differentiate_1_2_3`
- `differentiate_1_2_batched`

For a real-valued input on CPU, the flow is:

1. Build the right-hand side from the data using `BW @ f` and `BND @ f`.
2. Solve the cached KKT system for spline coefficients `P`.
3. Evaluate the spline `f_spline`.
4. Evaluate spline derivatives from the derivative basis matrices.
5. Compute the residual `r = f - f_spline`.
6. Apply FFT to the residual.
7. Multiply by `(i omega)^k`.
8. Transform back and add the correction to the spline derivative.

This hybrid approach is the reason the module can be both smooth and high-order:

- the spline handles non-periodic structure and boundary behavior,
- the FFT correction restores fine-scale accuracy lost in the spline fit.

## Boundary Conditions

The differentiation methods optionally accept:

`neumann_bc=(left_flux, right_flux)`

If given, those values overwrite the first-derivative rows in the boundary data vector. So the operator can enforce explicit Neumann boundary conditions at the two ends of the domain.

There is also `enforced_zero_flux`, which tries to repair endpoint values so the function is compatible with zero-flux behavior by using mirrored ghost points and a spline interpolant.

## Integration and Antiderivatives

### `definite_integral`

This computes:

- the integral of the spline part analytically from the B-spline basis,
- plus the integral of the residual using the trapezoidal rule.

So it follows the same split philosophy as differentiation.

### `antiderivative`

This supports order `1` or `2`.

It:

- integrates the spline part using B-spline antiderivatives,
- integrates the residual spectrally in Fourier space,
- and then fixes the nullspace using `left_value` and optionally `match_right`.

That nullspace handling matters because integration is only defined up to added constants or low-degree polynomials.

## Interpolation

There are two interpolation styles.

### `interpolate`

This creates a refined grid of size `2*N - 1` by inserting midpoints.

By default it:

- fits the spline,
- evaluates the spline on the refined grid.

It also has an optional `use_fft=True` path that performs interpolation purely with FFT assumptions on a periodic signal.

### `interpolate_split_mesh`

This is more explicit and better suited to non-periodic data. It:

1. fits the spline,
2. computes the residual,
3. interpolates the residual on a refined periodic mesh with FFT,
4. evaluates the spline on the refined physical mesh,
5. adds the two parts back together.

It returns both components separately, which is useful for analysis and debugging.

## Piecewise Mode for Discontinuities

`PiecewiseBSPF1D` is a wrapper for functions with known jumps.

It:

- splits the domain at user-provided breakpoints,
- builds one `bspf1d` operator per segment,
- differentiates each segment independently,
- stitches the segment results back together.

This is important because a single smooth spline across a discontinuity is usually a bad approximation.

## What the Public API Gives You

In practice, the module provides these main capabilities:

- `bspf1d.from_grid(...)`: construct the operator from a uniform grid,
- `differentiate(...)`: one derivative order at a time, up to 3,
- `differentiate_1_2(...)`: first and second derivatives together,
- `differentiate_1_2_3(...)`: first, second, and third derivatives together,
- `differentiate_1_2_batched(...)`: same as above for many columns at once,
- `fit_spline(...)`: expose spline coefficients, fitted spline, and residual,
- `definite_integral(...)`: scalar integral over an interval,
- `antiderivative(...)`: first or second antiderivative,
- `interpolate(...)`: refine by midpoint insertion,
- `interpolate_split_mesh(...)`: refined interpolation with separated spline/residual parts,
- `PiecewiseBSPF1D`: segment-wise differentiation for discontinuous signals.

## Performance-Oriented Parts

The code is not only numerical; it is also optimized in a few targeted ways:

- cached LU factorization of the KKT system by `lam`,
- cached derivative basis matrices,
- Fortran-order CPU arrays for more stable BLAS matvec performance,
- preallocated RHS and residual buffers,
- precomputed FFT multipliers for derivative orders 1, 2, and 3,
- fused multi-derivative methods to avoid repeated solves.

Those optimizations explain why there are separate methods for `differentiate`, `differentiate_1_2`, and `differentiate_1_2_3` instead of a single generic path.

## Important Limitations

From the code as written:

- the grid must be uniform,
- derivative orders are limited to `1`, `2`, and `3`,
- antiderivative orders are limited to `1` and `2`,
- some operations are CPU-only or partially CPU-based,
- `interpolate_split_mesh` explicitly rejects GPU mode,
- GPU mode requires strict CuPy inputs and refuses implicit host/device conversion,
- the piecewise wrapper currently exposes only `differentiate_1_2`.

One concrete caveat in the current implementation: the GPU branch inside `interpolate()` appears to call `lu_solve(..., rh, ...)` instead of `rhs` in [`bspf1d.py`](./bspf1d.py). That looks like a typo, so the documented GPU interpolation path may not work as intended without a fix.

## Mental Model

If you want a compact way to think about the module:

- fit a constrained smooth spline,
- measure what the spline missed,
- recover the missed high-frequency content with FFT,
- combine both pieces,
- optionally split the domain when the function is not globally smooth.

That is what `bspf1d.py` does.
