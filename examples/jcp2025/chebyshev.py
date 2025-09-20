import numpy as np
import matplotlib.pyplot as plt
import time


# ---------- FFT-based Chebyshev helpers (DCT-I / IDCT-I via rfft/irfft) ----------

def _chebyshev_coeffs_rfft(f_vals):
    """
    Chebyshev–I coefficients a_k from values at first-kind nodes via
    an rFFT-based DCT-I. Length of `f_vals` must be N+1.
    Scaling convention matches the even-FFT version:
        v_ext = [f_0, ..., f_N, f_{N-1}, ..., f_1]  (length 2N)
        a_k   = (1/N) * Re{RFFT_k(v_ext)} with a_0 and a_N halved.
    """
    N = f_vals.size - 1
    if N < 0:
        raise ValueError("f_vals must have length at least 1 (N+1 >= 1).")
    if N == 0:
        # With only one node, a_0 = f(x_0)
        return f_vals.copy()

    # Build even extension without concatenation
    v_ext = np.empty(2 * N, dtype=f_vals.dtype)
    v_ext[:N + 1] = f_vals
    v_ext[N + 1:] = f_vals[-2:0:-1]

    half = np.fft.rfft(v_ext)        # length N+1
    a = (half.real / N)              # real for even real input
    a[0] *= 0.5
    a[-1] *= 0.5
    return a


def _values_from_cheb_coeffs_irfft(a):
    """
    Inverse DCT-I: given Chebyshev–I coefficients a_k (with endpoints halved
    per `_chebyshev_coeffs_rfft`), return the values at first-kind nodes.

    Returns array of length N+1 with f(t_k), k=0..N, where t_k=cos(pi*k/N).
    """
    N = a.size - 1
    if N < 0:
        raise ValueError("a must have length at least 1 (N+1 >= 1).")
    if N == 0:
        return a.copy()

    # Rebuild the rFFT half-spectrum that would have been produced in forward pass
    half = a.astype(np.result_type(a.dtype, np.float64), copy=True)
    half[0] *= 2.0
    half[-1] *= 2.0
    half *= N

    v_ext = np.fft.irfft(half, n=2 * N)
    return v_ext[:N + 1]


# ---------- Coefficient-space derivative (unchanged complexity, stable) ----------

# def _chebyshev_derivative_coeffs(a):
#     """
#     Convert Chebyshev series coefficients a_k (T_k basis)
#     to coefficients b_k for the derivative in the T_k basis.

#     Recurrence (Trefethen, *Spectral Methods in MATLAB*, 2000, §4):
#         b_{N}       = 0
#         b_{N-1}     = 2N a_N
#         b_{k-2}     = b_{k} + 2(k-1) a_{k-1},  k = N-1 … 2
#         b_0         = b_0 / 2   (final halving)
#     """
#     N = a.size - 1
#     b = np.zeros_like(a)
#     if N == 0:
#         return b
#     b[N - 1] = 2 * N * a[N]
#     for k in range(N - 1, 1, -1):          # k = N-1, …, 2
#         b[k - 2] = b[k] + 2 * (k - 1) * a[k - 1]
#     b[0] *= 0.5
#     return b

def _chebyshev_derivative_coeffs(a):
    """
    Vectorized version (no Python loop).
    Given Chebyshev T_k coefficients a_k, return b_k for d/dx in the T_k basis.
    Uses the fact that b_j = b_{j+2} + 2(j+1) a_{j+1} with b_N = 0 and
    b_{N-1} = 2N a_N, which splits into two parity chains.
    """
    N = a.size - 1
    b = np.zeros_like(a)
    if N == 0:
        return b

    # r_j = 2 (j+1) a_{j+1}, j = 0..N-1  (length N)
    # Build it as r = (2*k*a_k)[1:] with k = 0..N
    r = (2.0 * np.arange(N + 1)) * a
    r = r[1:]  # length N, index j corresponds to coefficient a_{j+1}

    # Descending index lists for the two parity chains
    idx0 = np.arange(N - 1, -1, -2)  # same parity as N-1
    idx1 = np.arange(N - 2, -1, -2)

    # b_j = r_j + r_{j+2} + r_{j+4} + ...  (reversed cumsum on each chain)
    b[idx0] = np.cumsum(r[idx0])
    if idx1.size:
        b[idx1] = np.cumsum(r[idx1])

    # b_N is already zero; final halving of b_0 per the standard recurrence
    b[0] *= 0.5
    return b


def _construct_chebyshev_nodes(N, domain=(-1.0, 1.0)):
    """
    Construct first-kind Chebyshev nodes and map them to the given domain.
    
    Parameters
    ----------
    N : int
        Number of *intervals* (→ N+1 Chebyshev nodes).
    domain : (float, float), optional
        Interval [a, b] to map nodes to (default [-1, 1]).
        
    Returns
    -------
    x : ndarray
        Mapped Chebyshev nodes
    t : ndarray
        Original nodes on [-1, 1]
    """
    a_dom, b_dom = domain
    k = np.arange(N + 1)
    t = np.cos(np.pi * k / N)                 # first-kind nodes on [-1,1]
    x = (b_dom - a_dom) * 0.5 * t + (b_dom + a_dom) * 0.5
    return x, t

def chebyshev_derivative_from_values(f_vals, x, domain=(-1.0, 1.0)):
    """
    Version that accepts pre-computed values and nodes instead of function.
    
    Parameters
    ----------
    f_vals : ndarray
        Function values at Chebyshev nodes
    x : ndarray
        Pre-computed Chebyshev nodes mapped to domain
    domain : (float, float), optional
        Interval [a, b] on which nodes are mapped (default [-1, 1])
    
    Returns
    -------
    df_dx : ndarray
        Derivative values at the nodes
    """
    if len(f_vals) < 2:
        raise ValueError("f_vals must have length at least 2.")
    a_dom, b_dom = domain
    if not np.isfinite(a_dom) or not np.isfinite(b_dom) or a_dom == b_dom:
        raise ValueError("`domain` must be finite with a != b.")

    # 1) Chebyshev coefficients via rFFT-based DCT-I
    a_k = _chebyshev_coeffs_rfft(f_vals)

    # 2) Derivative coefficients via stable recurrence
    b_k = _chebyshev_derivative_coeffs(a_k)

    # 3) Evaluate derivative series at the nodes via inverse DCT-I
    df_dt = _values_from_cheb_coeffs_irfft(b_k)

    # 4) Chain rule for x-mapping t -> x
    df_dx = df_dt * (2.0 / (b_dom - a_dom))

    return df_dx


# ---------- Demo / test ----------

if __name__ == "__main__":
    # Test case: f(x) = sin(αx) * exp(x) with high frequency oscillations
    domain = (0.0, 2.0)
    alpha = 100  # frequency parameter

    # Function and its analytical derivative
    func = lambda x: np.sin(alpha * x) * np.exp(x)
    dfunc = lambda x: (np.sin(alpha * x) + alpha * np.cos(alpha * x)) * np.exp(x)

    # Time complexity analysis
    print("\n=== Time Complexity Analysis ===")
    sizes = [2**k for k in range(8, 16)]  # 256 to 32768 points
    times = []
    
    for n in sizes:
        # Pre-compute nodes (not included in timing)
        x, t = _construct_chebyshev_nodes(n-1, domain)
        f_vals = func(x)
        
        # Warmup
        _ = chebyshev_derivative_from_values(f_vals, x, domain)
        
        # Timing
        n_runs = max(1, 1000 // n)  # Fewer runs for larger sizes
        t0 = time.perf_counter()
        for _ in range(n_runs):
            chebyshev_derivative_from_values(f_vals, x, domain)
        times.append((time.perf_counter() - t0) / n_runs)
    
    # Convert to numpy arrays for analysis
    sizes = np.array(sizes)
    times = np.array(times)
    
    # Compute empirical scaling: if time ~ N^p * log(N), then
    # log(time) ~ p*log(N) + log(log(N))
    # We'll fit log(time) vs log(N) as approximation
    p = np.polyfit(np.log(sizes), np.log(times), 1)[0]
    
    print(f"\nEmpirical scaling: N^{p:.2f}")
    print("Expected: N log N (N^1 * log N)")
    
    # Plot timing results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.loglog(sizes, times, 'bo-', label='Measured time')
    # Plot N log N reference
    scale = times[-1] / (sizes[-1] * np.log(sizes[-1]))
    plt.loglog(sizes, scale * sizes * np.log(sizes), 'r--', 
               label='N log N reference')
    plt.grid(True)
    plt.xlabel('Grid size N')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Time Complexity Analysis')
    
    # Plot ratio to N log N
    plt.subplot(2, 1, 2)
    ratio = times / (scale * sizes * np.log(sizes))
    plt.semilogx(sizes, ratio, 'bo-')
    plt.grid(True)
    plt.xlabel('Grid size N')
    plt.ylabel('Time / (N log N)')
    plt.title('Ratio to N log N Scaling')
    
    plt.tight_layout()
    plt.show()

    # Original accuracy test
    print("\n=== Accuracy Test ===")
    n = 201  # number of intervals -> 202 nodes
    x, t = _construct_chebyshev_nodes(n-1, domain)
    f_vals = func(x)
    dY_cheb = chebyshev_derivative_from_values(f_vals, x, domain)

    # Compute exact derivative for comparison
    dY_exact = dfunc(x)

    # Compute and print maximum error
    max_error = np.max(np.abs(dY_cheb - dY_exact))
    print(f"Maximum error: {max_error:.2e}")

    # Plot accuracy results
    plt.figure(figsize=(12, 8))

    # Plot derivatives
    plt.subplot(2, 1, 1)
    plt.plot(x, dY_exact, '-', label='Analytical derivative')
    plt.plot(x, dY_cheb, '.', label='Chebyshev derivative', markersize=2)
    plt.title(f"Derivative of sin({alpha}x) * exp(x)")
    plt.legend()
    plt.grid(True)

    # Plot error
    plt.subplot(2, 1, 2)
    plt.semilogy(x, np.abs(dY_cheb - dY_exact), '-', label='Absolute Error')
    plt.title('Absolute Error (log scale)')
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
