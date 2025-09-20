import numpy as np

# -----------------------------
# Chebyshev RFFT helpers (your style)
# -----------------------------
def _chebyshev_coeffs_rfft(f_vals):
    N = f_vals.size - 1
    if N < 0:
        raise ValueError("f_vals must have length at least 1 (N+1 >= 1).")
    if N == 0:
        return f_vals.copy()
    v_ext = np.empty(2 * N, dtype=f_vals.dtype)
    v_ext[:N + 1] = f_vals
    v_ext[N + 1:] = f_vals[-2:0:-1]
    half = np.fft.rfft(v_ext)
    a = (half.real / N)
    a[0] *= 0.5
    a[-1] *= 0.5
    return a

def _values_from_cheb_coeffs_irfft(a):
    N = a.size - 1
    if N < 0:
        raise ValueError("a must have length at least 1 (N+1 >= 1).")
    if N == 0:
        return a.copy()
    half = a.astype(np.result_type(a.dtype, np.float64), copy=True)
    half[0] *= 2.0
    half[-1] *= 2.0
    half *= N
    v_ext = np.fft.irfft(half, n=2 * N)
    return v_ext[:N + 1]

def construct_chebyshev_nodes(N, domain=(0, 2*np.pi)):
    a_dom, b_dom = domain
    k = np.arange(N + 1)
    t = np.cos(np.pi * k / N)                           # 1 ... -1 (descending)
    x = (b_dom - a_dom) * 0.5 * t + (b_dom + a_dom) * 0.5
    return x, t

# (kept for completeness; not required by the integrator itself)
def _chebyshev_derivative_coeffs(a):
    N = a.size - 1
    b = np.zeros_like(a)
    if N == 0:
        return b
    r = (2.0 * np.arange(N + 1)) * a
    r = r[1:]
    idx0 = np.arange(N - 1, -1, -2)
    idx1 = np.arange(N - 2, -1, -2)
    b[idx0] = np.cumsum(r[idx0])
    if idx1.size:
        b[idx1] = np.cumsum(r[idx1])
    b[0] *= 0.5
    return b


# --------------------------------------
# Chebyshev integration in coefficient space
# --------------------------------------
def _chebyshev_integral_coeffs_t(a):
    """
    Integrate Chebyshev-T series in t on [-1,1] using the same normalization
    as _chebyshev_coeffs_rfft / _values_from_cheb_coeffs_irfft.
    Returns coefficients b for G'(t)=f(t). b[0] is the free constant.
    """
    N = a.size - 1
    b = np.zeros_like(a)
    if N == 0:
        return b

    if N == 1:
        # f = a0*T0 + a1*T1  => ∫f dt = (a0) T1/1 + (a1/4) T2 + const
        b[1] = a[0]                      # special: a0 contributes fully to T1
        b[1] += 0.0                      # (keep structure; no a2 here)
        # b[N] for N=1 handled below
    else:
        # k = 1 needs special handling because our a0 is "halved"
        b[1] = 0.5 * (2.0 * a[0] - a[2])  # = (a0_std - a2)/(2*1)
        # k = 2..N-1: standard recurrence
        if N >= 3:
            k = np.arange(2, N)
            b[2:N] = (a[1:N-1] - a[3:N+1]) / (2.0 * k)

    # tail (k = N): a_{N+1}=0
    b[N] = a[N-1] / (2.0 * N)
    # b[0] left as the free constant; set it later to match the anchor
    return b

def _apply_constant_to_match_anchor(b_coeffs, anchor, target_value):
    """
    Adjust constant term b0 so that the series value matches target_value at the anchor.
    Anchor: "left" -> x=a (t=-1, last node); "right" -> x=b (t=+1, first node).
    """
    vals = _values_from_cheb_coeffs_irfft(b_coeffs)
    idx = -1 if anchor == "left" else 0
    C = float(target_value) - vals[idx]
    b_coeffs = b_coeffs.copy()
    b_coeffs[0] += C
    return b_coeffs


# -------------------------------------------------
# FFT-based antiderivatives with single-side anchor
# -------------------------------------------------
def chebyshev_antiderivatives_fft(
    f, N=64, domain=(-1.0, 1.0), order=2,
    anchor="left", c1=0.0, c2=0.0
):
    """
    Compute 1st (and optionally 2nd) antiderivatives via FFT-based Chebyshev integration.

    Collocation nodes are Chebyshev–Lobatto (descending): x[0]=b, x[-1]=a.

    We integrate in 't' and scale by s=(b-a)/2:
      U1(t) = C1 + s * ∫ f(t) dt, chosen so U1(anchor)=c1
      U2(t) = C2 + s * ∫ U1(t) dt, chosen so U2(anchor)=c2
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if N < 1:
        # N=0 trivial but not useful for anchored antiderivative; require at least 2 nodes
        raise ValueError("N must be >= 1.")

    a_dom, b_dom = domain
    s = 0.5 * (b_dom - a_dom)
    x_nodes, _ = construct_chebyshev_nodes(N, domain=domain)  # descending
    f_vals = f(x_nodes)

    # --- U1: integrate w.r.t t, then scale by s, then set constant at anchor ---
    a_f = _chebyshev_coeffs_rfft(f_vals)
    b_u1_t = _chebyshev_integral_coeffs_t(a_f)
    b_u1 = s * b_u1_t
    b_u1 = _apply_constant_to_match_anchor(b_u1, anchor, c1)
    U1_vals = _values_from_cheb_coeffs_irfft(b_u1)

    if order == 1:
        return x_nodes, U1_vals

    # --- U2: integrate U1 w.r.t t, scale by s, set constant at anchor ---
    a_u1 = _chebyshev_coeffs_rfft(U1_vals)
    b_u2_t = _chebyshev_integral_coeffs_t(a_u1)
    b_u2 = s * b_u2_t
    b_u2 = _apply_constant_to_match_anchor(b_u2, anchor, c2)
    U2_vals = _values_from_cheb_coeffs_irfft(b_u2)

    x_nodes = np.flip(x_nodes)
    U1_vals = np.flip(U1_vals)
    U2_vals = np.flip(U2_vals)

    return x_nodes, U1_vals, U2_vals


# --------------------------
# Quick self-check (SymPy)
# --------------------------
if __name__ == "__main__":
    import sympy as sp

    # Setup
    N = 256
    domain = (0.0, 2.0 * np.pi)
    anchor = "left"   # or "right"
    c1, c2 = 0.0, 0.0

    x = sp.symbols("x")
    f_expr = sp.sin(1.5 * x)

    # Exact, anchored at chosen side:
    a_dom, b_dom = map(float, domain)
    t_sym, s_sym = sp.symbols("t s")
    lower = a_dom if anchor == "left" else b_dom

    # U1_exact = sp.simplify(c1 + (0.5*(b_dom - a_dom)) * sp.integrate(f_expr.subs(x, t_sym), (t_sym, lower, x)))
    # U2_exact = sp.simplify(c2 + (0.5*(b_dom - a_dom)) * sp.integrate(U1_exact.subs(x, s_sym), (s_sym, lower, x)))

    # After (integrate directly in x; no extra scaling):
    u, v = sp.symbols('u v')
    U1_exact = c1 + sp.integrate(f_expr.subs(x, u), (u, lower, x))
    U2_exact = c2 + sp.integrate(U1_exact.subs(x, v), (v, lower, x))
    
    # print(c1, c2)

    f_num = sp.lambdify(x, f_expr, "numpy")
    U1_exact_num = sp.lambdify(x, U1_exact, "numpy")
    U2_exact_num = sp.lambdify(x, U2_exact, "numpy")

    # Solve with FFT-based method (single-side anchor)
    x_nodes, U1_num, U2_num = chebyshev_antiderivatives_fft(
        f_num, N=N, domain=domain, order=2, anchor=anchor, c1=c1, c2=c2
    )

    
    # Compare
    U1_ref = U1_exact_num(x_nodes)
    U2_ref = U2_exact_num(x_nodes)

    

    err1_inf = float(np.max(np.abs(U1_num - U1_ref)))
    err1_l2  = float(np.linalg.norm(U1_num - U1_ref) / np.sqrt(len(x_nodes)))
    # err2_inf = float(np.max(np.abs(U2_num - U2_ref)))
    # err2_l2  = float(np.linalg.norm(U2_num - U2_ref) / np.sqrt(len(x_nodes)))

    print("=== FFT-based Chebyshev antiderivatives (single-side anchor) ===")
    print(f"N={N}, domain={domain}, anchor='{anchor}', c1={c1}, c2={c2}")
    print("\n--- First antiderivative ---")
    print("U1 max error (inf):", f"{err1_inf:.3e}")
    print("U1 RMS error      :", f"{err1_l2:.3e}")
    # print("\n--- Second antiderivative ---")
    # print("U2 max error (inf):", f"{err2_inf:.3e}")
    # print("U2 RMS error      :", f"{err2_l2:.3e}")


    left_value = U1_exact.subs(x, lower)
    right_value = U1_exact.subs(x, b_dom)
    print(left_value, right_value)
    # c1 = left_value - U2_num[0]
    c2 = ((right_value - U1_num[-1]) - (left_value - U1_num[0])) / (b_dom - a_dom)
    print(c2)
    lift_function = c2*x_nodes + c1
    U1_num = U1_num+lift_function
    # print(U2_num[-1], U2_num[0])
    # print(c1, c2)

    from matplotlib import pyplot as plt
    # plt.plot(x_nodes, U1_num, label="U1_num")
    # plt.plot(x_nodes, U2_num, label="U2_num")
    # plt.plot(x_nodes, U1_exact_num(x_nodes), label="U1_exact")
    plt.plot(x_nodes, U1_exact_num(x_nodes) -U1_num, label="U1_num")
    plt.plot(x_nodes, U2_exact_num(x_nodes)-U2_num, label="U2_exact")
    plt.legend()
    plt.show()
