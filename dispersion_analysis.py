# dispersion_only_bspf_vs_pade_optionA.py
# Focus: DISPERSION ONLY
# - Fix for last-point drop (Option A): restrict θ to the FFT's true top bin.
# - Compares bspf1d (spline + spectral residual) vs compact Padé (orders 4–10).
# - Plots: modified wavenumber.

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ---- adjust this import to your layout ----
from bspf import bspf1d  # assumes your class is importable
try:
    from examples.padefd import padefd  # if 'examples' is a package
except Exception:
    import importlib.util as _ilutil
    import pathlib as _pl
    import sys as _sys
    _p = _pl.Path(__file__).resolve().parent / "examples" / "padefd.py"
    _spec = _ilutil.spec_from_file_location("_padefd_mod", str(_p))
    if _spec and _spec.loader:
        _mod = _ilutil.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        padefd = _mod.padefd
    else:
        raise ImportError("Could not import Padé operator from examples/padefd.py")

# ------------------------- user params -------------------------
N = 1000           # grid points (large to isolate interior)
degree = 9         # spline degree
M_interior = 1   # interior averaging window for symbol estimation
n_theta = 500      # number of θ samples (we'll cap the upper end by theta_max)
# ---------------------------------------------------------------

# ---- bspf1d helpers ----
def build_bspf():
    x = np.linspace(0.0, 1.0, N)  # endpoint-inclusive grid is fine
    return bspf1d.from_grid(
        degree=degree,
        x=x,
        use_clustering=True,
        clustering_factor=2.0,
        correction="spectral",   # full operator: spline + spectral residual
    )

def differentiate_complex(b, arr, k=1, lam=0.0):
    # Combine two real runs; clear cached RHS each time to avoid stale '2bw'
    b._cached_arrays.clear()
    df_r, _ = b.differentiate(arr.real.astype(np.float64), k=k, lam=lam)
    b._cached_arrays.clear()
    df_i, _ = b.differentiate(arr.imag.astype(np.float64), k=k, lam=lam)
    return df_r + 1j * df_i

def interior_symbol_bspf(b, thetas, k=1, lam=0.0, M=256):
    N = b.grid.n
    j = np.arange(N)
    j0 = N // 2 - M // 2
    jj = np.arange(j0, j0 + M)

    G = np.zeros_like(thetas, dtype=np.complex128)
    for t, theta in enumerate(thetas):
        f = np.exp(1j * theta * j)
        df = differentiate_complex(b, f, k=k, lam=lam)
        G[t] = np.mean(df[jj] * np.exp(-1j * theta * jj))
    return G

# ---- Padé (compact) helpers ----
def differentiate_complex_pade(op, arr):
    df_r = op(arr.real.astype(np.float64))
    df_i = op(arr.imag.astype(np.float64))
    return df_r + 1j * df_i

def interior_symbol_pade(op, thetas, M=256):
    N = op.N
    j = np.arange(N)
    j0 = N // 2 - M // 2
    jj = np.arange(j0, j0 + M)

    G = np.zeros_like(thetas, dtype=np.complex128)
    for t, theta in enumerate(thetas):
        f = np.exp(1j * theta * j)
        df = differentiate_complex_pade(op, f)
        G[t] = np.mean(df[jj] * np.exp(-1j * theta * jj))
    return G

# ---- build operator and set θ-range per Option A ----
b = build_bspf()
dx = b.grid.dx
# TRUE top reduced wavenumber supported by the rFFT grid:
theta_max = float(b.grid.omega[-1] * dx)  # omega = 2π * rfftfreq(N, d=dx)
# Sample θ safely below that cap (avoid Nyquist/leakage issues)
thetas = np.linspace(1e-4, theta_max, n_theta)

# ---- compute curves ----
# bspf modified wavenumber: mw_bspf = Im(G1) * Δx
G1_bspf = interior_symbol_bspf(b, thetas, k=1, M=M_interior)
mw_bspf = np.imag(G1_bspf) * dx

# Padé compact derivatives: orders 4, 6, 8, 10
pade_orders = [4, 6, 8, 10]
mw_pade = {}
for _ord in pade_orders:
    _op = padefd(N=b.grid.n, h=dx, order=_ord)
    _G1 = interior_symbol_pade(_op, thetas, M=M_interior)
    mw_pade[_ord] = np.imag(_G1) * dx

# Pseudo-spectral (ideal): mw_ps(θ) = θ
mw_ps = thetas.copy()


# ============================ PLOTS ============================

# 1) Modified wavenumber
plt.figure(figsize=(12,6))
plt.subplot(1, 1, 1)
plt.plot(thetas[:-1], mw_bspf[:-1], label="BSPF")
for _ord in pade_orders:
    plt.plot(thetas[:-1], mw_pade[_ord][:-1], label=f"Padé-{_ord}")
plt.plot(thetas[:-1], mw_ps[:-1],'k--', label="ideal")
# plt.title("Modified wavenumber: BSPF vs Padé (4–10) vs PS")
plt.xlabel("θ")
plt.ylabel("Im(G₁) · Δx")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()