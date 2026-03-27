import numpy as np
import matplotlib.pyplot as plt
from zhao import zhao2025_spectral_derivative
from bspf1d import bspf1d

try:
    # Optional: extension visualization (requires SciPy inside zhao.py)
    from zhao import zhao2025_extend
except Exception:
    zhao2025_extend = None

# 1. 准备数据 (非周期函数)
DOMAIN = (-np.pi+0.03, np.pi-0.07)
DEGREE = 11
NUM_BOUNDARY_POINTS = DEGREE + 5
USE_CLUSTERING = True
CLUSTERING_FACTOR = 3.0
N_BASIS = 2 * (DEGREE + 1) * 2
x = np.linspace(DOMAIN[0], DOMAIN[1], 10001)
import sympy as sp

x_sym = sp.Symbol("x", real=True)
# y_sym = sp.sin(100.5 * x_sym)
y_sym = sp.sin(x_sym/(1.02+sp.cos(x_sym)))
dy_sym = sp.diff(y_sym, x_sym)

y = sp.lambdify(x_sym, y_sym, "numpy")(x)
dy_exact = sp.lambdify(x_sym, dy_sym, "numpy")(x)



# 2. 计算导数
dy_zhao = zhao2025_spectral_derivative(x, y, order=1)

# 计算 bspf 导数（原始方法）
bspf_op = bspf1d.from_grid(degree=DEGREE, x=x, domain=DOMAIN, n_basis=N_BASIS, num_boundary_points=NUM_BOUNDARY_POINTS, use_clustering=USE_CLUSTERING, clustering_factor=CLUSTERING_FACTOR)
dy_bspf, f_spline_bspf = bspf_op.differentiate(y, k=1)

# 3. 验证精度
error_zhao = np.max(np.abs(dy_zhao - dy_exact))
error_bspf = np.max(np.abs(dy_bspf - dy_exact))
print(f"Zhao2025 Max Error: {error_zhao:.2e}")
print(f"BSPF Max Error: {error_bspf:.2e}")

# 4. 可视化（3 panels）
fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

# Panel 1: 原函数 + 延拓函数（如果可用）
ax1 = axes[0]
ax1.plot(x, y, "b-", lw=2, label="Original y(x)")
if zhao2025_extend is not None:
    try:
        x_ext, y_ext = zhao2025_extend(x, y, domain=DOMAIN)
        # ax1.plot(x_ext, y_ext, color="0.4", lw=1.5, alpha=0.9, label="Zhao2025 extension")
        ax1.axvline(x[-1], color="0.7", lw=1, ls="--")
    except Exception as e:
        ax1.text(
            0.02,
            0.95,
            f"Extension unavailable:\n{type(e).__name__}: {e}",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
        )
else:
    ax1.text(
        0.02,
        0.95,
        "Extension unavailable (SciPy not installed).",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )
ax1.set_title("Panel 1: Original function and (optional) Zhao2025 extension")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True, alpha=0.25)
ax1.legend(loc="best")

# Panel 2: 误差分布（dy）
ax2 = axes[1]
err_zhao = dy_zhao - dy_exact
err_bspf = dy_bspf - dy_exact
ax2.semilogy(x, np.abs(err_zhao), "k-", lw=1.5, label="Error: dy_zhao - dy_exact")
ax2.semilogy(x, np.abs(err_bspf), "r--", lw=1.5, label="Error: dy_bspf - dy_exact")
ax2.axhline(0.0, color="0.7", lw=1)
ax2.set_title("Panel 2: Derivative error distribution vs x")
ax2.set_xlabel("x")
ax2.set_ylabel("error")
ax2.grid(True, alpha=0.25)
ax2.legend(loc="best")

# Panel 3: 收敛性研究（L2 norm error vs N）
ax3 = axes[2]
Ns = [1001, 1501, 2001,2501,3001,3501,4001,4501,5001,5501,6001,6501,7001,7501,8001,8501,9001,9501,10001]
errs_zhao = []
errs_bspf = []
for N in Ns:
    xx = np.linspace(DOMAIN[0], DOMAIN[1], N)
    # if "sp" in globals():
    yy = sp.lambdify(x_sym, y_sym, "numpy")(xx)
    ddy_exact = sp.lambdify(x_sym, dy_sym, "numpy")(xx)

    # For small N, make sure extension parameters are valid (m_delta <= N).
    m_delta = min(25, N)
    ddy_zhao = zhao2025_spectral_derivative(xx, yy, order=1, domain=DOMAIN, m_delta=m_delta)
    err_zhao = ddy_zhao - ddy_exact
    errs_zhao.append(np.linalg.norm(err_zhao))
    
    # BSPF convergence study (原始方法)
    bspf_op_n = bspf1d.from_grid(degree=DEGREE, x=xx, domain=DOMAIN, n_basis=N_BASIS, num_boundary_points=NUM_BOUNDARY_POINTS, use_clustering=USE_CLUSTERING, clustering_factor=CLUSTERING_FACTOR)
    ddy_bspf, _ = bspf_op_n.differentiate(yy, k=1)
    err_bspf = ddy_bspf - ddy_exact
    errs_bspf.append(np.linalg.norm(err_bspf))

ax3.loglog(Ns, errs_zhao, "ko-", lw=2, label="Zhao2025 L2 norm")
ax3.loglog(Ns, errs_bspf, "ro--", lw=2, label="BSPF L2 norm")

# Fit convergence rate for BSPF using last 10 points
if len(errs_bspf) >= 10:
    from scipy import stats
    n_fit = 10
    Ns_fit = Ns[-n_fit:]
    errs_bspf_fit = errs_bspf[-n_fit:]
    
    # Fit power law: error = C * N^(-p) in log-log space
    log_Ns = np.log(Ns_fit)
    log_errs = np.log(errs_bspf_fit)
    
    # Linear fit: log(error) = log(C) - p * log(N)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_Ns, log_errs)
    convergence_rate = -slope  # p in error = C * N^(-p)
    log_C = intercept
    
    # Generate fitted curve
    Ns_fit_plot = np.array(Ns_fit)
    errs_fit = np.exp(log_C) * (Ns_fit_plot ** (-convergence_rate))
    
    # Plot fitted line
    ax3.loglog(Ns_fit_plot, errs_fit, "r-", lw=2, alpha=0.6, 
               label=f"BSPF fit (rate={convergence_rate:.2f}, R²={r_value**2:.4f})")
    
    # Print convergence rate
    print(f"\nBSPF Convergence Rate (last {n_fit} points):")
    print(f"  Rate: {convergence_rate:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  C: {np.exp(log_C):.2e}")

ax3.set_title("Panel 3: Convergence study (L2 norm error vs N)")
ax3.set_xlabel("N (odd grid size)")
ax3.set_ylabel("L2 norm |dy - dy_exact|")
ax3.grid(True, which="both", alpha=0.25)
ax3.legend(loc="best")

plt.show()