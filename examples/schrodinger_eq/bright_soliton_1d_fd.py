from __future__ import annotations
import numpy as np
import numpy.typing as npt

try:
    from bspf.utils.padefd import padefd
except ImportError:
    raise ImportError("padefd requires bspf.utils.padefd. Ensure bspf is installed.")

ArrayC = npt.NDArray[np.complex128]
ArrayR = npt.NDArray[np.float64]

# ============================================================
# Problem setup
# ============================================================
# Domain
L = 100.0          # domain length
nx = 1000          # number of grid points
x: ArrayR = np.linspace(0.0, L, nx)
dx = x[1] - x[0]

# Time
T = 10.0           # final time
dt = 0.001         # time step
nt = int(T / dt) + 1

# NLS:  i ψ_t = -(1/2) ψ_xx + V ψ + g |ψ|^2 ψ
A = 1.0            # soliton amplitude
v = 5.0            # soliton velocity
g = -1.0           # focusing
x0 = 0.9 * L       # initial center
V0 = 0.0           # constant potential
V: ArrayR = np.full(nx, V0, dtype=np.float64)


# ============================================================
# Initial condition: bright soliton (free space)
# ============================================================
def bright_soliton_initial(x: ArrayR, A: float, v: float, x0: float, g: float) -> ArrayC:
    """
    Bright soliton at t=0 for
        i ψ_t = -(1/2) ψ_xx + V ψ + g |ψ|^2 ψ,  g < 0.
    """
    kappa = A * np.sqrt(-g)
    envelope = A / np.cosh(kappa * (x - x0))
    phase = np.exp(1j * v * (x - x0))
    return (envelope * phase).astype(np.complex128)


psi0: ArrayC = bright_soliton_initial(x, A, v, x0, g)


# ============================================================
# Neumann Laplacian (for building H)
# ============================================================
def build_H_linear(dx: float, V: ArrayR) -> tuple[ArrayR, ArrayR]:
    """
    Build the *linear* Hamiltonian H_lin corresponding to
        i ψ_t = H_lin ψ,
    where
        H_lin = -(1/2) Δ_h + diag(V),
    and Δ_h is the FD Laplacian with Neumann BC:
        (Δψ)_0     = (ψ_1   - ψ_0   ) / dx^2
        (Δψ)_j     = (ψ_{j+1} - 2ψ_j + ψ_{j-1}) / dx^2, 1<=j<=N-2
        (Δψ)_{N-1} = (ψ_{N-2} - ψ_{N-1}) / dx^2

    We return the *diagonal* and *off-diagonal* of H_lin:
        diag_H[j], off_H[j] = H_{j,j}, H_{j,j+1} (=H_{j+1,j})
    """
    N = V.size
    dx2 = dx * dx

    diag_H = np.empty(N, dtype=np.float64)
    off_H = np.empty(N - 1, dtype=np.float64)

    # off-diagonal is constant: from -1/2 * (1/dx^2)
    off_val = -1.0 / (2.0 * dx2)
    off_H[:] = off_val

    # interior diagonals: -1/2 * (-2/dx^2) + V = 1/dx^2 + V
    diag_H[1:-1] = 1.0 / dx2 + V[1:-1]

    # boundaries: -1/2 * (-1/dx^2) + V = 1/(2 dx^2) + V
    diag_H[0] = 0.5 / dx2 + V[0]
    diag_H[-1] = 0.5 / dx2 + V[-1]

    return diag_H, off_H


# ============================================================
# Tridiagonal factorization + solver (Thomas algorithm)
# ============================================================
def factor_tridiagonal(a: ArrayC, b: ArrayC, c: ArrayC):
    """
    Factor a tridiagonal matrix with diagonals (a,b,c) using
    Thomas algorithm. We overwrite 'a' and 'b' with the LU factors.

    a: sub-diagonal, length N-1
    b: main diagonal, length N
    c: super-diagonal, length N-1

    Returns modified (a, b, c), to be reused for many solves.
    """
    N = b.size
    a = a.copy()
    b = b.copy()
    c = c.copy()

    for i in range(1, N):
        m = a[i - 1] / b[i - 1]
        a[i - 1] = m            # store multiplier
        b[i] = b[i] - m * c[i - 1]

    return a, b, c


def solve_tridiagonal_factored(a: ArrayC, b: ArrayC, c: ArrayC, rhs: ArrayC) -> ArrayC:
    """
    Solve (LU) x = rhs for a tridiagonal matrix whose LU factors
    are stored in (a,b,c) as produced by factor_tridiagonal.
    """
    N = b.size
    y = rhs.copy()

    # Forward substitution (L y = rhs)
    for i in range(1, N):
        y[i] = y[i] - a[i - 1] * y[i - 1]

    # Backward substitution (U x = y)
    x = np.empty_like(y)
    x[-1] = y[-1] / b[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / b[i]

    return x


# ============================================================
# Build Crank–Nicolson operator for linear step
# ============================================================
def build_CN_operators(diag_H: ArrayR, off_H: ArrayR, dt: float):
    """
    Build matrices A and B for the CN step:
        (I + i dt/2 H) ψ^{n+1} = (I - i dt/2 H) ψ^n

    We do NOT build full matrices, only tridiagonal
    coefficients and factor A once.
    """
    N = diag_H.size
    h = dt / 2.0

    # A = I + i h H
    bA = np.ones(N, dtype=np.complex128) + 1j * h * diag_H
    cA = 1j * h * off_H.astype(np.complex128)   # super
    aA = 1j * h * off_H.astype(np.complex128)   # sub

    # B = I - i h H
    bB = np.ones(N, dtype=np.complex128) - 1j * h * diag_H
    cB = -1j * h * off_H.astype(np.complex128)
    aB = -1j * h * off_H.astype(np.complex128)

    # Factor A once
    aA_fac, bA_fac, cA_fac = factor_tridiagonal(aA, bA, cA)

    return (aA_fac, bA_fac, cA_fac), (aB, bB, cB)


def apply_tridiag(a: ArrayC, b: ArrayC, c: ArrayC, psi: ArrayC) -> ArrayC:
    """
    y = (tridiagonal matrix with a,b,c) * psi
    a: sub, b: main, c: super
    """
    N = psi.size
    y = b * psi
    # interior contributions
    y[1:] += a * psi[:-1]
    y[:-1] += c * psi[1:]
    return y


# Linear CN step: psi -> psi_new
def linear_CN_step(psi: ArrayC,
                   factored_A,
                   B_tridiag) -> ArrayC:
    aA_fac, bA_fac, cA_fac = factored_A
    aB, bB, cB = B_tridiag

    rhs = apply_tridiag(aB, bB, cB, psi)
    psi_new = solve_tridiagonal_factored(aA_fac, bA_fac, cA_fac, rhs)
    return psi_new


# ============================================================
# Nonlinear exact substep
# ============================================================
def nonlinear_step(psi: ArrayC, dt_sub: float, g: float) -> ArrayC:
    """
    Exact solution of i ψ_t = g |ψ|^2 ψ over time dt_sub:

      ψ(x, t + dt_sub) = exp(-i g |ψ(x,t)|^2 dt_sub) ψ(x,t).
    """
    phase = np.exp(-1j * g * np.abs(psi)**2 * dt_sub)
    return phase * psi


# ============================================================
# Diagnostics: mass and energy
# ============================================================
def compute_mass(psi: ArrayC, x: ArrayR) -> float:
    density = np.abs(psi)**2
    return float(np.trapz(density, x))


def compute_energy(psi: ArrayC, x: ArrayR, V: ArrayR, g: float, pade_op: padefd = None) -> float:
    """
    E = ∫ [ (1/2)|ψ_x|^2 + V|ψ|^2 + (g/2)|ψ|^4 ] dx
    Uses Padé-4 scheme for derivative computation if pade_op is provided,
    otherwise falls back to centered differences with Neumann BC.
    """
    dx = x[1] - x[0]
    
    if pade_op is not None:
        # Use Padé-4 for first derivative
        # Padé works with real arrays, so compute real and imaginary parts separately
        dpsi_real = pade_op(np.real(psi))
        dpsi_imag = pade_op(np.imag(psi))
        dpsi = dpsi_real + 1j * dpsi_imag
        
        # Enforce Neumann BC: zero derivative at boundaries
        dpsi[0] = 0.0
        dpsi[-1] = 0.0
    else:
        # Fallback to centered differences
        dpsi = np.zeros_like(psi, dtype=np.complex128)
        dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dx)
        dpsi[0] = 0.0
        dpsi[-1] = 0.0

    kinetic = 0.5 * np.abs(dpsi)**2
    potential = V * np.abs(psi)**2
    nonlinear = 0.5 * g * np.abs(psi)**4

    density = kinetic + potential + nonlinear
    return float(np.trapz(density, x))


# ============================================================
# Time integration: Strang splitting + CN
# ============================================================
# Build Padé-4 operator for derivative computation
print("Building Padé-4 operator for derivative computation...")
pade_op = padefd(N=nx, h=dx, order=10)
print(f"  Padé-4 operator created (order=4, N={nx}, h={dx:.6e})")

# Build linear operator and CN coefficients
diag_H, off_H = build_H_linear(dx, V)
factored_A, B_tridiag = build_CN_operators(diag_H, off_H, dt)

psi: ArrayC = psi0.copy()
Psi = np.empty((nt, nx), dtype=np.complex128)
Psi[0] = psi.copy()

times = np.linspace(0.0, T, nt)
energies = np.empty(nt, dtype=np.float64)
masses = np.empty(nt, dtype=np.float64)

energies[0] = compute_energy(psi, x, V, g, pade_op=pade_op)
masses[0] = compute_mass(psi, x)

print("=" * 60)
print("1D NLS – Strang splitting + Crank–Nicolson (Neumann walls)")
print("=" * 60)
print(f"Domain: L = {L:.3f}, nx = {nx}, dx = {dx:.6e}")
print(f"Time:   T = {T:.3f}, dt = {dt:.6e}, nt = {nt}")
print(f"NLS:    g = {g:.3f}, V0 = {V0:.3f}")
print(f"IC:     bright soliton, A = {A:.3f}, v = {v:.3f}, x0 = {x0:.3f}")
print(f"Derivative: Padé-4 scheme (order=4)")
print("=" * 60)

for n in range(1, nt):
    print(f"Time step {n} of {nt}")
    # Strang splitting: N(dt/2) → L(dt) → N(dt/2)
    psi = nonlinear_step(psi, dt / 2.0, g)
    psi = linear_CN_step(psi, factored_A, B_tridiag)
    psi = nonlinear_step(psi, dt / 2.0, g)

    Psi[n] = psi
    energies[n] = compute_energy(psi, x, V, g, pade_op=pade_op)
    masses[n] = compute_mass(psi, x)

print("Time integration finished.")

E0 = energies[0]
M0 = masses[0]

dE = energies - E0
dM = masses - M0

print("\n" + "=" * 60)
print("Conservation diagnostics (Strang + CN, Neumann BC)")
print("=" * 60)
print(f"Initial energy:  E0 = {E0:.10e}")
print(f"Final energy:    ET = {energies[-1]:.10e}")
print(f"Max |E(t)-E0|:   {np.max(np.abs(dE)):.10e}")
print(f"Initial mass:    M0 = {M0:.10e}")
print(f"Final mass:      MT = {masses[-1]:.10e}")
print(f"Max |M(t)-M0|:   {np.max(np.abs(dM)):.10e}")
print("=" * 60)


# ============================================================
# Optional plots
# ============================================================
try:
    import matplotlib.pyplot as plt

    # |psi|^2 initial vs final
    plt.figure(figsize=(10, 5))
    plt.plot(x, np.abs(Psi[0])**2, label="t=0", linewidth=2)
    plt.plot(x, np.abs(Psi[-1])**2, label=f"t={T}", linestyle="--", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("|ψ|²")
    plt.title("Bright soliton with reflecting walls (Strang + CN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Space–time
    plt.figure(figsize=(10, 5))
    plt.imshow(
        np.abs(Psi)**2,
        origin="lower",
        aspect="auto",
        extent=[0.0, L, 0.0, T],
        cmap="plasma",
        interpolation="bilinear",
    )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("|ψ|² evolution")
    cbar = plt.colorbar()
    cbar.set_label("|ψ|²")
    plt.tight_layout()

    # Energy error
    plt.figure(figsize=(10, 4))
    plt.plot(times, dE)
    plt.axhline(0.0, linestyle="--", color="k", linewidth=1, alpha=0.5)
    plt.xlabel("t")
    plt.ylabel("E(t) - E0")
    plt.yscale("symlog", linthresh=1e-12)
    plt.title("Energy error (Strang + CN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Mass error
    plt.figure(figsize=(10, 4))
    plt.plot(times, dM)
    plt.axhline(0.0, linestyle="--", color="k", linewidth=1, alpha=0.5)
    plt.xlabel("t")
    plt.ylabel("M(t) - M0")
    plt.yscale("symlog", linthresh=1e-14)
    plt.title("Mass error (Strang + CN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
except Exception as exc:
    print("Plotting skipped:", exc)
