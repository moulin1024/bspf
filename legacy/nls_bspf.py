import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    from bspf1d import bspf1d
    HAS_BSPF = True
except ImportError:
    HAS_BSPF = False
    print("Warning: 'bspf1d' not found. Please install BSPF library.")


# -----------------------------
# Basis functions for diagnostics
# -----------------------------
def sine_mode(n, x, L):
    # Dirichlet eigenmode on (0,L): sqrt(2/L) sin(n pi x/L), n>=1
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


# -----------------------------
# Boundary condition enforcement for Dirichlet
# -----------------------------
def enforce_dirichlet_bc(psi):
    """
    Enforce Dirichlet BC: psi(0) = psi(L) = 0
    """
    psi_bc = psi.copy()
    psi_bc[0] = 0.0
    psi_bc[-1] = 0.0
    return psi_bc


# -----------------------------
# Initial condition helpers
# -----------------------------
def initial_condition(x, L, a=1.0, b=0.6, phase=0.5*np.pi, n1=1, n2=2):
    """
    Build a simple two-mode IC for Dirichlet BC: sine modes n1,n2
    """
    psi = a * sine_mode(n1, x, L) + b * np.exp(1j * phase) * sine_mode(n2, x, L)
    psi[0] = 0.0
    psi[-1] = 0.0
    return psi.astype(np.complex128)


# -----------------------------
# Mode projection / heatmap builders
# -----------------------------
def project_modes_dirichlet(Psi_full, x, L, dx, Nmodes=80):
    # Psi_full shape: (N, Nt). Use interior only.
    xi = x[1:-1]
    Ui = Psi_full[1:-1, :]
    modes = np.arange(1, Nmodes + 1)
    Phi = np.stack([sine_mode(n, xi, L) for n in modes], axis=0)   # (Nmodes, Ni)
    C = dx * (Phi.conj() @ Ui)                                     # (Nmodes, Nt)
    E = np.abs(C)**2
    y_label = "Mode index n (Dirichlet sine)"
    return modes, E, y_label


# -----------------------------
# Main experiment runner
# -----------------------------
def run_nls_bspf(
    L=np.pi,
    N=1024,
    g=1.0,
    Tfinal=40.0,
    Nt_out=1000,
    dt_int_target_factor=1.0,   # dt_int_target = factor * dx^2
    Nmodes=80,
    degree=6,                   # BSPF degree
    use_clustering=True,
    clustering_factor=2.0,
    a=1.0, b=0.6, phase=0.5*np.pi,
    n1=1, n2=2,
):
    if not HAS_BSPF:
        raise ImportError("BSPF library not available. Please install bspf1d.")
    
    # Grid with Dirichlet boundaries
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    t_eval = np.linspace(0.0, Tfinal, Nt_out)
    dt_out = float(t_eval[1] - t_eval[0])
    dt_int_target = float(dt_int_target_factor) * dx**2

    # Create BSPF operator
    bspf_op = bspf1d.from_grid(
        degree=degree,
        n_basis=4*degree,
        num_boundary_points=degree,
        x=x,
        use_clustering=use_clustering,
        clustering_factor=clustering_factor
    )

    # Initial condition
    psi0_full = initial_condition(x, L, a=a, b=b, phase=phase, n1=n1, n2=n2)
    psi = psi0_full.copy()  # Full field including boundaries

    # Storage for full field
    Psi_full = np.empty((N, Nt_out), dtype=np.complex128)
    Psi_full[:, 0] = psi

    def rhs(_, psi_vec):
        """
        RHS for NLS: i*psi_t = -(1/2)*psi_xx - g*|psi|^2*psi
        Rearranged: psi_t = (i/2)*psi_xx - i*g*|psi|^2*psi
        """
        # Enforce Dirichlet BCs before differentiation
        psi_bc = enforce_dirichlet_bc(psi_vec)
        
        # Compute second derivative using BSPF
        d2psi_dx2, _ = bspf_op.differentiate(psi_bc, k=2)
        
        # Enforce Dirichlet BCs on derivative
        d2psi_dx2 = enforce_dirichlet_bc(d2psi_dx2)
        
        # Linear term: (i/2)*psi_xx
        linear_term = (1j / 2.0) * d2psi_dx2
        
        # Nonlinear term: -i*g*|psi|^2*psi
        nonlinear_term = -1j * g * (np.abs(psi_bc)**2) * psi_bc
        
        dpsi_dt = linear_term + nonlinear_term
        
        # Enforce Dirichlet BCs on RHS: dpsi/dt = 0 at boundaries
        dpsi_dt[0] = 0.0
        dpsi_dt[-1] = 0.0
        
        return dpsi_dt

    # Time stepping
    total_steps = 0
    for k in range(Nt_out - 1):
        total_steps += int(np.ceil(dt_out / dt_int_target))

    pbar = tqdm(total=total_steps, desc=f"RK4 BSPF (degree={degree})", unit="step")
    try:
        for k in range(Nt_out - 1):
            nsub = int(np.ceil(dt_out / dt_int_target))
            dt = dt_out / nsub
            for _ in range(nsub):
                k1 = rhs(None, psi)
                k2 = rhs(None, psi + 0.5 * dt * k1)
                k3 = rhs(None, psi + 0.5 * dt * k2)
                k4 = rhs(None, psi + dt * k3)
                psi = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                # Enforce Dirichlet BCs after each step
                psi = enforce_dirichlet_bc(psi)
                pbar.update(1)
            Psi_full[:, k + 1] = psi
    finally:
        pbar.close()

    modes, En, y_label = project_modes_dirichlet(Psi_full, x, L, dx, Nmodes=Nmodes)

    # -----------------------------
    # Plots
    # -----------------------------
    En_log = np.log10(En + 1e-16)

    # selected-mode plot
    plt.figure()
    show = [1, 2, 3, 4, 5, 6]
    for idx in show:
        if idx < En.shape[0]:
            plt.plot(t_eval, En[idx, :], label=f"n={idx}")
    plt.xlabel("t")
    plt.ylabel(r"$E_n(t)$")
    plt.title(f"Selected mode energies (Dirichlet, BSPF degree={degree})")
    plt.legend()
    plt.show()

    # heatmap
    plt.figure()
    extent = [t_eval[0], t_eval[-1], modes[0], modes[-1]]
    plt.imshow(
        En_log,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    plt.xlabel("t")
    plt.ylabel(y_label)
    plt.title(rf"Mode-index heatmap (Dirichlet, BSPF): $\log_{{10}}(E)$")
    plt.colorbar(label=r"$\log_{10}(E)$")
    plt.show()

    # final intensity
    plt.figure()
    plt.plot(x, np.abs(Psi_full[:, -1])**2)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi|^2$")
    plt.title(f"Final intensity |psi|^2 (Dirichlet, BSPF degree={degree})")
    plt.show()

    return x, t_eval, Psi_full, modes, En


if __name__ == "__main__":
    run_nls_bspf(
        L=np.pi,
        N=256,
        g=-1.0,               # focusing if negative; be cautious with dt and resolution
        Tfinal=10,
        Nt_out=1000,
        dt_int_target_factor=0.5,  # dt_int_target = factor * dx^2; reduce (e.g. 0.3) for more stability
        Nmodes=100,
        degree=6,             # BSPF degree
        use_clustering=False,
        a=1.0, b=0.6, phase=0.5*np.pi,
        n1=1, n2=2,
    )
