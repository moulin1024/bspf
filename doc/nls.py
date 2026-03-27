import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from tqdm.auto import tqdm


# -----------------------------
# Basis functions for diagnostics
# -----------------------------
def sine_mode(n, x, L):
    # Dirichlet eigenmode on (0,L): sqrt(2/L) sin(n pi x/L), n>=1
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


# -----------------------------
# FD Laplacian for Dirichlet BC (2nd order)
# -----------------------------
def make_D2_dirichlet(N, dx):
    """
    Dirichlet BC: psi(0)=psi(L)=0.
    We evolve interior unknowns u = psi[1:-1] of length N-2.
    """
    Ni = N - 2
    main = (-2.0 / dx**2) * np.ones(Ni)
    off  = ( 1.0 / dx**2) * np.ones(Ni - 1)
    return diags([off, main, off], offsets=[-1, 0, 1], format="csc")


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
def run_nls_fd(
    L=np.pi,
    N=1024,
    g=-1.0,
    Tfinal=40.0,
    Nt_out=1000,
    dt_int_target_factor=1.0,   # dt_int_target = factor * dx^2
    Nmodes=80,
    a=1.0, b=0.6, phase=0.5*np.pi,
    n1=1, n2=2,
):
    # Grid with Dirichlet boundaries
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    t_eval = np.linspace(0.0, Tfinal, Nt_out)
    dt_out = float(t_eval[1] - t_eval[0])
    dt_int_target = float(dt_int_target_factor) * dx**2

    # Operator and state storage
    D2 = make_D2_dirichlet(N, dx)
    Lin = (1j / 2.0) * D2

    psi0_full = initial_condition(x, L, a=a, b=b, phase=phase, n1=n1, n2=n2)
    u = psi0_full[1:-1].copy()            # interior unknowns
    Ni = u.size

    U = np.empty((Ni, Nt_out), dtype=np.complex128)
    U[:, 0] = u

    def rhs(_, uu):
        return Lin @ uu - 1j * g * (np.abs(uu)**2) * uu

    # Time stepping
    total_steps = 0
    for k in range(Nt_out - 1):
        total_steps += int(np.ceil(dt_out / dt_int_target))

    pbar = tqdm(total=total_steps, desc="RK4 (Dirichlet)", unit="step")
    try:
        for k in range(Nt_out - 1):
            nsub = int(np.ceil(dt_out / dt_int_target))
            dt = dt_out / nsub
            for _ in range(nsub):
                k1 = rhs(None, u)
                k2 = rhs(None, u + 0.5 * dt * k1)
                k3 = rhs(None, u + 0.5 * dt * k2)
                k4 = rhs(None, u + dt * k3)
                u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                pbar.update(1)
            U[:, k + 1] = u
    finally:
        pbar.close()

    # Reconstruct full field for plotting/projection
    Psi_full = np.zeros((N, Nt_out), dtype=np.complex128)
    Psi_full[1:-1, :] = U
    # boundaries remain zero

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
    plt.title("Selected mode energies (Dirichlet)")
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
    plt.title(r"Mode-index heatmap (Dirichlet): $\log_{10}(E)$")
    plt.colorbar(label=r"$\log_{10}(E)$")
    plt.show()

    # final intensity
    plt.figure()
    plt.plot(x, np.abs(Psi_full[:, -1])**2)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi|^2$")
    plt.title("Final intensity |psi|^2 (Dirichlet)")
    plt.show()

    return x, t_eval, Psi_full, modes, En


if __name__ == "__main__":
    run_nls_fd(
        L=np.pi,
        N=512,
        g=-1.0,               # focusing if negative; be cautious with dt and resolution
        Tfinal=10.0,
        Nt_out=1000,
        dt_int_target_factor=1.0,  # dt_int_target = factor * dx^2; reduce (e.g. 0.3) for more stability
        Nmodes=100,
        a=1.0, b=0.6, phase=0.5*np.pi,
        n1=1, n2=2,
    )
