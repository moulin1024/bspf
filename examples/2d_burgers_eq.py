#!/usr/bin/env python3
"""
2-D viscous Burgers on a periodic box  –  XY-indexing everywhere
----------------------------------------------------------------
• spatial derivatives by FFT (pseudo-spectral, 2/3 dealiasing)
• adaptive time integration with scipy.solve_ivp
• initial and final vector-field plots (pseudocolour + streamlines)
"""

import numpy as np
from numpy.fft import rfftn, irfftn, fftfreq, rfftfreq
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from bspf import bspf2d

# ------------------- domain & physical parameters ---------------------------
L   = 2.0 * np.pi        # box length  (same in x and y)
N   = 128                # grid points per direction  (even)
nu  = 1e-2               # viscosity
t_end = 2.0              # final simulation time

dx = L / N               # grid spacing                  # avoid 0/0 in viscous term

# ------------------- helpers ------------------------------------------------

def pack(u, v):               # (N,N)×2  → flat vector
    return np.concatenate([u.ravel(), v.ravel()])

def unpack(q):                # flat vector → two (N,N) fields
    u = q[:N*N].reshape(N, N)
    v = q[N*N:].reshape(N, N)
    return u, v

# ------------------- RHS supplied to solve_ivp ------------------------------
def rhs(t, q, m2d):
    u, v = unpack(q)

    ux = m2d.partial_dx(u, order=1)
    uy = m2d.partial_dy(u, order=1)
    vx = m2d.partial_dx(v, order=1)
    vy = m2d.partial_dy(v, order=1)
    laplacian_u = m2d.laplacian(u)
    laplacian_v = m2d.laplacian(v)

    # Burgers RHS
    du = -(u*ux + v*uy) + nu * laplacian_u
    dv = -(u*vx + v*vy) + nu * laplacian_v

    # ---- Dirichlet BC: keep the rim fixed -----------------
    du[[0, -1], :] = 0.0        # bottom & top
    du[:, [0, -1]] = 0.0        # left   & right
    dv[[0, -1], :] = 0.0
    dv[:, [0, -1]] = 0.0

    rhs_value = pack(du, dv)
    return rhs_value

# ------------------- initial condition --------------------------------------
x = np.linspace(0.0, L, N)
y = x.copy()
X, Y = np.meshgrid(x, y)          # (N, N)

m2d = bspf2d.from_grids(x=x, y=y, degree_x=9, degree_y=9, n_basis_x=40, n_basis_y=40)

u0 = -np.cos(0.5*Y) * np.sin(0.5*X)
v0 =  np.cos(0.5*X) * np.sin(0.5*Y)
speed0 = np.sqrt(u0**2 + v0**2)

# ------------------- time integration ---------------------------------------
sol = solve_ivp(rhs, (0.0, t_end), pack(u0, v0),args=(m2d,), 
                method='RK45',
                t_eval=np.linspace(0.0, t_end, 101),
                rtol=1e-6, atol=1e-9,
                max_step=0.01)

u_final, v_final = unpack(sol.y[:, -1])
speed_final = np.sqrt(u_final**2 + v_final**2)

# ------------------- plotting ------------------------------------------------
step = 4                        # down-sampling for arrows
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

# -- initial field -----------------------------------------------------------
pcm0 = ax0.pcolormesh(X, Y, speed0,vmin=0,vmax=1, shading='nearest')
ax0.streamplot(X, Y, u0, v0,
               density=1.0, color='k', linewidth=1.0)
ax0.set_title('Initial field  $t=0$')
ax0.set_xlabel('x');  ax0.set_ylabel('y');  ax0.set_aspect('equal')
fig.colorbar(pcm0, ax=ax0, label='|u|')

# -- final field -------------------------------------------------------------
pcm1 = ax1.pcolormesh(X, Y, speed_final,vmin=0,vmax=1, shading='nearest')
strm  = ax1.streamplot(X, Y, u_final, v_final,
                       density=1.0,
                       color="k",        # colour by magnitude!
                       linewidth=1.0)
ax1.set_title(f'Final field  $t={t_end}$')
ax1.set_xlabel('x');  ax1.set_ylabel('y');  ax1.set_aspect('equal')
fig.colorbar(strm.lines, ax=ax1, label='|u|')

plt.tight_layout()
plt.show()
