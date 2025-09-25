from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
data_bfpsm = np.load("data/tsunami_bfpsm_dctdst_201.npz")

eta1, M1, N1 = data_bfpsm['eta'], data_bfpsm['M'], data_bfpsm['N']
x1 = np.linspace(0, 100, eta1.shape[1])

slice_path = Path('data/eta_row100_slices.npz')
slice_path.parent.mkdir(exist_ok=True)

with np.load(slice_path) as slice_data:
    eta2_slice = slice_data['eta2_slice']
    eta3_slice = slice_data['eta3_slice']
    eta4_slice = slice_data['eta4_slice']
    eta5_slice = slice_data['eta5_slice']

x2 = np.linspace(0, 100, eta2_slice.size)
x3 = np.linspace(0, 100, eta3_slice.size)
x4 = np.linspace(0, 100, eta4_slice.size)
x5 = np.linspace(0, 100, eta5_slice.size)

# For top row plots
x_big = np.linspace(0, 100, 1601)
h = 50 - 25*np.tanh((x_big - 50.) / 10.)
phi1 = np.exp(-((x_big - 50)**2 / 10))

extent = [0, 100, 0, 100]
cmap = 'Blues_r'

# =========================
# Styling
# =========================
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'lines.linewidth': 1.5,
    'axes.grid': True
})

# =========================
# Figure (normal 2x2, no spanning)
# =========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # ----- (a) Seafloor depth -----
ax = axes[0, 0]
ax.plot(x_big, -h, 'k-', label='Seafloor')
# ax.plot(x_big, phi1, '--', label='Initial condition')
ax.set_title('(a)', loc='left',x = -0.1,y=1.05,fontsize=30,fontweight='bold')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('Depth (m)')
ax.set_ylim(-80,-20)
ax.legend(loc='lower right')


# ----- (b) Initial condition -----
ax = axes[0, 1]
ax.plot(x_big[:-1], phi1[:-1], label='$t=0$ s')
ax.plot(x1[:-1], eta1[100, :-1], label='$t=2$ s')
ax.set_title('(b)', loc='left',x = -0.1,y=1.05,fontsize=30,fontweight='bold')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$\eta$ (m)')
# ax.set_ylim(-1,1)
ax.legend(loc='upper left')

# ----- (c) Field with colorbar (same as your (a)) -----
ax = axes[1,0]
im = ax.imshow(eta1, extent=extent, interpolation='spline36',
               cmap=cmap, vmin=-0.81, vmax=0.81, origin='upper')
ax.plot([72, 92], [50, 50], 'k--', label='Sample line', linewidth=1.5)
ax.set_title('(c)', loc='left',x = -0.275,y=1.05,fontsize=30,fontweight='bold')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
ax.legend(loc="upper right")
# Attach colorbar to this subplot only
cbar = fig.colorbar(im, ax=ax)
# If you want label on top:
cbar.ax.set_xlabel(r'$\eta$ (m)')
cbar.ax.xaxis.set_label_position('top')
# cbar.ax.xaxis.label.set_y(1.12)

# ----- (d) Line comparisons (same as your (b)) -----
ax = axes[1,1]
ax.plot(x1, eta1[100, :], 'o', label='BSPF (N = 201)')
ax.plot(x3, eta3_slice, '-', label='FinDiff-2 (N = 201)')
ax.plot(x4, eta4_slice, '-', label='FinDiff-2 (N = 401)')
ax.plot(x5, eta5_slice, '-', label='FinDiff-2 (N = 801)')
ax.plot(x2, eta2_slice, '-', label='FinDiff-2 (N = 1601)')
ax.set_xlim(72, 92)
ax.set_ylim(-0.6, 1.0)
ax.set_title('(d)', loc='left',x = -0.15,y=1.05,fontsize=30,fontweight='bold')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel(r'$\eta$ (m)')
ax.legend(loc="upper left")

fig.tight_layout()
plt.savefig('figs/fig7.pdf', dpi=300, bbox_inches="tight")
# plt.show()
# plt.savefig('tsunami_postprocessing.png', dpi=300, bbox_inches="tight")
