# Plotting script for fig1
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('data/fig1_data.npz')

# Extract data
domain = data['domain']
x = data['x']
y_spline = data['y_spline']
x_fine = data['x_fine']
y_fine = data['y_fine']
xi_fine = data['xi_fine']
x_sample = data['x_sample']
xi_sample = data['xi_sample']
grid_sizes = data['grid_sizes']
errors_bfpsm = data['errors_bfpsm']
errors_bfpsm_orig = data['errors_bfpsm_orig']
errors_cheb_orig = data['errors_cheb_orig']
errors_fd = data['errors_fd']
x_bfpsm_error = data['x_bfpsm_error']
error_bfpsm_error = data['error_bfpsm_error']
x_cheb_error = data['x_cheb_error']
error_cheb_error = data['error_cheb_error']
x_fd_error = data['x_fd_error']
error_fd_error = data['error_fd_error']
norm_exact_bfpsm_selected = data['norm_exact_bfpsm_selected']
norm_exact_orig_selected = data['norm_exact_orig_selected']
norm_exact_cheb_selected = data['norm_exact_cheb_selected']
selected_N = int(data['selected_N'])
target_N = int(data['target_N'])

# Set up global plotting parameters
plt.rcParams.update({
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 24,
    'axes.grid': True,
    'grid.alpha': 0.5
})

# Create figure with custom grid layout (2 rows, 2 columns)
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ==== Panel (a): Original function ====
ax1 = fig.add_subplot(gs[0, 0])
# Interpolate y_spline from original grid x to fine grid x_fine
y_spline_fine = np.interp(x_fine, x, y_spline)

# Plot y_fine on left y-axis
color1 = default_colors[0]
color2 = default_colors[1]
ax1.plot(x_fine, y_fine, '-', color=color1, linewidth=2, label='$f(x)$')
ax1.plot(x_fine, y_spline_fine, '--', color=color2, linewidth=2, label='B-spline')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$f(x)$')
ax1.legend(loc='best', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.set_title('(a)', loc='left', x=-0.1, fontsize=24, fontweight='bold')

# ==== Panel (b): Mapping function ====
ax2 = fig.add_subplot(gs[0, 1])

# Primary plot - mapping function
ax2.plot(xi_fine, x_fine, '-', color=default_colors[0], linewidth=2, label='BSPF')

# Show some grid points
ax2.plot(xi_sample, np.zeros_like(xi_sample), 'o', color=default_colors[0], markersize=2, alpha=0.8)

# Chebyshev-equivalent mapping function xi(x) on uniform x
s_fine = (x_fine - domain[0]) / (domain[1] - domain[0])
xi_cheb_map = 0.5 * (domain[0] + domain[1]) + 0.5 * (domain[1] - domain[0]) * np.cos(np.pi * (1.0 - s_fine))
ax2.plot(xi_cheb_map, x_fine, '-', color=default_colors[1], linewidth=2, label='Cheb.')

ax2.plot(x_fine, x_fine, '--', color=default_colors[2], alpha=0.7, linewidth=2, label='FD (4)')

# Draw arrows from grid points to mapping
for i in range(len(x_sample)):
    ax2.annotate('', xy=(xi_sample[i], 0), xytext=(xi_sample[i], x_sample[i]),
                 arrowprops=dict(arrowstyle='-', color=default_colors[0], alpha=0.8, lw=1))

ax2.set_ylabel('$x$')
ax2.set_xlabel('$\zeta(x)$')
ax2.legend(loc='upper left', fontsize=16)
ax2.grid(True, alpha=0.3)
ax2.set_title('(b)', loc='left', x=-0.1, fontsize=24, fontweight='bold')

# ==== Panel (c): Error distribution ====
ax4 = fig.add_subplot(gs[1, 0])

# Plot error distributions (normalized to relative errors)
ax4.plot(x_bfpsm_error, np.abs(error_bfpsm_error)/norm_exact_bfpsm_selected, '-', 
         label='BSPF', color=default_colors[0], linewidth=2)
ax4.plot(x_cheb_error, np.abs(error_cheb_error)/norm_exact_cheb_selected, '-', 
         label='Cheb.', color=default_colors[1], linewidth=2)
ax4.plot(x_fd_error, np.abs(error_fd_error)/norm_exact_orig_selected, '-', 
         label='FD (4)', color=default_colors[2], linewidth=2)

ax4.set_xlabel('$x$', fontsize=24)
ax4.set_ylabel('Rel. $L_\infty$ Error', fontsize=24)
ax4.set_title('(c)', loc='left', x=-0.1, fontsize=24, fontweight='bold')
ax4.set_yscale('log')
ax4.set_ylim(1e-16, 0.9*1e5)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=16, ncol=3, loc='upper right')

# ==== Panel (d): Convergence study ====
ax3 = fig.add_subplot(gs[1, 1])

# Plot convergence
ax3.loglog(grid_sizes, errors_bfpsm, '.-', label='BSPF', color=default_colors[0], linewidth=2)
ax3.loglog(grid_sizes, errors_cheb_orig, '.-', label='Cheb.', color=default_colors[1], linewidth=2)
ax3.loglog(grid_sizes, errors_fd, '.-', label='FD (4)', color=default_colors[2], linewidth=2, markersize=4)

# Draw vertical line at target_N
ax3.plot([target_N, target_N], [1e-16, 1e7], '--', color='gray', linewidth=2.5)
ax3.text(target_N, 2*1e-16, f'(c)', color='gray', fontsize=24)

ax3.set_xlabel('$N$', fontsize=24)
ax3.set_ylabel('Rel. $L_\infty$ Error', fontsize=24)
ax3.set_title('(d)', loc='left', x=-0.1, fontsize=24, fontweight='bold')
ax3.set_ylim(1e-16, 0.9*1e5)
ax3.grid(True)
ax3.legend(fontsize=16, ncol=3, loc='upper right')

plt.tight_layout()
plt.savefig('figs/fig1.pdf', dpi=300, bbox_inches='tight')
print("Saved figure to figs/fig1.pdf")
