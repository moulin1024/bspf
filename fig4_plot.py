import matplotlib.pyplot as plt
import numpy as np

data = np.load('data/bspf1d_benchmark_nu0.01.npz')
# mesh_sizes = data['mesh_sizes']
linf_errors_chebyshev = data['linf_errors_chebyshev_BDF']
linf_errors_RK45 = data['linf_errors_BSPF_RK45']
linf_errors_cheb_RK45 = data['linf_errors_Chebyshev_RK45']
timings_chebyshev = data['timings_chebyshev_BDF']
time_measure_RK45 = data['time_measure_BSPF_RK45']
time_measure_cheb_RK45 = data['time_measure_Chebyshev_RK45']

mesh_sizes = np.arange(100, 1001, 100)
mesh_sizes_short = [100, 200, 300]
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

plt.figure(figsize=(16, 6))

# Set up global plotting parameters
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'axes.grid': True,
    'grid.alpha': 0.5
})

plt.subplot(1, 2, 1)
plt.loglog(mesh_sizes, linf_errors_chebyshev, 'o--', label='Chebyshev+BDF', linewidth=2, markersize=8, color=default_colors[1])
plt.loglog(mesh_sizes, linf_errors_RK45, 's-', label='BSPF+RK45', linewidth=2, markersize=8, color=default_colors[0])
plt.loglog(mesh_sizes_short, linf_errors_cheb_RK45, 's-', label='Chebyshev+RK45', linewidth=2, markersize=8, color=default_colors[1])
plt.xlabel('$N$')
plt.ylabel('$\|Error\|_\infty$')
plt.title('(a)', loc='left', x = -0.1, fontsize=24, fontweight='bold')
plt.grid(True, which='both')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.loglog(mesh_sizes, timings_chebyshev, 'o--', label='Chebyshev+BDF', linewidth=2, markersize=8, color=default_colors[1])
plt.loglog(mesh_sizes, time_measure_RK45, 's-', label='BSPF+RK45', linewidth=2, markersize=8, color=default_colors[0])
plt.loglog(mesh_sizes_short, time_measure_cheb_RK45, 's-', label='Chebyshev+RK45', linewidth=2, markersize=8, color=default_colors[1])
plt.xlabel('$N$')
plt.ylabel('Wall time [s]')
plt.title('(b)', loc='left', x = -0.1, fontsize=24, fontweight='bold')
plt.grid(True, which='both')
plt.legend(loc='best')
plt.ylim(1e-1, 2*1e2)
plt.tight_layout()
plt.savefig(f'figs/fig4.pdf', dpi=300, bbox_inches='tight')
