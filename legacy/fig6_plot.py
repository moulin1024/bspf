import numpy as np
import matplotlib.pyplot as plt

# # Data from provided table
# N = np.array([100, 200, 300, 400, 500, 600, 700, 800])
# E_bspf = np.array([
#      2.596242e-02, 1.509325e-03, 3.108401e-05,  4.045169e-06, 7.768910e-07,
#     2.183405e-07, 7.545224e-08, 2.940068e-08
# ])
# E_fd = np.array([
#     3.524870e-01, 4.689446e-02, 1.243810e-02, 4.516010e-03, 1.985511e-03,
#     9.938997e-04, 5.488871e-04, 3.271600e-04
# ])

data = np.load('data/mesh_convergence.npz')
N = data['nx']
E_bspf = data['errors_linf_bspf']
E_fd = data['errors_linf_fd']
T_bspf = data['walltime_bspf']
T_fd = data['walltime_fd']

# Append new results
new_N = np.array([1100, 1200, 1300, 1400])
new_E_bspf = np.array([3.701658e-09, 1.095697e-09, 6.172621e-10, 1.930932e-10])
new_E_fd = np.array([9.369253e-05, 6.643500e-05, 4.836501e-05, 3.604829e-05])
new_T_bspf = np.array([632.85, 841.91,  1144.80 , 1505.80])
new_T_fd = np.array([430.81, 559.05, 725.24, 923.34])

# Concatenate with existing data
N = np.concatenate([N, new_N])
E_bspf = np.concatenate([E_bspf, new_E_bspf])
E_fd = np.concatenate([E_fd, new_E_fd])
T_bspf = np.concatenate([T_bspf, new_T_bspf])
T_fd = np.concatenate([T_fd, new_T_fd])

# Use the last data point as starting point for reference lines
N_last = N[-1]
E_bspf_last = E_bspf[-1]
print(E_bspf_last)
E_fd_last = E_fd[-1]

# Create reference lines going backward from last point
# For FD: E = C * N^-4, so C = E_fd_last * (N_last)^4
C_fd_ref = 1.2*E_fd_last * (float(N_last) ** 4)
E_fd_ref = C_fd_ref * (N[-10:] ** -4.0)

# For BSPF: E = C * N^-8, so C = E_bspf_last * (N_last)^8
C_bspf_ref = 1.2*E_bspf_last * (float(N_last)**8)
print(C_bspf_ref)
E_bspf_ref = C_bspf_ref * (N[-10:] ** -8.0)
print(E_bspf_ref)

# Fit power law to wall time data: T = C * N^α
# Use log-log linear fit: log(T) = log(C) + α * log(N)
# Fit BSPF wall time
log_N = np.log(N)
log_T_bspf = np.log(T_bspf)
log_T_fd = np.log(T_fd)

# Fit linear regression in log space: y = a*x + b, where y=log(T), x=log(N)
# This gives: log(T) = α*log(N) + log(C), so T = C * N^α
coeff_bspf = np.polyfit(log_N, log_T_bspf, 1)
alpha_bspf = coeff_bspf[0]  # exponent
C_bspf_time = np.exp(coeff_bspf[1])  # coefficient

coeff_fd = np.polyfit(log_N, log_T_fd, 1)
alpha_fd = coeff_fd[0]  # exponent
C_fd_time = np.exp(coeff_fd[1])  # coefficient

# Generate reference lines based on fitted scaling
T_bspf_ref = C_bspf_time * (N ** alpha_bspf)
T_fd_ref = C_fd_time * (N ** alpha_fd)
# 4. Create the plot with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# plt.figure(figsize=(16, 6))

# Set up global plotting parameters
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,  # Increased x-axis tick label size
    'ytick.labelsize': 20,  # Increased y-axis tick label size
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'axes.grid': True,
    'grid.alpha': 0.5
})

# Left subplot: Error convergence
ax1.loglog(N, E_bspf, 'o-', label='BSPF', linewidth=2)
ax1.loglog(N, E_fd, 's-', label='FD-4', linewidth=2)

# Reset colors for reference lines
ax1.set_prop_cycle(None)

# Plot reference lines (going backward from last point)
ax1.loglog(N[-10:], E_bspf_ref, '--', label=r'$O(N^{-8})$', linewidth=2)
ax1.loglog(N[-10:], E_fd_ref, '--', label=r'$O(N^{-4})$', linewidth=2)

ax1.set_xlabel(r'$N$', fontsize=20)
ax1.set_ylabel(r'Rel. $L_\infty$ Error', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick label size
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.set_title('(a)', loc='left', x = -0.1, fontsize=24, fontweight='bold')

# Right subplot: Wall time
ax2.loglog(N, T_bspf, 'o', label='BSPF', linewidth=2)
ax2.loglog(N, T_fd, 's', label='FD-4', linewidth=2)

# Reset colors for reference lines
ax2.set_prop_cycle(None)

# Plot fitted reference lines
ax2.loglog(N, T_bspf_ref, '--', label=rf'BSPF: $O(N^{{{alpha_bspf:.1f}}})$', linewidth=2)
ax2.loglog(N, T_fd_ref, '--', label=rf'FD-4: $O(N^{{{alpha_fd:.1f}}})$', linewidth=2)

ax2.set_xlabel(r'$N$', fontsize=20)
ax2.set_ylabel(r'Wall Time (s)', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick label size
ax2.legend()
ax2.set_title('(b)', loc='left', x = -0.1, fontsize=24, fontweight='bold')
ax2.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()

# Save and show
plt.savefig('figs/fig6.pdf',bbox_inches='tight')
# plt.show()

