#!/usr/bin/env python3
"""
Test script for onesoliton_analytical function.
Plots the soliton at a given time for both periodic and Neumann BCs.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'schrodinger_eq'))
from single_soliton import onesoliton_analytical

# Parameters
L = 100.0
nx = 1000
x = np.linspace(0, L, nx, endpoint=False)  # Periodic: doesn't include L

a0 = 1.0
v0 = 1.0
x0 = 50.0
p = 1.0
q = 1.0

# Time to plot
t = 60.0

print("=" * 80)
print("Testing onesoliton_analytical Function")
print("=" * 80)
print(f"Domain: L = {L}, nx = {nx}")
print(f"Soliton: a0 = {a0}, v0 = {v0}, x0 = {x0}")
print(f"Time: t = {t}")
print("=" * 80)

# Compute analytical solutions
print("\nComputing analytical solutions...")
E_periodic = onesoliton_analytical(x, t, a0, v0, x0, p, q, L=L, bc_type='periodic')
E_neumann = onesoliton_analytical(x, t, a0, v0, x0, p, q, L=L, bc_type='neumann')

# Compute positions
x_t_raw = x0 + v0 * t
x_t_periodic = x_t_raw % L
if x_t_periodic < 0:
    x_t_periodic += L

# For Neumann, compute reflected position
x_t_shifted = x_t_raw - x0
x_t_periodic_neumann = x_t_shifted % (2 * L)
if x_t_periodic_neumann < 0:
    x_t_periodic_neumann += 2 * L
if x_t_periodic_neumann > L:
    x_t_neumann = 2 * L - x_t_periodic_neumann
else:
    x_t_neumann = x_t_periodic_neumann
x_t_neumann = x_t_neumann + x0
if x_t_neumann < 0:
    x_t_neumann = -x_t_neumann
if x_t_neumann > L:
    x_t_neumann = 2 * L - x_t_neumann

print(f"x_t_raw = {x_t_raw:.2f}")
print(f"x_t (periodic, wrapped) = {x_t_periodic:.2f}")
print(f"x_t (neumann, reflected) = {x_t_neumann:.2f}")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: Periodic BC - |E|²
ax1 = axes[0, 0]
ax1.plot(x, np.abs(E_periodic)**2, 'b-', linewidth=2, label='|E|²')
ax1.axvline(x_t_periodic, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Soliton center')
ax1.axvline(0, color='k', linestyle=':', alpha=0.5)
ax1.axvline(L, color='k', linestyle=':', alpha=0.5)
ax1.set_xlim(0, L)
ax1.set_ylim(0, 2.5)
ax1.set_xlabel('x')
ax1.set_ylabel('|E|²')
ax1.set_title(f'Periodic BC: |E|² at t={t:.2f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Periodic BC - Re(E) and Im(E)
ax2 = axes[0, 1]
ax2.plot(x, np.real(E_periodic), 'r-', linewidth=2, label='Re(E)', alpha=0.7)
ax2.plot(x, np.imag(E_periodic), 'g-', linewidth=2, label='Im(E)', alpha=0.7)
ax2.axvline(x_t_periodic, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(0, color='k', linestyle=':', alpha=0.5)
ax2.axvline(L, color='k', linestyle=':', alpha=0.5)
ax2.set_xlim(0, L)
ax2.set_ylim(-2.5, 2.5)
ax2.set_xlabel('x')
ax2.set_ylabel('Re(E), Im(E)')
ax2.set_title(f'Periodic BC: Re(E) and Im(E) at t={t:.2f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom left: Neumann BC - |E|²
ax3 = axes[1, 0]
ax3.plot(x, np.abs(E_neumann)**2, 'r-', linewidth=2, label='|E|²')
ax3.axvline(x_t_neumann, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Soliton center')
ax3.axvline(0, color='k', linestyle=':', alpha=0.5)
ax3.axvline(L, color='k', linestyle=':', alpha=0.5)
ax3.set_xlim(0, L)
ax3.set_ylim(0, 2.5)
ax3.set_xlabel('x')
ax3.set_ylabel('|E|²')
ax3.set_title(f'Neumann BC (Reflective): |E|² at t={t:.2f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bottom right: Neumann BC - Re(E) and Im(E)
ax4 = axes[1, 1]
ax4.plot(x, np.real(E_neumann), 'r-', linewidth=2, label='Re(E)', alpha=0.7)
ax4.plot(x, np.imag(E_neumann), 'g-', linewidth=2, label='Im(E)', alpha=0.7)
ax4.axvline(x_t_neumann, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
ax4.axvline(0, color='k', linestyle=':', alpha=0.5)
ax4.axvline(L, color='k', linestyle=':', alpha=0.5)
ax4.set_xlim(0, L)
ax4.set_ylim(-2.5, 2.5)
ax4.set_xlabel('x')
ax4.set_ylabel('Re(E), Im(E)')
ax4.set_title(f'Neumann BC (Reflective): Re(E) and Im(E) at t={t:.2f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Print summary information
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"Periodic BC:")
print(f"  Max |E|² = {np.max(np.abs(E_periodic)**2):.6f}")
print(f"  Peak at x = {x[np.argmax(np.abs(E_periodic)**2)]:.2f}")
print(f"  Expected at x = {x_t_periodic:.2f}")
print(f"  |E|² at x=0: {np.abs(E_periodic[0])**2:.6f}")
print(f"  |E|² at x=L-dx: {np.abs(E_periodic[-1])**2:.6f}")
print(f"  Phase at x=0: {np.angle(E_periodic[0]):.6f}")
print(f"  Phase at x=L-dx: {np.angle(E_periodic[-1]):.6f}")

print(f"\nNeumann BC:")
print(f"  Max |E|² = {np.max(np.abs(E_neumann)**2):.6f}")
print(f"  Peak at x = {x[np.argmax(np.abs(E_neumann)**2)]:.2f}")
print(f"  Expected at x = {x_t_neumann:.2f}")

# Check periodic boundary continuity
error_periodic = np.abs(E_periodic[0] - E_periodic[-1])
print(f"\nPeriodic BC continuity check:")
print(f"  |E(0) - E(L-dx)| = {error_periodic:.6e}")
print(f"  Relative error: {error_periodic / np.abs(E_periodic[0]):.6e}")

# Save plot
output_file = 'onesoliton_analytical_test.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

plt.show()

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
