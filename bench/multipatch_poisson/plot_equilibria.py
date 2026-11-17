"""Plot meshing inputs, not an extracted or validated computational boundary."""
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path.home() / 'genex'
cases = [('TCV-X21', root/'run_genex/TCV-X21.nc', .74),
         ('ITER-q10', root/'tools/iter_strong_scaling/inputs/ITER_q10.nc', .90)]
fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
for ax, (name, filename, rmin) in zip(axes, cases):
    with h5py.File(filename) as f:
        g, limits = f['Magnetic_geometry'], f['Psi_limits']
        scalar = lambda attrs, key: float(np.asarray(attrs[key]).ravel()[0])
        R, Z, psi = g['R'][:], -g['Z'][:][::-1], g['psi'][:][::-1]
        p0 = scalar(limits.attrs, 'psi_axis')
        px = scalar(limits.attrs, 'psi_seperatrix')
        rho2 = (psi-p0)/(px-p0)
        ax.contour(R, Z, rho2, levels=[rmin*rmin], colors=['#009e73'], linewidths=2)
        ax.contour(R, Z, rho2, levels=[1.], colors=['#0072b2'], linewidths=1.5)
        ax.contour(R, Z, rho2, levels=[scalar(limits.attrs, 'rho_max')**2],
                   colors=['#777777'], linestyles=':', linewidths=1)
        for key, color, style in [('divertor_polygon', '#222222', '-'),
                                  ('exclusion_polygon', '#d55e00', '--')]:
            x, y = f[key]['R_points'][:], -f[key]['Z_points'][:]
            if name == 'TCV-X21' and key == 'exclusion_polygon':
                x = [.675, 1.26250217, 1.26250217, .675, .675]
                y = [.84, .84, -.55, -.55, .84]
            ax.plot(x, y, color=color, ls=style, lw=1.2)
        scale = scalar(g.attrs, 'magnetic_axis_R')
        ax.plot(scalar(g.attrs, 'x_point_R')/scale,
                -scalar(g.attrs, 'x_point_Z')/scale, 'x', color='#0072b2', ms=8)
        ax.set_title(f'{name}: input geometry, flipped Z\ncore cutoff rho = {rmin}')
        ax.set_xlim(R[0], R[-1]); ax.set_ylim(Z[0], Z[-1])
        ax.set_aspect('equal'); ax.set_xlabel('Normalized R'); ax.set_ylabel('Normalized Z')
        ax.grid(alpha=.15)
handles = [Line2D([], [], color=c, ls=ls, label=label) for c, ls, label in
           [('#222222','-', 'Wall/divertor polygon'), ('#d55e00','--','Exclusion envelope'),
            ('#009e73','-', 'Core-cutoff flux contours'), ('#0072b2','-', 'Separatrix'),
            ('#777777',':', 'Global outer flux limit')]]
fig.legend(handles=handles, loc='outside lower center', ncol=3)
fig.suptitle('Boundary ingredients for patch design — not yet the classified active domain')
fig.savefig(Path(__file__).with_name('equilibria_meshing_inputs.png'), dpi=180)
