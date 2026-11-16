"""Two-panel schematic of the wave and localized components of the mixed MMS."""
import argparse
import json
from pathlib import Path
import numpy as np
from plot_bubbles_3d import plt, Patch, POS, NEG, TANH, HERE, domain, gaussians, tanh
from mms import CurvedBox, PlaneWaveMMS
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=HERE/'evidence/nonperiodic_curved_gaussian')
    args = parser.parse_args()
    meta = json.loads((args.dataset/'manifest.json').read_text())
    parameters = {k: np.asarray(v) for k, v in meta['parameters'].items()}
    has_tanh = len(parameters['tanh_amplitudes']) > 0
    wave = PlaneWaveMMS.load(args.dataset/'modes.npz')
    box = CurvedBox()
    uv = np.stack(np.meshgrid(np.linspace(0, 1, 181), np.linspace(0, 1, 181)), axis=-1)
    faces = []
    # Three rear faces keep the interior features visible from the chosen camera.
    for axis, side in ((0, 0), (1, 1), (2, 0)):
        reference = np.empty((*uv.shape[:-1], 3))
        reference[..., axis] = side
        reference[..., [i for i in range(3) if i != axis]] = uv
        xyz = box.map(reference)[0]
        values = wave.evaluate(xyz.reshape(-1, 3), derivatives=False)['velocity'][:, 0].reshape(uv.shape[:-1])
        faces.append((xyz, values))
    limit = max(np.max(np.abs(values)) for _, values in faces)
    norm = Normalize(-limit, limit)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(15, 7.5), facecolor='white')
    ax = fig.add_axes([.015, .22, .43, .68], projection='3d', computed_zorder=False)
    bubbles = fig.add_axes([.56, .22, .43, .68], projection='3d', computed_zorder=False)
    for panel in (ax, bubbles):
        domain(panel)
        # Avoid colliding tick labels at the front corner of the 3D box.
        panel.set_xticks([0, .5]); panel.set_yticks([.5, 1])
    for xyz, values in faces:
        ax.plot_surface(*np.moveaxis(xyz, -1, 0), facecolors=cmap(norm(values)),
                        rstride=1, cstride=1, shade=False, linewidth=0,
                        antialiased=False, alpha=.76, rasterized=True, zorder=0.5)
    gaussians(bubbles, parameters, alpha=.68)
    tanh(bubbles, parameters, combined=True)
    ax.set_title('(a) Plane-wave synthetic turbulence', fontsize=15, pad=16)
    bubbles.set_title('(b) Gaussian mixture'+(' + tanh bubble' if has_tanh else ''), fontsize=15, pad=16)
    cax = fig.add_axes([.465, .34, .012, .40])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('Plane-wave component $u_x$', fontsize=10)
    handles=[Patch(facecolor=POS, label='Positive Gaussian'),Patch(facecolor=NEG, label='Negative Gaussian')]
    if has_tanh:handles.append(Patch(facecolor=TANH, alpha=.5, label='Tanh interface'))
    fig.legend(handles=handles,
               loc='lower center', bbox_to_anchor=(.76, .115), ncol=3, frameon=False, fontsize=9)
    fig.text(.235, .14, 'Wave field sampled on three curved faces', ha='center', fontsize=10, color='#475569')
    fig.text(.5, .065, 'Bubble surfaces show half-amplitude levels.'+(' Dashed rings show the tanh 10% / 90% transition.' if has_tanh else ''),
             ha='center', fontsize=9, color='#475569')
    fig.text(.5, .035, 'Component schematic using the saved MMS parameters; the plane-wave field also fills the interior.',
             ha='center', fontsize=9, color='#475569')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.dataset/f'mixed_3d.{ext}', dpi=200, facecolor='white')
    print(args.dataset/'mixed_3d.png')


if __name__ == '__main__':
    main()
