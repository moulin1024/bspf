"""Draw the exact saved curved-box Poisson MMS used in the Leonardo comparison."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from plot_bubbles_3d import plt, Patch, POS, NEG, HERE, domain, gaussians
from mms import CurvedBox, PlaneWaveMMS

DATA = HERE / "evidence/nonperiodic_curved_gaussian/modes.npz"
OUT = HERE / "evidence/comparison_leonardo_to257"


def main():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == (
        "c9119da1d12d3bbca43497bcddc5f5b92baf6611cdd047c49866e025812c9949")
    with np.load(DATA) as saved:
        parameters = {k: saved[k] for k in
                      ("gaussian_centers", "gaussian_sigmas", "gaussian_amplitudes")}
        assert len(parameters["gaussian_amplitudes"]) == 8
        assert len(saved["tanh_amplitudes"]) == 0
    wave, box = PlaneWaveMMS.load(DATA), CurvedBox()
    uv = np.stack(np.meshgrid(np.linspace(0, 1, 121), np.linspace(0, 1, 121)), axis=-1)
    faces = []
    for axis, side in ((0, 0), (1, 1), (2, 0)):
        ref = np.empty((*uv.shape[:-1], 3))
        ref[..., axis] = side
        ref[..., [i for i in range(3) if i != axis]] = uv
        xyz, _, determinant = box.map(ref)
        assert np.all(determinant > 0)
        values = wave.evaluate(xyz, derivatives=False)["velocity"][..., 0]
        faces.append((xyz, values))
    limit = max(np.abs(v).max() for _, v in faces)
    norm, cmap = Normalize(-limit, limit), plt.get_cmap("coolwarm")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 7.2), facecolor="white")
    axes = [fig.add_axes([left, .28, .29, .56], projection="3d",
                        computed_zorder=False) for left in (.015, .355, .695)]
    for ax in axes:
        domain(ax)
        ax.set(xlim=(-.08, 1.08), ylim=(-.08, 1.08), zlim=(-.08, 1.08))
        ax.set_xticks([0, .5]); ax.set_yticks([.5, 1])
    for ax in (axes[0], axes[2]):
        for xyz, values in faces:
            ax.plot_surface(*np.moveaxis(xyz, -1, 0),
                            facecolors=cmap(norm(values)), rstride=1, cstride=1,
                            shade=False, linewidth=0, antialiased=False,
                            alpha=.8 if ax is axes[0] else .48, rasterized=True,
                            zorder=.5)
    for ax in axes[1:]:
        gaussians(ax, parameters, alpha=.85)
    for ax, title in zip(axes, ("Synthetic turbulence", "Eight Gaussian blobs",
                                "Combined MMS schematic")):
        ax.set_title(title, pad=12, fontsize=14)
    fig.suptitle("Manufactured Poisson solution on a curved Q2 box", fontsize=19, y=.96)
    fig.text(.335, .57, "+", fontsize=30, ha="center")
    fig.text(.675, .57, "=", fontsize=30, ha="center")
    cax = fig.add_axes([.075, .225, .19, .018])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(r"Plane-wave component $u_x$", fontsize=10)
    fig.legend(handles=[Patch(facecolor=POS, label="Positive Gaussian"),
                        Patch(facecolor=NEG, label="Negative Gaussian")],
               loc="center", bbox_to_anchor=(.5, .21), frameon=False, ncol=2)
    fig.text(.84, .21, "Exact geometry and saved blob parameters", ha="center", fontsize=10)
    fig.text(.5, .125,
             r"$\phi(\mathbf{x})=u_x^{\mathrm{wave}}(\mathbf{x})"
             r"+\sum_{j=1}^{8} A_j\exp\!\left[-\frac{1}{2}"
             r"\sum_{d=1}^{3}\left(\frac{x_d-c_{jd}}{\sigma_{jd}}\right)^2\right]"
             r",\qquad -\Delta\phi=f,\quad \phi|_{\partial\Omega}=\phi_{\mathrm{exact}}$",
             ha="center", fontsize=15)
    fig.text(.5, .065,
             "Colors on three curved faces show the wave component; the field also fills the interior.",
             ha="center", fontsize=10, color="#475569")
    fig.text(.5, .035,
             "Ellipsoids mark each Gaussian's half-amplitude surface, not solid inclusions. Six nonperiodic Dirichlet faces.",
             ha="center", fontsize=10, color="#475569")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"mms_schematic.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
