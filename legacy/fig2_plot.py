import numpy as np
import matplotlib.pyplot as plt


def load_timing_data(path: str = "data/timing.txt"):
    """
    Load timing data from a text file with '|' separators.

    Expected format (with header and separator line):
      N(ref) |  N(BSPF) |   BSPF ms |  Cheb ms |
      -----------------------------------------
          65 |       64 |     0.022 |    0.025 |
          ...
    """
    data = np.genfromtxt(
        path,
        delimiter="|",
        skip_header=2,
        autostrip=True,
    )

    # Columns after splitting on '|' are:
    # 0: N(ref), 1: N(BSPF), 2: BSPF ms, 3: Cheb ms, (last empty column may appear)
    n_ref = data[:, 0]
    n_bspf = data[:, 1]
    bspf_ms = data[:, 2]
    cheb_ms = data[:, 3]
    return n_ref, n_bspf, bspf_ms, cheb_ms


def plot_timing(path: str = "data/timing.txt"):
    n_ref, n_bspf, bspf_ms, cheb_ms = load_timing_data(path)

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
    plt.figure(figsize=(9, 6))
    # Use reference N on x-axis so both methods are comparable
    plt.loglog(n_ref, bspf_ms, "o-", label="BSPF", linewidth=2, markersize=6)
    plt.loglog(n_ref, cheb_ms, "s-", label="Chebyshev", linewidth=2, markersize=6)

    anchor_idx = len(n_ref) - 1  # middle point
    ref_shape = n_ref * np.log(n_ref)
    scale = bspf_ms[anchor_idx] / ref_shape[anchor_idx]
    nlogn_ms = scale * ref_shape
    plt.loglog(n_ref, nlogn_ms, "--", color='gray', label="N log N", linewidth=2)

    plt.xlabel("N (grid points)")
    plt.ylabel("Time (ms)")
    # plt.title("Performance Benchmark: BSPF vs Chebyshev", fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(1e-2, 1e0)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('figs/fig2.pdf', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_timing()


