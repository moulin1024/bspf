"""Plot the saved five-solver Leonardo comparison through 257^3.

Run with Python, NumPy and Matplotlib; no solver installations are needed.
"""
import csv
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bspf-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "evidence/comparison_leonardo_to257"
STYLES = [
    ("bspf", "BSPF Galerkin", "#0072B2", "o", "-"),
    ("chebyshev", "Chebyshev Galerkin", "#CC79A7", "s", "--"),
    ("FC-Gram", "FC-Gram", "#009E73", "^", "-"),
    ("MFEM/Hypre", "MFEM/Hypre (p=8)", "#D55E00", "D", "-"),
    ("NekRS", "NekRS (p=8)", "#303030", "v", "--"),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bspf-dir',type=Path,help='Replace BSPF rows with a complete seven-resolution sweep, then redraw')
    args=parser.parse_args()
    with (OUT / "comparison.csv").open(newline="") as handle:
        reader=csv.DictReader(handle);fieldnames=reader.fieldnames
        rows = list(reader)
    expected = [17, 33, 65, 97, 129, 193, 257]
    if args.bspf_dir:
        replacements={}
        for n in expected:
            q=max(96,(3*n+1)//2)
            path=args.bspf_dir/f'n{n}_q{q}/result.json'
            result=json.loads(path.read_text())
            assert result['n']==n and result['q']==q
            assert result['boundary_error']==0
            assert all(r['relative_residual']<1e-11 for r in result['runs'])
            assert len(result['audits'])>=2
            assert abs(result['audits'][-2]['relative_l2']/result['relative_l2']-1)<1e-4
            for key in ('relative_l2','warm_wall_s'):
                assert math.isfinite(result[key]) and result[key]>0
            replacements[n]=(path,result)
        # Validate every replacement before changing the saved comparison.
        for row in rows:
            if row['solver']!='bspf':continue
            path,result=replacements[int(row['n'])]
            assert result['modes_sha256']==row['modes_sha256']
            row.update(relative_l2=result['relative_l2'],warm_full_solve_s=result['warm_wall_s'],
                       source=str(path.resolve().relative_to(OUT.parents[3])))
        with (OUT/'comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fieldnames,lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    cuda13=any(r['solver']=='bspf' and 'cuda13_sweep_' in r['source'] for r in rows)
    assert len(rows) == len(expected) * len(STYLES)
    for row in rows:
        row["n"] = int(row["n"])
        for key in ("relative_l2", "warm_full_solve_s"):
            row[key] = float(row[key])
            assert math.isfinite(row[key]) and row[key] > 0

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 200,
    })
    for metric, xlabel in (
        ("dof", r"Unique physical nodal DOFs, $n^3$ (including boundary)"),
        ("walltime", "Warm full-solve wall time (seconds)"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 6.3))
        for key, label, color, marker, linestyle in STYLES:
            if key=='bspf' and cuda13:label='BSPF Galerkin (CUDA 13)'
            data = sorted((r for r in rows if r["solver"] == key), key=lambda r: r["n"])
            assert [r["n"] for r in data] == expected
            x = [r["n"] ** 3 if metric == "dof" else r["warm_full_solve_s"] for r in data]
            ax.loglog(x, [r["relative_l2"] for r in data], label=label,
                      color=color, marker=marker, linestyle=linestyle,
                      linewidth=1.7, markersize=6, markerfacecolor="white",
                      markeredgewidth=1.4)
        ax.set(xlabel=xlabel, ylabel=r"Relative $L^2$ solution error (lower is better)",
               title="Curved 3D Poisson MMS · Leonardo · one A100",
               ylim=(7e-13, 12))
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="lower left" if metric == "dof" else "lower right",
                  bbox_to_anchor=(0.02 if metric == "dof" else 0.98, 0.12), framealpha=0.95)
        fig.text(0.12, 0.065, "Grids: 17³, 33³, 65³, 97³, 129³, 193³, 257³; different trial spaces.",
                 fontsize=9)
        fig.text(0.12, 0.035, "Time: median of 3 warm full solves; excludes setup, JIT, sampling and error audits.",
                 fontsize=8.5)
        if cuda13:
            fig.text(0.12,0.012,"BSPF: full nodal forcing, preassembled mass RHS; other curves retain saved configurations.",fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        for extension in ("png", "pdf"):
            path = OUT / f"accuracy_{metric}.{extension}"
            fig.savefig(path)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
