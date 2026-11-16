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
    parser.add_argument('--bspf-dir',type=Path,help='Replace BSPF rows with a complete sweep, then redraw')
    parser.add_argument('--chebyshev-dir',type=Path,help='Replace Chebyshev rows with a complete sweep, then redraw')
    parser.add_argument('--nodes',type=int,nargs='+',help='Expected sizes for imported sweeps; default is the original seven sizes')
    args=parser.parse_args()
    with (OUT / "comparison.csv").open(newline="") as handle:
        reader=csv.DictReader(handle);fieldnames=reader.fieldnames
        rows = list(reader)
    expected = [17, 33, 65, 97, 129, 193, 257]
    imported=args.nodes or expected
    assert imported==sorted(set(imported)) and imported[0]==17 and imported[-1]==257
    for solver,directory in [('bspf',args.bspf_dir),('chebyshev',args.chebyshev_dir)]:
        if directory is None:continue
        replacements={}
        for n in imported:
            q=max(96,(3*n+1)//2)
            path=directory/f'n{n}_q{q}/result.json'
            result=json.loads(path.read_text())
            assert result['n']==n and result['q']==q
            assert result['boundary_error']==0
            assert all(r['relative_residual']<1e-11 for r in result['runs'])
            assert len(result['audits'])>=2
            assert abs(result['audits'][-2]['relative_l2']/result['relative_l2']-1)<1e-4
            for key in ('relative_l2','warm_wall_s'):
                assert math.isfinite(result[key]) and result[key]>0
            replacements[n]=(path,result)
        # Validate the whole sweep before replacing this solver's rows.
        hashes={row['modes_sha256'] for row in rows if row['solver']==solver}
        assert len(hashes)==1
        updated=[]
        for n,(path,result) in replacements.items():
            assert result['modes_sha256'] in hashes
            updated.append(dict(solver=solver,n=n,relative_l2=result['relative_l2'],
                                warm_full_solve_s=result['warm_wall_s'],modes_sha256=result['modes_sha256'],
                                source=str(path.resolve().relative_to(OUT.parents[3]))))
        rows=[row for row in rows if row['solver']!=solver]+updated
    if args.bspf_dir or args.chebyshev_dir:
        with (OUT/'comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fieldnames,lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    cuda13=any(r['solver']=='bspf' and ('cuda13_sweep_' in r['source'] or 'geosweep_cuda13_' in r['source']) for r in rows)
    cheb_cuda13=any(r['solver']=='chebyshev' and ('chebyshev_cuda13_' in r['source'] or 'geosweep_cuda13_' in r['source']) for r in rows)
    assert {r["solver"] for r in rows}=={s[0] for s in STYLES}
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
            if key=='chebyshev' and cheb_cuda13:label='Chebyshev Galerkin (CUDA 13)'
            data = sorted((r for r in rows if r["solver"] == key), key=lambda r: r["n"])
            ns=[r["n"] for r in data]
            assert ns==sorted(set(ns)) and ns[0]==17 and ns[-1]==257
            x = [r["n"] ** 3 if metric == "dof" else r["warm_full_solve_s"] for r in data]
            ax.loglog(x, [r["relative_l2"] for r in data], label=label,
                      color=color, marker=marker, linestyle=linestyle,
                      linewidth=1.7, markersize=6, markerfacecolor="white",
                      markeredgewidth=1.4)
        ax.set(xlabel=xlabel, ylabel=r"Relative $L^2$ solution error (lower is better)",
               title="Curved 3D Poisson MMS · Leonardo · one A100",
               ylim=(min(7e-13,0.7*min(r["relative_l2"] for r in rows)), 12))
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="lower left" if metric == "dof" else "lower right",
                  bbox_to_anchor=(0.02 if metric == "dof" else 0.98, 0.12), framealpha=0.95)
        counts={key:sum(r['solver']==key for r in rows) for key, *_ in STYLES}
        grid_note=(f"Grids: 17³–257³; BSPF {counts['bspf']}, Chebyshev {counts['chebyshev']} sizes; other solvers 7 sizes."
                   if len(set(counts.values()))>1 else "Grids: 17³, 33³, 65³, 97³, 129³, 193³, 257³; different trial spaces.")
        fig.text(0.12, 0.065, grid_note,
                 fontsize=9)
        fig.text(0.12, 0.035, "Time: median of 3 warm full solves; excludes setup, JIT, sampling and error audits.",
                 fontsize=8.5)
        if cuda13:
            note=("BSPF: nodal forcing + mass RHS; Chebyshev: Gauss forcing; other curves retain saved configurations."
                  if cheb_cuda13 else "BSPF: full nodal forcing, preassembled mass RHS; other curves retain saved configurations.")
            fig.text(0.12,0.012,note,fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        for extension in ("png", "pdf"):
            path = OUT / f"accuracy_{metric}.{extension}"
            fig.savefig(path)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
