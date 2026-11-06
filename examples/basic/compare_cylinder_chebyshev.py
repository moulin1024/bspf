# SPDX-License-Identifier: GPL-3.0-or-later
"""BSPF versus four-block Chebyshev-Lobatto differentiation using NumPy FFTs.

Same polynomial geometry and MMS, equal node counts, different node locations.
Chebyshev differentiates independently in each block then averages shared
traces. BSPF is measured both without overlap and with eight ghost samples.
Timings are warm median Laplacian applications, excluding geometry/setup/MMS.
This compares the current C++ BSPF and NumPy FFT implementations, not languages
or equally optimized kernels. No dense differentiation matrices are used.
"""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from diff_cylinder_rectangle_2d import cylinder_boundaries, cylinder_rectangle, manufactured
from diff_multipatch_2d import NAMES


def cheb_diff(values, axis):
    """Differentiate at ascending s=(1-cos(j*pi/N))/2 using FFT transforms.

    Even extension -> Chebyshev coefficients -> coefficient derivative
    recurrence -> inverse FFT. The factor -2 converts d/dcos(theta) to d/ds.
    """
    v = np.moveaxis(np.asarray(values), axis, 0)
    degree = v.shape[0]-1
    extended = np.concatenate((v, v[-2:0:-1]), axis=0)
    coeff = np.fft.rfft(extended, axis=0).real/degree
    coeff[[0, -1]] *= 0.5
    derivative = np.polynomial.chebyshev.chebder(coeff, scl=-2, axis=0)
    spectrum = np.zeros_like(coeff)
    spectrum[:degree] = derivative*degree
    spectrum[[0, -1]] *= 2
    result = np.fft.irfft(spectrum, n=2*degree, axis=0)[:degree+1]
    return np.moveaxis(result, 0, axis)


def geometry(n):
    s = (1-np.cos(np.pi*np.arange(n)/(n-1)))/2
    return geometry_at(s)


def geometry_at(s):
    """Exact metrics of X(s,t)=(1-s) arc(t)+s rectangle_side(t)."""
    t = s
    radial = s[:, None, None]
    grids = []
    for _, b, _, d, arc in cylinder_boundaries():
        a, at, att = [arc.evaluate(t, derivative=i)[None, :, :] for i in range(3)]
        outer = ((1-t[:, None])*b + t[:, None]*d)[None, :, :]
        xs = np.broadcast_to(outer-a, (len(s), len(t), 2))
        xt = (1-radial)*at + radial*(d-b)
        xst = np.broadcast_to(d-b-at, xs.shape)
        xtt = (1-radial)*att
        xy = (1-radial)*a + radial*outer
        jac = xs[..., 0]*xt[..., 1]-xs[..., 1]*xt[..., 0]
        if not np.all(jac > 0):
            raise ValueError("folded geometry")
        grids.append(dict(x=xy[..., 0], y=xy[..., 1],
                          a=xt[..., 1]/jac, b=-xs[..., 1]/jac,
                          c=-xt[..., 0]/jac, d=xs[..., 0]/jac,
                          xst=xst, xtt=xtt))
    return grids


def average_shared(fields):
    result = [v.copy() for v in fields]
    for k in range(4):
        j = (k+1) % 4
        mean = (fields[k][:, -1] + fields[j][:, 0])/2
        result[k][:, -1] = result[j][:, 0] = mean
    return result


def cheb_operators(grids, fields, lap_only=False):
    results = {name: [] for name in (("lap",) if lap_only else (*NAMES.values(), "lap"))}
    for g, u in zip(grids, fields):
        us, ut = cheb_diff(u, 0), cheb_diff(u, 1)
        uss, ust, utt = cheb_diff(us, 0), cheb_diff(us, 1), cheb_diff(ut, 1)
        a, b, c, d = (g[key] for key in ("a", "b", "c", "d"))
        ux, uy = a*us+b*ut, c*us+d*ut
        hst = ust - ux*g["xst"][..., 0] - uy*g["xst"][..., 1]
        htt = utt - ux*g["xtt"][..., 0] - uy*g["xtt"][..., 1]
        uxx = a*a*uss + 2*a*b*hst + b*b*htt
        uyy = c*c*uss + 2*c*d*hst + d*d*htt
        if lap_only:
            results["lap"].append(uxx+uyy)
        else:
            uxy = a*c*uss + (a*d+b*c)*hst + b*d*htt
            for key, value in zip(results, (ux, uy, uxx, uyy, uxy, uxx+uyy)):
                results[key].append(value)
    return {key: average_shared(value) for key, value in results.items()}


def errors(numerical, exact):
    return {name: max(float(np.max(np.abs(v-e[name]))) for v, e in zip(values, exact))
            / max(float(np.max(np.abs(e[name]))) for e in exact)
            for name, values in numerical.items()}


def median_time(call, repeats):
    call()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(perf_counter()-start)
    return float(np.median(times))


def laplacian_peak(numerical, exact, coordinates):
    """Location and absolute magnitude of the worst nodal Laplacian error."""
    candidates = []
    for k, (value, reference, (x, y)) in enumerate(zip(numerical, exact, coordinates)):
        error = np.abs(value-reference["lap"])
        index = np.unravel_index(np.argmax(error), error.shape)
        candidates.append(dict(absolute_error=float(error[index]), patch=k,
                               x=float(x[index]), y=float(y[index]),
                               radius=float(np.hypot(x[index], y[index]))))
    return max(candidates, key=lambda entry: entry["absolute_error"])


def run(sizes, thickness, repeats, **shock):
    rows = []
    for n in sizes:
        grids = geometry(n)
        exact = [manufactured(g["x"], g["y"], thickness, **shock) for g in grids]
        fields = [e["u"] for e in exact]
        values = cheb_operators(grids, fields)
        elapsed = median_time(lambda: cheb_operators(grids, fields, lap_only=True), repeats)
        rows.append(dict(n=n, nodes=4*n*(n-1), method="Chebyshev FFT", errors=errors(values, exact), seconds=elapsed,
                         laplacian_peak=laplacian_peak(values["lap"], exact, [(g["x"], g["y"]) for g in grids])))
        for overlap in (0, 8):
            mp, _ = cylinder_rectangle(n, overlap)
            exact = [manufactured(mp.patch(k).x, mp.patch(k).y, thickness, **shock) for k in range(4)]
            fields = [e["u"] for e in exact]
            values = {name: mp.partial(fields, orders) for orders, name in NAMES.items()}
            values["lap"] = mp.laplacian(fields)
            elapsed = median_time(lambda: mp.laplacian(fields), repeats)
            rows.append(dict(n=n, nodes=mp.node_count, method=f"BSPF overlap {overlap}", errors=errors(values, exact), seconds=elapsed,
                             laplacian_peak=laplacian_peak(values["lap"], exact, [(mp.patch(k).x, mp.patch(k).y) for k in range(4)])))
        for row in rows[-3:]:
            print(f"n={n:3d} {row['method']:17s} " + " ".join(f"{name}={e:.3e}" for name, e in row["errors"].items())
                  + f" time={row['seconds']*1e3:.3f} ms", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[17, 33, 65, 129])
    parser.add_argument("--layer-thickness", type=float, default=.1)
    parser.add_argument("--shock-amplitude", type=float, default=0., help="jump added inside the shock ring; 0 disables it")
    parser.add_argument("--shock-radius", type=float, default=1.)
    parser.add_argument("--shock-width", type=float, default=.02, help="tanh transition scale")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json")
    parser.add_argument("--plot")
    args = parser.parse_args()
    if min(args.sizes) < 17 or args.repeats < 1 or not np.isfinite(args.layer_thickness) or args.layer_thickness <= 0:
        parser.error("sizes >=17, repeats >=1 and finite positive thickness required")
    shock = dict(shock_amplitude=args.shock_amplitude, shock_radius=args.shock_radius, shock_width=args.shock_width)
    try:
        manufactured(np.array(1.), np.array(0.), args.layer_thickness, **shock)
    except ValueError as error:
        parser.error(str(error))
    print(f"MMS: delta={args.layer_thickness:g}, shock parameters={shock}", flush=True)
    rows = run(sorted(set(args.sizes)), args.layer_thickness, args.repeats, **shock)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(layer_thickness=args.layer_thickness, **shock,
            platform=platform.platform(), numpy=np.__version__, repeats=args.repeats,
            timing="warm median Laplacian applications; setup excluded",
            errors="relative nodal maximum including walls; nodal locations differ", results=rows), indent=2)+"\n")
    if args.plot:
        import matplotlib.pyplot as plt
        show_shock = args.shock_amplitude > 0
        fig, axes = plt.subplots(2 if show_shock else 1, 3, figsize=(15, 9 if show_shock else 4.5), constrained_layout=True)
        axes = axes.ravel()
        for method in dict.fromkeys(r["method"] for r in rows):
            subset = [r for r in rows if r["method"] == method]
            nodes = [r["nodes"] for r in subset]
            for ax, name in zip(axes, ("f_x", "lap")):
                ax.loglog(nodes, [r["errors"][name] for r in subset], "o-", label=method)
                ax.set(xlabel="Distinct nodes", ylabel="Relative maximum error", title=name)
            axes[2].loglog(nodes, [r["seconds"]*1e3 for r in subset], "o-", label=method)
        axes[2].set(xlabel="Distinct nodes", ylabel="Warm median time (ms)", title="Laplacian application")
        for ax in axes[:3]:
            ax.legend()
            ax.grid(alpha=.2)
        if show_shock:
            grids = geometry(257)
            exact = [manufactured(g["x"], g["y"], args.layer_thickness, **shock) for g in grids]
            for ax, name in zip(axes[3:5], ("u", "lap")):
                limit = max(float(np.abs(e[name]).max()) for e in exact)
                for g, e in zip(grids, exact):
                    im = ax.pcolormesh(g["x"], g["y"], e[name], shading="gouraud", cmap="coolwarm", vmin=-limit, vmax=limit)
                fig.colorbar(im, ax=ax)
                ax.set(aspect="equal", xlabel="x", ylabel="y", title=f"Exact {name}: shock ring")
            radius = np.linspace(.5, 2, 2001)
            profile = manufactured(radius, np.zeros_like(radius), args.layer_thickness, **shock)
            axes[5].plot(radius, profile["u"])
            axes[5].axvline(args.shock_radius, color="k", ls="--", lw=.8)
            axes[5].set(xlabel="r along y=0, x>0", ylabel="u", title="Exact wall layer + shock profile")
        fig.suptitle(f"Four blocks: δ={args.layer_thickness:g}, shock A={args.shock_amplitude:g}, rₛ={args.shock_radius:g}, w={args.shock_width:g}")
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
