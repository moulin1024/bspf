# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused 3D steady-state timings; pass the output JSON path as argv[1].

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 and PYTHONPATH=python.
Setup and first-call FFT planning are excluded. Values are medians of five
batches of fifteen calls. Compare builds in separate, otherwise idle processes.
"""
import json
import sys
import time

import numpy as np
import bspf_cpp as bc


def main():
    rows = []
    for n in (48, 96):
        axes = [np.linspace(0, 1 + i, n) for i in range(3)]
        grid = bc.Grid.uniform(axes)
        f = np.random.default_rng(4).normal(size=(n, n, n))
        coords = [np.linspace(0, 1 + i, 3 * n // 4) for i in range(3)]
        ops = [grid.line(i).evaluate(c) for i, c in enumerate(coords)]

        def interp():
            a = f
            for i, op in enumerate(ops):
                a = op(a, axis=i)
            return a

        funcs = {
            **{f'd{i}': lambda i=i: grid.partial(f, tuple(2 if j == i else 0 for j in range(3)))
               for i in range(3)},
            'laplacian': lambda: grid.laplacian(f),
            'interpolate': interp,
            'interpolate_grid': lambda: grid.interpolate(f, coords),
            'integrate': lambda: grid.integrate(f),
        }
        for name, fn in funcs.items():
            fn()
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(15):
                    fn()
                timings.append((time.perf_counter() - start) / 15)
            row = dict(n=n, operation=name, seconds=float(np.median(timings)))
            rows.append(row)
            print(row, flush=True)
    with open(sys.argv[1], 'w') as out:
        json.dump(rows, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
