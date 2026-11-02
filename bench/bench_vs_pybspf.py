# SPDX-License-Identifier: GPL-3.0-or-later
"""Online throughput of bspf_cpp vs jitted pybspf (JAX x64), one CPU thread each.

Setup (plans, compiled ops, jit compilation) is excluded; each case times the
steady-state call. Run with XLA_FLAGS set as in bench/README or the default below.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pybspf

import bspf_cpp as bc


def timeit(fn, min_time=0.3):
    fn()
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = time.perf_counter() - t0
        if dt > min_time:
            return dt / reps
        reps *= 2


def block(fn):
    return lambda: jax.block_until_ready(fn())


def cases_1d(n, m, rng):
        x = np.linspace(0, 1, n)
        f = rng.standard_normal((n, m))
        line, plan = bc.Line(x), pybspf.plan_1d(jnp.asarray(x))
        fj = jnp.asarray(f)
        d1 = line.derivative(1)
        t = np.linspace(0, 1, 200)
        ev = line.evaluate(t)
        q = line.quadrature()
        jd = jax.jit(lambda p, g: pybspf.differentiate(p, g, order=1))
        ji = jax.jit(lambda p, g: pybspf.interpolate(p, g, jnp.asarray(t)))
        jq = jax.jit(lambda p, g: pybspf.integrate(p, g))
        tag = f"1D N={n} x{m}"
        return [(tag, "d/dx", lambda: d1(f), block(lambda: jd(plan, fj))),
                (tag, "interpolate 200 pts", lambda: ev(f), block(lambda: ji(plan, fj))),
                (tag, "integrate", lambda: q(f), block(lambda: jq(plan, fj)))]


def cases_grid(shape, rng):
        axes = [np.linspace(0, 1 + i, n) for i, n in enumerate(shape)]
        grid = bc.Grid.uniform(axes)
        plan = pybspf.tensor_plan(*[pybspf.plan_1d(jnp.asarray(a)) for a in axes])
        f = rng.standard_normal(shape)
        fj = jnp.asarray(f)
        d = len(shape)
        coords = [np.linspace(a[0], a[-1], max(8, n * 3 // 4)) for a, n in zip(axes, shape)]
        evs = [grid.line(i).evaluate(c) for i, c in enumerate(coords)]

        def interp(g=f):
            for i, e in enumerate(evs):
                g = e(g, axis=i)
            return g

        last = tuple(int(i == d - 1) for i in range(d))
        jx = jax.jit(lambda p, g: pybspf.mixed_partial(p, g, (1,) + (0,) * (d - 1)))
        jz = jax.jit(lambda p, g: pybspf.mixed_partial(p, g, last))
        jl = jax.jit(lambda p, g: pybspf.laplacian(p, g))
        ji = jax.jit(lambda p, g: pybspf.interpolate_grid(p, g, [jnp.asarray(c) for c in coords]))
        jq = jax.jit(lambda p, g: pybspf.integrate_box(p, g))
        tag = f"{d}D " + "x".join(map(str, shape))
        return [(tag, "d/dx (outer axis)", lambda: grid.partial(f, (1,) + (0,) * (d - 1)), block(lambda: jx(plan, fj))),
                (tag, "d/d(last axis)", lambda: grid.partial(f, last), block(lambda: jz(plan, fj))),
                (tag, "laplacian", lambda: grid.laplacian(f), block(lambda: jl(plan, fj))),
                (tag, "interpolate (0.75 n)^d", interp, block(lambda: ji(plan, fj))),
                (tag, "integrate box", lambda: grid.integrate(f), block(lambda: jq(plan, fj)))]


def cases():
    rng = np.random.default_rng(0)
    out = []
    for n, m in [(64, 4096), (256, 1024), (1024, 256)]:
        out += cases_1d(n, m, rng)
    for shape in [(128, 128), (256, 256), (48, 48, 48), (96, 96, 96)]:
        out += cases_grid(shape, rng)
    return out


def main():
    rows = []
    print(f"{'case':<20} {'operation':<24} {'bspf_cpp':>11} {'pybspf':>11} {'speedup':>8}")
    for tag, name, ours, theirs in cases():
        a, b = timeit(ours), timeit(theirs)
        rows.append(dict(case=tag, operation=name, bspf_cpp_s=a, pybspf_s=b))
        print(f"{tag:<20} {name:<24} {a * 1e3:9.3f}ms {b * 1e3:9.3f}ms {b / a:7.1f}x", flush=True)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
