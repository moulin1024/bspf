# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate the FP32 CUDA backend against exact solutions in 1D, 2D and 3D.

Non-periodic analytic fields on rectangles; relative max errors of the CPU
(float64) and GPU (float32) results against the exact values, plus the GPU's
distance from the CPU. Pass an output JSON path as argv[1] (optional).

    PYTHONPATH=python python bench/validate_cuda.py bench/results/cuda_validation.json
"""
import json
import sys

import numpy as np

import bspf_cpp as bc


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))


def report(rows, dim, n, name, cpu, gpu, exact):
    row = dict(dim=dim, n=n, quantity=name, cpu64=rel(cpu, exact), gpu32=rel(gpu, exact), gpu_vs_cpu=rel(gpu, cpu))
    rows.append(row)
    print(f"{dim}D n={n:4d} {name:22s} cpu64 {row['cpu64']:8.1e}  gpu32 {row['gpu32']:8.1e}  gpu-cpu {row['gpu_vs_cpu']:8.1e}",
          flush=True)


def one_d(rows):
    # f = sin(5x) + exp(x) on [0, 1.7]: neither values nor slopes are periodic.
    f = lambda x: np.sin(5 * x) + np.exp(x)
    df = [f, lambda x: 5 * np.cos(5 * x) + np.exp(x), lambda x: -25 * np.sin(5 * x) + np.exp(x)]
    F = lambda x: -np.cos(5 * x) / 5 + np.exp(x)
    for n in (64, 128, 256, 512):
        x = np.linspace(0, 1.7, n)
        line = bc.Line(x)
        u = np.stack([f(x) * (1 + 0.1 * k) for k in range(8)], axis=1)  # 8 lines
        scale = 1 + 0.1 * np.arange(8)
        for k in (1, 2):
            op = line.derivative(k)
            report(rows, 1, n, f"d{k}/dx{k}", op(u), bc.cuda.Op(op)(u), df[k](x)[:, None] * scale)
        t = np.random.default_rng(0).uniform(0, 1.7, 200)
        op = line.evaluate(t)
        report(rows, 1, n, "interpolation", op(u), bc.cuda.Op(op)(u), f(t)[:, None] * scale)
        op = line.quadrature(0.2, 1.5)
        report(rows, 1, n, "integral [0.2,1.5]", op(u), bc.cuda.Op(op)(u), (F(1.5) - F(0.2)) * scale[None, :])
        op = line.primitive(1)
        report(rows, 1, n, "primitive", op(u), bc.cuda.Op(op)(u), (F(x) - F(0))[:, None] * scale)


def two_d(rows):
    # u = exp(x + y/2) sin(3x) cos(2y) on [0, 1.5] x [-0.5, 1.2].
    for n in (64, 128, 256, 512):
        x, y = np.linspace(0, 1.5, n), np.linspace(-0.5, 1.2, n + 7)
        X, Y = np.meshgrid(x, y, indexing="ij")
        e, s, c = np.exp(X + Y / 2), np.sin(3 * X), np.cos(2 * Y)
        u = e * s * c
        ux = e * (s + 3 * np.cos(3 * X)) * c
        uy = e * s * (0.5 * c - 2 * np.sin(2 * Y))
        uxx = e * c * (s + 6 * np.cos(3 * X) - 9 * s)
        uyy = e * s * (0.25 * c - 2 * np.sin(2 * Y) - 4 * c)
        uxy = e * (s + 3 * np.cos(3 * X)) * (0.5 * c - 2 * np.sin(2 * Y))
        grid = bc.Grid.uniform((x, y))
        gpu = bc.cuda.Grid(grid)
        d = bc.cuda.Array(u)
        report(rows, 2, n, "u_x", grid.partial(u, (1, 0)), gpu.partial(d, (1, 0)).get(), ux)
        report(rows, 2, n, "u_y", grid.partial(u, (0, 1)), gpu.partial(d, (0, 1)).get(), uy)
        report(rows, 2, n, "u_xy", grid.partial(u, (1, 1)), gpu.partial(d, (1, 1)).get(), uxy)
        report(rows, 2, n, "laplacian", grid.laplacian(u), gpu.laplacian(d).get(), uxx + uyy)
        xs, ys = np.linspace(0.05, 1.45, 50), np.linspace(-0.45, 1.1, 40)
        XS, YS = np.meshgrid(xs, ys, indexing="ij")
        report(rows, 2, n, "interpolation", grid.interpolate(u, (xs, ys)), gpu.interpolate(d, (xs, ys)).get(),
               np.exp(XS + YS / 2) * np.sin(3 * XS) * np.cos(2 * YS))
        # Integral by separation: int e^x sin3x dx * int e^{y/2} cos2y dy.
        ix = lambda a: np.exp(a) * (np.sin(3 * a) - 3 * np.cos(3 * a)) / 10
        iy = lambda b: np.exp(b / 2) * (0.5 * np.cos(2 * b) + 2 * np.sin(2 * b)) / 4.25
        exact = (ix(1.5) - ix(0)) * (iy(1.2) - iy(-0.5))
        report(rows, 2, n, "integral", grid.integrate(u), gpu.integrate(d).get(), np.array(exact))


def three_d(rows):
    # u = sin(2x + y) exp(z) cos(z) on [0,1] x [0,1.3] x [-0.2,0.9].
    for n in (32, 64, 96, 128):
        x, y, z = np.linspace(0, 1, n), np.linspace(0, 1.3, n + 3), np.linspace(-0.2, 0.9, n - 2)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        a, ez, cz, sz = 2 * X + Y, np.exp(Z), np.cos(Z), np.sin(Z)
        u = np.sin(a) * ez * cz
        grads = [2 * np.cos(a) * ez * cz, np.cos(a) * ez * cz, np.sin(a) * ez * (cz - sz)]
        lap = -5 * np.sin(a) * ez * cz + np.sin(a) * ez * (-2 * sz)
        grid = bc.Grid.uniform((x, y, z))
        gpu = bc.cuda.Grid(grid)
        d = bc.cuda.Array(u)
        for i, name in enumerate(("u_x", "u_y", "u_z")):
            o = tuple(int(i == j) for j in range(3))
            report(rows, 3, n, name, grid.partial(u, o), gpu.partial(d, o).get(), grads[i])
        report(rows, 3, n, "laplacian", grid.laplacian(u), gpu.laplacian(d).get(), lap)
        cs = (np.linspace(0.02, 0.98, 20), np.linspace(0.05, 1.25, 21), np.linspace(-0.15, 0.85, 22))
        XS, YS, ZS = np.meshgrid(*cs, indexing="ij")
        report(rows, 3, n, "interpolation", grid.interpolate(u, cs), gpu.interpolate(d, cs).get(),
               np.sin(2 * XS + YS) * np.exp(ZS) * np.cos(ZS))
        # int_0^1 sin(2x+y) dx = (cos y - cos(2+y))/2, then over y in [0, 1.3].
        ixy = (np.sin(1.3) - np.sin(3.3) + np.sin(2)) / 2
        iz = lambda b: np.exp(b) * (np.cos(b) + np.sin(b)) / 2
        report(rows, 3, n, "integral", grid.integrate(u), gpu.integrate(d).get(), np.array(ixy * (iz(0.9) - iz(-0.2))))


def main():
    print("device:", bc.cuda.device_name())
    rows = []
    one_d(rows)
    two_d(rows)
    three_d(rows)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as out:
            json.dump(rows, out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    main()
