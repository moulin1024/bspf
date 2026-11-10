# SPDX-License-Identifier: GPL-3.0-or-later
"""FP32 CUDA backend against the float64 CPU implementation.

Tolerance: the device computes in float32, so its error is compared with the
float32 floor of the same operator, i.e. how far the float64 CPU result moves
when only the input is rounded to float32. No float32 method can beat that.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bspf_cpp as bc

pytestmark = pytest.mark.skipif(bc.cuda is None or not bc.cuda.available(), reason="no CUDA backend/device")

FACTOR = 8.0     # device error / float32 input floor
ABSOLUTE = 5e-7  # float32 arithmetic on O(1) relative outputs
# 3rd/4th derivatives: knots finer than the grid (clustering at N < 64) make the
# spline rows cancel; there the float32 floor itself is already 1e-3..1e-1.
HIGH_ORDER_FACTOR = 40.0

CONFIGS = [
    dict(degree=5),
    dict(degree=7),
    dict(degree=5, lam=1e-6),
    dict(degree=5, clustering=1.5),
    dict(degree=4, constraint_order=2, boundary_points=6),
    dict(degree=5, endpoint_method="chebyshev"),
]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def agrees(device, host, *inputs, factor=FACTOR):
    """device(*inputs) matches host(*inputs) within the float32 floor."""
    rounded = [np.asarray(a, dtype=np.float32).astype(np.float64) for a in inputs]
    reference = host(*inputs)
    floor = rel(host(*rounded), reference)
    err = rel(device(*inputs), reference)
    assert err <= factor * floor + ABSOLUTE, (err, floor)


def field(x, batch=3):
    t = (x - x[0]) / (x[-1] - x[0])
    return np.stack([np.sin(3 * t + c) + np.exp(0.7 * t) * c + t ** 7 for c in range(batch)], axis=1)


@pytest.mark.parametrize("options", CONFIGS)
@pytest.mark.parametrize("n", [48, 63, 128])
def test_line_ops(n, options):
    x = np.linspace(-0.4, 1.3, n)
    line = bc.Line(x, **options)
    f = field(x)
    points = np.linspace(x[0], x[-1], 29)
    ops = [line.derivative(k) for k in range(3 if n > 64 else 5)]
    ops += [line.derivative(1, correction=False), line.evaluate(points), line.evaluate(points[3:-3], derivative=1),
            line.quadrature(), line.quadrature(0.9, -0.1), line.primitive(1), line.primitive(2), line.jets()]
    for i, op in enumerate(ops):
        gpu = bc.cuda.Op(op)
        factor = HIGH_ORDER_FACTOR if i in (3, 4) else FACTOR
        for axis, data in ((0, f), (1, np.ascontiguousarray(f.T))):
            agrees(lambda g, gpu=gpu, axis=axis: gpu(g, axis=axis), lambda g, op=op, axis=axis: op(g, axis=axis), data,
                   factor=factor)


def test_given_jets_along_middle_axis():
    x = np.linspace(0, 2, 40)
    line = bc.Line(x, degree=5)
    rng = np.random.default_rng(1)
    f = np.sin(x)[None, :, None] * rng.uniform(1, 2, (5, 1, 4))
    jets = line.jets()(f, axis=1) * 1.01  # given jets, not the estimate
    op, gpu = line.derivative(1), bc.cuda.Op(line.derivative(1))
    agrees(lambda a, g: gpu(a, axis=1, jets=g), lambda a, g: op(a, axis=1, jets=g), f, jets)


def grid_and_field(shape, batch=()):
    axes = [np.linspace(-0.3 + 0.1 * i, 1.0 + 0.4 * i, n) for i, n in enumerate(shape)]
    grid = bc.Grid.uniform(axes, degree=5)
    mesh = np.meshgrid(*axes, indexing="ij")
    f = np.exp(0.5 * mesh[0]) * np.cos(sum((i + 1) * m for i, m in enumerate(mesh)))
    f = f.reshape(f.shape + (1,) * len(batch)) * (1 + np.arange(int(np.prod(batch)))).reshape((1,) * len(shape) + batch)
    return axes, grid, np.ascontiguousarray(f)


@pytest.mark.parametrize("shape,batch", [((37,), (4,)), ((33, 40), ()), ((33, 40), (2,)), ((20, 23, 17), ()),
                                         ((20, 23, 17), (3,))])
def test_grid_ops(shape, batch):
    axes, grid, f = grid_and_field(shape, batch)
    gpu = bc.cuda.Grid(grid)
    d = len(shape)
    for orders in [tuple(int(i == a) for i in range(d)) for a in range(d)] + [(1,) * d, (2,) + (0,) * (d - 1)]:
        agrees(lambda g, o=orders: gpu.partial(g, o), lambda g, o=orders: grid.partial(g, o), f)
    agrees(gpu.laplacian, grid.laplacian, f)
    coords = [np.linspace(a[1], a[-2], 11 + i) for i, a in enumerate(axes)]
    agrees(lambda g: gpu.interpolate(g, coords), lambda g: grid.interpolate(g, coords), f)
    derivs = [1] + [0] * (d - 1)
    agrees(lambda g: gpu.interpolate(g, coords, derivs), lambda g: grid.interpolate(g, coords, derivs), f)
    agrees(gpu.integrate, grid.integrate, f)
    bounds = [(a[2], 0.5 * (a[0] + a[-1])) for a in axes]
    agrees(lambda g: gpu.integrate(g, bounds), lambda g: grid.integrate(g, bounds=bounds), f)


def test_device_resident_chain():
    axes, grid, f = grid_and_field((24, 30, 18))
    gpu = bc.cuda.Grid(grid)
    u = bc.cuda.Array(f)
    assert u.shape == f.shape
    lap = gpu.laplacian(u)
    assert isinstance(lap, bc.cuda.Array)
    agrees(lambda g: gpu.laplacian(bc.cuda.Array(g)).get(), grid.laplacian, f)
    total = gpu.integrate(lap).get()
    assert total.shape == ()
    np.testing.assert_allclose(np.asarray(u), f.astype(np.float32), rtol=0, atol=0)


def test_errors():
    axes, grid, f = grid_and_field((20, 21))
    gpu = bc.cuda.Grid(grid)
    with pytest.raises(ValueError):
        gpu.laplacian(np.zeros((21, 20)))
    op = bc.cuda.Op(grid.line(0).derivative(1))
    with pytest.raises(ValueError):
        op(f, axis=1)
    with pytest.raises(ValueError):
        op(f, axis=0, jets=np.zeros((3, 21)))


CHUNKED = """
import numpy as np, bspf_cpp as bc, sys
sys.path.insert(0, {tests!r})
from test_cuda import agrees, grid_and_field
for shape, batch in [((40, 37, 29), ()), ((300, 45), ()), ((45, 300), (2,)), ((64,), (700,))]:
    axes, grid, f = grid_and_field(shape, batch)
    gpu = bc.cuda.Grid(grid)
    agrees(gpu.laplacian, grid.laplacian, f)
    agrees(gpu.integrate, grid.integrate, f)
    coords = [np.linspace(a[0], a[-1], 50) for a in axes]
    agrees(lambda g: gpu.interpolate(g, coords), lambda g: grid.interpolate(g, coords), f)
print("ok")
"""


def test_chunked_paths():
    """Tiny chunks: many blocks per axis and slabs split along the line index."""
    tests = str(Path(__file__).parent)
    env = dict(os.environ, BSPF_CUDA_CHUNK="700")
    out = subprocess.run([sys.executable, "-c", CHUNKED.format(tests=tests)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
