# SPDX-License-Identifier: GPL-3.0-or-later
"""Cache-group boundaries, explicit jets, and native Laplacian accumulation."""
import numpy as np
import pytest
import bspf_cpp as bc


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('integral', [False, True])
def test_spectral_groups(axis, integral):
    n, count = 31, 1103  # exceeds a spectral group and leaves a partial block
    line = bc.Line(np.linspace(0, 1, n))
    op = line.quadrature(0.1, 0.9) if integral else line.evaluate(np.linspace(0, 1, 17))
    rng = np.random.default_rng(12)
    shape = [(n, count), (2, n, count), (2, count, n)][axis]
    f = rng.normal(size=shape)
    jet_shape = list(shape)
    jet_shape[axis] = op.jet_count
    jets = rng.normal(size=jet_shape)
    actual = op(f, axis=axis, jets=jets)
    # Small independent applications exercise the same map without the grouped
    # spectral path, including the last line on either side of each boundary.
    for i in (0, 127, 128, 1023, 1024, count - 1):
        sl = (slice(None), i) if axis == 0 else ((1, slice(None), i) if axis == 1 else (1, i, slice(None)))
        expected = op(f[sl], jets=jets[sl])
        np.testing.assert_allclose(actual[sl], expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize('shape', [(31,), (31, 25), (31, 25, 23)])
@pytest.mark.parametrize('batch', [(), (2, 3), (0,)])
def test_native_laplacian(shape, batch):
    grid = bc.Grid.uniform([np.linspace(0, 1 + i, n) for i, n in enumerate(shape)])
    f = np.random.default_rng(7).normal(size=shape + batch)
    expected = sum(grid.partial(f, tuple(2 if i == j else 0 for i in range(len(shape))))
                   for j in range(len(shape)))
    np.testing.assert_array_equal(grid.laplacian(f), expected)
    with pytest.raises(ValueError, match='spatial shape'):
        grid.laplacian(np.zeros((3,)))


@pytest.mark.parametrize('kind', ['identity', 'derivative', 'spline', 'primitive', 'evaluate', 'quadrature', 'jets'])
def test_workspace_reuse_preserves_inputs(kind):
    line = bc.Line(np.linspace(0, 1, 31))
    op = {
        'identity': lambda: line.derivative(0),
        'derivative': lambda: line.derivative(2),
        'spline': lambda: line.derivative(2, correction=False),
        'primitive': lambda: line.primitive(2),
        'evaluate': lambda: line.evaluate(np.linspace(0, 1, 17)),
        'quadrature': line.quadrature,
        'jets': line.jets,
    }[kind]()
    rng = np.random.default_rng(31)
    f = rng.normal(size=(3, 31, 137))  # multiple slabs and a short final block
    jets = rng.normal(size=(3, op.jet_count, 137))
    original_f, original_jets = f.copy(), jets.copy()
    f.setflags(write=False)
    jets.setflags(write=False)
    for given in (False, True):
        kwargs = {'jets': jets} if given else {}
        actual = op(f, axis=1, **kwargs)
        expected = np.stack([op(f[i], axis=0, **({'jets': jets[i]} if given else {}))
                             for i in range(3)])
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(f, original_f)
    np.testing.assert_array_equal(jets, original_jets)


def test_workspace_is_private_to_each_call():
    from concurrent.futures import ThreadPoolExecutor

    line = bc.Line(np.linspace(0, 1, 31))
    op = line.derivative(2)
    fields = [np.random.default_rng(i).normal(size=(3, 31, n)) for i, n in enumerate((128, 137, 256, 73))]
    expected = [op(f, axis=1) for f in fields]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda f: op(f, axis=1), fields))
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
