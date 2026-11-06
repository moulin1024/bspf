# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-patch differentiation: conforming deformed patches joined at faces."""
from math import comb

import numpy as np
import pytest

import bspf_cpp as bc

CHEB6 = dict(degree=9, n_basis=18, lam=1e-6, endpoint_method="chebyshev", boundary_points=16,
             chebyshev_modes=12, chebyshev_alpha=1e-12, constraint_order=6)


def line(a, b):
    return bc.SplineCurve(np.array([a, b], dtype=float), 1)


def bezier_fit(points, degree):
    s = np.linspace(0, 1, len(points))
    B = np.stack([comb(degree, k) * s**k * (1 - s) ** (degree - k) for k in range(degree + 1)], axis=1)
    ctrl = np.zeros((degree + 1, 2))
    ctrl[0], ctrl[-1] = points[0], points[-1]
    rhs = points - B[:, [0]] * ctrl[0] - B[:, [-1]] * ctrl[-1]
    ctrl[1:-1] = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)[0]
    return ctrl


def arc(radius, t0, t1, degree=9):
    t = np.linspace(t0, t1, 200)
    r = radius(t)
    return bc.SplineCurve(bezier_fit(np.column_stack((r * np.cos(t), r * np.sin(t))), degree), degree)


INNER = lambda t: 0.5 + 0 * t
OUTER = lambda t: 1.2 + 0.15 * np.cos(3 * t)


def annulus(n, options=CHEB6, patches=4):
    """A doubly connected ring: xi radial (inner -> outer), eta along the angle."""
    grids = []
    for k in range(patches):
        t0, t1 = 2 * np.pi * k / patches, 2 * np.pi * (k + 1) / patches
        ray = lambda t: line(INNER(t) * np.array([np.cos(t), np.sin(t)]), OUTER(t) * np.array([np.cos(t), np.sin(t)]))
        grids.append(bc.MappedGrid2D(ray(t0), arc(OUTER, t0, t1), ray(t1), arc(INNER, t0, t1), (n, n), **options))
    return bc.MultiPatch2D(grids, [((k, 1, 1), ((k + 1) % patches, 1, 0), False) for k in range(patches)])


def f(x, y):   return np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * np.exp(x * y)
def fx(x, y):  return 2 * np.cos(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * y * np.exp(x * y)
def fyy(x, y): return -9 * np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * x * x * np.exp(x * y)


def samples(mp, fn):
    return [fn(mp.patch(p).x, mp.patch(p).y) for p in range(mp.patch_count)]


def relative_error(values, exact):
    scale = max(np.abs(e).max() for e in exact)
    return max(np.abs(v - e).max() for v, e in zip(values, exact)) / scale


def test_annulus_topology():
    mp = annulus(17)
    assert mp.patch_count == 4
    assert mp.node_count == 4 * 17 * 17 - 4 * 17           # four shared radial cuts
    assert sorted(mp.boundary_faces) == sorted((p, 0, s) for p in range(4) for s in (0, 1))  # inner and outer rings
    assert mp.shared_jump(samples(mp, f)) < 1e-12            # the same physical nodes


def test_shared_nodes_are_single_valued():
    mp = annulus(25)
    u = samples(mp, f)
    raw = mp.partial(u, (1, 0), average_shared=False)
    assert mp.shared_jump(raw) > 0                           # patch-wise values differ slightly...
    assert mp.shared_jump(mp.partial(u, (1, 0))) == 0        # ...and are averaged
    with_batch = [np.stack((a, 2 * a), axis=-1) for a in u]
    np.testing.assert_allclose(mp.partial(with_batch, (1, 0))[2][..., 1], 2 * mp.partial(u, (1, 0))[2], rtol=1e-12)


@pytest.mark.parametrize("orders,exact,bound", [((1, 0), fx, 1e-7), ((0, 2), fyy, 1e-5)])
def test_annulus_converges(orders, exact, bound):
    errors = []
    for n in (17, 33, 65):
        mp = annulus(n)
        errors.append(relative_error(mp.partial(samples(mp, f), orders), samples(mp, exact)))
    assert errors[2] < errors[0] / 1000 and errors[2] < bound


def split_square(reverse_second=True, n=(21, 17)):
    """[0, 1]^2 cut at x = 1/2; the second patch is rotated by 180 degrees."""
    a = bc.MappedGrid2D(line((0, 0), (0.5, 0)), line((0.5, 0), (0.5, 1)), line((0, 1), (0.5, 1)), line((0, 0), (0, 1)), n, **CHEB6)
    b = bc.MappedGrid2D(line((1, 1), (0.5, 1)), line((0.5, 1), (0.5, 0)), line((1, 0), (0.5, 0)), line((1, 1), (1, 0)), n, **CHEB6)
    return [a, b], [((0, 0, 1), (1, 0, 1), reverse_second)]


def test_reversed_interface():
    patches, links = split_square()
    mp = bc.MultiPatch2D(patches, links)
    u = samples(mp, f)
    assert mp.shared_jump(u) < 1e-14
    assert relative_error(mp.partial(u, (1, 0)), samples(mp, fx)) < 1e-6
    with pytest.raises(ValueError, match="coincide"):
        bc.MultiPatch2D(*split_square(reverse_second=False))


def test_single_patch_matches_mapped_grid():
    patches, _ = split_square()
    mp = bc.MultiPatch2D(patches[:1], [])
    u = samples(mp, f)
    np.testing.assert_array_equal(mp.partial(u, (2, 0))[0], patches[0].partial(u[0], (2, 0)))


def test_invalid_topologies():
    patches, links = split_square()
    with pytest.raises(ValueError, match="used twice"):
        bc.MultiPatch2D(patches, links + [((0, 0, 1), (1, 1, 0), False)])
    with pytest.raises(ValueError, match="connected"):
        bc.MultiPatch2D(patches, [])
    other = bc.MappedGrid2D(line((1, 1), (0.5, 1)), line((0.5, 1), (0.5, 0)), line((1, 0), (0.5, 0)), line((1, 1), (1, 0)), (21, 13))
    with pytest.raises(ValueError, match="nodes"):
        bc.MultiPatch2D([patches[0], other], links)


BEST = dict(CHEB6, boundary_points=12, chebyshev_modes=12, constraint_order=8)
EXACT = {(1, 0): fx, (0, 2): fyy,
         (0, 1): lambda x, y: -3 * np.sin(2 * x + 0.5) * np.sin(3 * y - 0.2) + 0.5 * x * np.exp(x * y),
         (2, 0): lambda x, y: -4 * np.sin(2 * x + 0.5) * np.cos(3 * y - 0.2) + 0.5 * y * y * np.exp(x * y),
         (1, 1): lambda x, y: -6 * np.cos(2 * x + 0.5) * np.sin(3 * y - 0.2) + 0.5 * (1 + x * y) * np.exp(x * y)}


def ring(nr, nt, overlap, options=BEST):
    grids = []
    for k in range(4):
        t0, t1 = np.pi * k / 2, np.pi * (k + 1) / 2
        ray = lambda t: line(INNER(t) * np.array([np.cos(t), np.sin(t)]), OUTER(t) * np.array([np.cos(t), np.sin(t)]))
        grids.append(bc.MappedGrid2D(ray(t0), arc(OUTER, t0, t1), ray(t1), arc(INNER, t0, t1), (nr, nt), **options))
    return bc.MultiPatch2D(grids, [((k, 1, 1), ((k + 1) % 4, 1, 0), False) for k in range(4)], overlap=overlap)


def interface_error(mp, orders, rows=6):
    """Relative error on the nodes within `rows` of the eta faces (the ring's interfaces)."""
    u = samples(mp, f)
    exact = samples(mp, EXACT[orders])
    scale = max(np.abs(e).max() for e in exact)
    errors = [np.abs(v - e) / scale for v, e in zip(mp.partial(u, orders), exact)]
    return max(max(e[:, :rows].max(), e[:, -rows:].max()) for e in errors)


@pytest.mark.parametrize("orders", list(EXACT))
def test_overlap_removes_the_interface_error(orders):
    one_sided, overlapped = interface_error(ring(33, 65, 0), orders), interface_error(ring(33, 65, 8), orders)
    assert overlapped < one_sided / 30
    assert relative_error(ring(33, 65, 8).partial(samples(ring(33, 65, 8), f), orders),
                          samples(ring(33, 65, 8), EXACT[orders])) < 1e-9


def test_overlap_depth_is_capped_by_the_patches():
    coarse = ring(9, 9, 12, options={})   # capped at a quarter of the 9-point lines (2 ghosts)
    u = samples(coarse, f)
    assert relative_error(coarse.partial(u, (1, 0)), samples(coarse, fx)) < 0.1
    with pytest.raises(ValueError, match="nonnegative"):
        ring(9, 9, -1, options={})


def bezier(points):
    pts = np.asarray(points, float)
    return bc.SplineCurve(pts, len(pts) - 1)


def quartered_square(n, overlap):
    """[0, 1]^2 in 2 x 2 patches; curved cuts meet at an off-center point, so every
    patch has interfaces along both axes (mixed derivatives use gradient ghosts)."""
    c = (0.52, 0.47)
    v_low = bezier([(0.5, 0), (0.56, 0.2), (0.47, 0.35), c])
    v_high = bezier([c, (0.58, 0.65), (0.45, 0.8), (0.5, 1)])
    h_left = bezier([(0, 0.5), (0.2, 0.44), (0.35, 0.55), c])
    h_right = bezier([c, (0.7, 0.4), (0.85, 0.55), (1, 0.5)])
    sw = bc.MappedGrid2D(line((0, 0), (0.5, 0)), v_low, h_left, line((0, 0), (0, 0.5)), (n, n), **BEST)
    se = bc.MappedGrid2D(line((0.5, 0), (1, 0)), line((1, 0), (1, 0.5)), h_right, v_low, (n, n), **BEST)
    nw = bc.MappedGrid2D(h_left, v_high, line((0, 1), (0.5, 1)), line((0, 0.5), (0, 1)), (n, n), **BEST)
    ne = bc.MappedGrid2D(h_right, line((1, 0.5), (1, 1)), line((0.5, 1), (1, 1)), v_high, (n, n), **BEST)
    links = [((0, 0, 1), (1, 0, 0), False), ((0, 1, 1), (2, 1, 0), False),
             ((1, 1, 1), (3, 1, 0), False), ((2, 0, 1), (3, 0, 0), False)]
    return bc.MultiPatch2D([sw, se, nw, ne], links, overlap=overlap)


QUARTER_INTERFACES = {0: ((0, 1), (1, 1)), 1: ((0, 0), (1, 1)), 2: ((0, 1), (1, 0)), 3: ((0, 0), (1, 0))}


def interface_interior_error(mp, orders, k=6):
    """Relative error near interface faces but away from the physical walls."""
    u, exact = samples(mp, f), samples(mp, EXACT[orders])
    scale = max(np.abs(e).max() for e in exact)
    worst = 0.0
    for p, (v, e) in enumerate(zip(mp.partial(u, orders), exact)):
        err, mask = np.abs(v - e) / scale, np.zeros(e.shape, bool)
        for axis, side in QUARTER_INTERFACES[p]:
            sl = [slice(None)] * 2
            sl[axis] = slice(-k, None) if side else slice(0, k)
            mask[tuple(sl)] = True
        for axis in (0, 1):
            for side in (0, 1):
                if (axis, side) not in QUARTER_INTERFACES[p]:
                    sl = [slice(None)] * 2
                    sl[axis] = slice(-k, None) if side else slice(0, k)
                    mask[tuple(sl)] = False
        worst = max(worst, err[mask].max())
    return worst


@pytest.mark.parametrize("orders", list(EXACT))
def test_interfaces_along_both_axes(orders):
    one_sided, overlapped = quartered_square(33, 0), quartered_square(33, 8)
    assert overlapped.node_count == 4 * 33 * 33 - 4 * 33 + 1  # four cuts and the shared center
    # Overlap reduces the interface error (7-42x for most orders; walls stay one-sided).
    assert interface_interior_error(overlapped, orders) < interface_interior_error(one_sided, orders) / 1.5
    total = relative_error(overlapped.partial(samples(overlapped, f), orders), samples(overlapped, EXACT[orders]))
    assert total < (1e-9 if sum(orders) == 1 else 5e-8)


# A channel with wavy walls, periodic in x with period 1: the faces x = 0 and x = 1
# are joined by a periodic interface with shift (1, 0).
TAU = 2 * np.pi


def wavy(y0, phase, x0, x1, degree=11):
    x = np.linspace(x0, x1, 200)
    pts = np.column_stack((x, y0 + 0.1 * np.sin(TAU * x + phase)))
    return bc.SplineCurve(bezier_fit(pts, degree), degree)


def channel(n, overlap=8, periodic=True, patches=2, shift=(1.0, 0.0)):
    cuts = np.linspace(0, 1, patches + 1)
    side = lambda x: line((x, 0.1 * np.sin(TAU * x)), (x, 1 + 0.1 * np.sin(TAU * x + 1)))
    shape = ((n - 1) * 2 // patches + 1, 2 * n - 1)
    grids = [bc.MappedGrid2D(wavy(0, 0, a, b), side(b), wavy(1, 1, a, b), side(a), shape, **BEST) for a, b in zip(cuts, cuts[1:])]
    links = [((k, 0, 1), (k + 1, 0, 0), False) for k in range(patches - 1)]
    per = [((0, 0, 0), (patches - 1, 0, 1), False, shift)] if periodic else []
    return bc.MultiPatch2D(grids, links, periodic=per, overlap=overlap)


def g(x, y):   return np.sin(TAU * x + 0.3) * np.cos(2 * y - 0.1) + 0.3 * np.cos(2 * TAU * x) * y * y
def gx(x, y):  return TAU * np.cos(TAU * x + 0.3) * np.cos(2 * y - 0.1) - 0.6 * TAU * np.sin(2 * TAU * x) * y * y
def gxx(x, y): return -TAU**2 * np.sin(TAU * x + 0.3) * np.cos(2 * y - 0.1) - 1.2 * TAU**2 * np.cos(2 * TAU * x) * y * y
def gxy(x, y): return -2 * TAU * np.cos(TAU * x + 0.3) * np.sin(2 * y - 0.1) - 1.2 * TAU * np.sin(2 * TAU * x) * y


PERIODIC_EXACT = {(1, 0): gx, (2, 0): gxx, (1, 1): gxy}


def away_from_walls(mp, orders, rows=6):
    """Relative error at nodes more than `rows` lines from the walls (eta = 0, 1)."""
    u, exact = samples(mp, g), samples(mp, PERIODIC_EXACT[orders])
    scale = max(np.abs(e).max() for e in exact)
    return max(np.abs(v - e)[:, rows:-rows].max() for v, e in zip(mp.partial(u, orders), exact)) / scale


def test_periodic_topology():
    n = 17
    mp = channel(n)
    assert mp.node_count == 2 * n * (2 * n - 1) - 2 * (2 * n - 1)  # the cut and the periodic pair merge
    assert mp.boundary_faces == [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]  # only the walls
    assert mp.ghost_depth((0, 0, 0)) == mp.ghost_depth((1, 0, 1)) == (n - 1) // 4  # capped by the patch size
    assert mp.shared_jump(mp.partial(samples(mp, g), (1, 0))) == 0.0
    with pytest.raises(ValueError, match="shift"):
        channel(n, shift=(0.9, 0.0))


@pytest.mark.parametrize("orders", list(PERIODIC_EXACT))
def test_periodic_faces_are_interior(orders):
    one_sided, overlapped = channel(17, overlap=0), channel(17)
    walls = channel(17, periodic=False)
    assert away_from_walls(overlapped, orders) < away_from_walls(one_sided, orders) / 100
    assert away_from_walls(overlapped, orders) < away_from_walls(walls, orders) / 100
    assert away_from_walls(overlapped, orders) < 2e-7


def test_patch_periodic_with_itself():
    single = channel(17, patches=1)
    assert single.boundary_faces == [(0, 1, 0), (0, 1, 1)]
    for orders in PERIODIC_EXACT:
        assert away_from_walls(single, orders) < 1e-7
