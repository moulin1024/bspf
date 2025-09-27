import numpy as np
import numpy.testing as npt
import pytest

from bspf.bspf1d import _Knot, Grid1D  # replace with your actual module path


def test_knot_all():
    # ---------- setup ----------
    degree = 3
    a, b = -2.0, 4.0
    x = np.linspace(a, b, 41)
    grid = Grid1D(x)

    # ---------- _generate: raises when n_basis <= degree ----------
    with pytest.raises(ValueError, match="exceed"):
        _Knot._generate(degree=degree, domain=(a, b), n_basis=degree, use_clustering=False, clustering_factor=2.0)

    # ---------- _generate: no interior knots case (n_interior == 0) ----------
    # minimal valid n_basis gives only the repeated end-knots
    n_basis_min = degree + 1
    k0 = _Knot._generate(
        degree=degree, domain=(a, b), n_basis=n_basis_min, use_clustering=False, clustering_factor=2.0
    )
    # length = n_basis + degree + 1
    assert k0.size == n_basis_min + degree + 1
    # endpoints repeated degree+1 times
    npt.assert_allclose(k0[: degree + 1], a)
    npt.assert_allclose(k0[-(degree + 1):], b)

    # ---------- _generate: with interior knots, no clustering (uniform in domain) ----------
    n_basis = 8  # > degree+1 -> has interior knots
    k_uni = _Knot._generate(
        degree=degree, domain=(a, b), n_basis=n_basis, use_clustering=False, clustering_factor=2.0
    )
    assert k_uni.size == n_basis + degree + 1
    npt.assert_allclose(k_uni[: degree + 1], a)
    npt.assert_allclose(k_uni[-(degree + 1):], b)

    # interior knots should be linearly spaced in [a, b]
    interior = k_uni[degree + 1 : -(degree + 1)]
    # There should be n_interior = (n_basis + degree + 1) - 2*(degree+1) interior knots
    n_interior = (n_basis + degree + 1) - 2 * (degree + 1)
    assert interior.size == n_interior
    # Check equal spacing
    diffs = np.diff(interior)
    npt.assert_allclose(diffs, diffs[0])

    # ---------- _generate: with interior knots, clustering ----------
    k_clu = _Knot._generate(
        degree=degree, domain=(a, b), n_basis=n_basis, use_clustering=True, clustering_factor=3.0
    )
    # Same endpoints
    npt.assert_allclose(k_clu[: degree + 1], a)
    npt.assert_allclose(k_clu[-(degree + 1):], b)
    interior_c = k_clu[degree + 1 : -(degree + 1)]
    # Monotone increasing
    assert np.all(np.diff(interior_c) > 0)
    # Heavier density near ends than middle (edge gaps smaller than middle gap)
    edge_gap = min(interior_c[0] - a, b - interior_c[-1])
    mid_gap = np.max(np.diff(interior_c))
    assert edge_gap < mid_gap

    # ---------- resolve: return provided knots verbatim (and 1D check) ----------
    custom_knots = np.array([a, a, a, a, -1.0, 0.0, 2.0, b, b, b, b], dtype=np.float64)
    out = _Knot.resolve(
        degree=degree, grid=grid, knots=custom_knots, n_basis=None, domain=None,
        use_clustering=False, clustering_factor=2.0
    )
    npt.assert_array_equal(out, custom_knots)

    # Non-1D knots should raise
    with pytest.raises(ValueError, match="1D"):
        _Knot.resolve(
            degree=degree, grid=grid, knots=np.array([[a, b]]),
            n_basis=None, domain=None, use_clustering=False, clustering_factor=2.0
        )

    # ---------- resolve: when knots=None uses defaults for n_basis and domain ----------
    # Default n_basis = 2*(degree+1)*2 = 4*(degree+1)
    expected_n_basis = 4 * (degree + 1)
    out2 = _Knot.resolve(
        degree=degree, grid=grid, knots=None, n_basis=None, domain=None,
        use_clustering=False, clustering_factor=2.0
    )
    assert out2.size == expected_n_basis + degree + 1
    # domain taken from grid
    npt.assert_allclose(out2[: degree + 1], grid.a)
    npt.assert_allclose(out2[-(degree + 1):], grid.b)

    # ---------- resolve: explicit n_basis and domain override ----------
    nb = 6
    dom = (1.5, 3.0)
    out3 = _Knot.resolve(
        degree=degree, grid=grid, knots=None, n_basis=nb, domain=dom,
        use_clustering=False, clustering_factor=2.0
    )
    assert out3.size == nb + degree + 1
    npt.assert_allclose(out3[: degree + 1], dom[0])
    npt.assert_allclose(out3[-(degree + 1):], dom[1])
