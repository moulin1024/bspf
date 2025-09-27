import numpy as np
import numpy.testing as npt
import pytest

from bspf.bspf1d import Grid1D  # replace with the actual module path

def test_grid1d_all():
    # --- basic uniform grid ---
    x = np.linspace(0.0, 1.0, 5)  # dx=0.25
    g = Grid1D(x)
    assert g.dx == pytest.approx(0.25)
    npt.assert_array_equal(g.x, x.astype(np.float64))

    expected_omega = 2.0 * np.pi * np.fft.rfftfreq(len(x), d=g.dx)
    npt.assert_allclose(g.omega, expected_omega, rtol=0, atol=0)

    w = np.full(x.size, g.dx, dtype=np.float64)
    w[0] = w[-1] = g.dx / 2.0
    npt.assert_allclose(g.trap, w)
    assert g.a == pytest.approx(0.0)
    assert g.b == pytest.approx(1.0)
    assert g.n == 5

    # --- too few points raises ---
    with pytest.raises(ValueError, match="at least 2 points"):
        Grid1D(np.array([0.0]))

    # --- atol allows small variation ---
    x2 = np.linspace(0.0, 1.0, 11)  # dx=0.1
    x2p = x2.copy()
    x2p[2] += 1e-7
    g2 = Grid1D(x2p, atol=1e-6)
    assert g2.n == 11
    assert g2.dx == pytest.approx(0.1)

    # --- trap weights sum equals interval length ---
    x3 = np.linspace(-2.0, 3.0, 21)  # length=5.0
    g3 = Grid1D(x3)
    assert np.sum(g3.trap) == pytest.approx(g3.b - g3.a)

    # --- omega properties ---
    x4 = np.linspace(0.0, 1.0, 8)  # rfft length = 5
    g4 = Grid1D(x4)
    assert g4.omega.shape[0] == x4.size // 2 + 1
    assert g4.omega[0] == pytest.approx(0.0)
    assert np.all(np.diff(g4.omega) >= -1e-15)
    assert np.all(g4.omega >= -1e-15)
