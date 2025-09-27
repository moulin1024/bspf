import numpy as np
import numpy.testing as npt
import pytest

from bspf.bspf1d import ResidualCorrection  # ← replace with your actual import

def _rfft_omega(n: int, dx: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)


def test_residual_correction_none_returns_zeros():
    n = 128
    r = np.random.default_rng(0).standard_normal(n)
    omega = _rfft_omega(n, dx=1.0 / n)
    out = ResidualCorrection.none(r, omega, kind="diff", order=1, n=n)
    npt.assert_array_equal(out, np.zeros(n))

# --- keep the imports and helpers as-is ---

def test_spectral_diff_matches_analytic_derivatives():
    n = 1024
    dx = 1.0 / n
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    k = 10
    r = np.sin(2.0 * np.pi * k * x)
    omega = _rfft_omega(n, dx)

    # 1st derivative (tight)
    d1 = ResidualCorrection.spectral(r, omega, kind="diff", order=1, n=n, x=x)
    d1_true = 2.0 * np.pi * k * np.cos(2.0 * np.pi * k * x)
    np.testing.assert_allclose(d1, d1_true, atol=1e-10, rtol=0)

    # 2nd derivative: allow FFT roundoff amplification (~1e-8)
    d2 = ResidualCorrection.spectral(r, omega, kind="diff", order=2, n=n, x=x)
    d2_true = -(2.0 * np.pi * k) ** 2 * np.sin(2.0 * np.pi * k * x)
    np.testing.assert_allclose(d2, d2_true, atol=1e-7, rtol=0)


def test_spectral_int_order1_zero_mean_and_nonzero_mean():
    # Zero-mean residual: r(x) = cos(2πkx) → ∫ r = (1/(2πk)) sin(2πkx), anchored so F(0)=0
    n = 4096
    dx = 1.0 / n
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    k = 7
    r = np.cos(2.0 * np.pi * k * x)
    omega = _rfft_omega(n, dx)

    F = ResidualCorrection.spectral(r, omega, kind="int", order=1, n=n, x=x)
    F_true = (1.0 / (2.0 * np.pi * k)) * np.sin(2.0 * np.pi * k * x)
    # anchored at left: F(0) == 0
    assert abs(F[0]) <= 1e-12
    npt.assert_allclose(F, F_true, atol=2e-10, rtol=0)

    # Non-zero mean residual: r(x) = 1 → ∫ r = x - x0 (with left anchoring to 0)
    r_const = np.ones_like(x)
    F_const = ResidualCorrection.spectral(r_const, omega, kind="int", order=1, n=n, x=x)
    npt.assert_allclose(F_const, x - x[0], atol=5e-13, rtol=0)


def test_spectral_int_order1_nonunit_domain_with_x_argument():
    # Domain [2,5], constant residual → integral is (x - 2), anchored at left
    n = 1024
    a, b = 2.0, 5.0
    x = np.linspace(a, b, n, endpoint=False)
    dx = (b - a) / n
    omega = _rfft_omega(n, dx)

    r_const = np.ones(n)
    F = ResidualCorrection.spectral(r_const, omega, kind="int", order=1, n=n, x=x)
    npt.assert_allclose(F, x - a, atol=5e-12, rtol=0)
    assert abs(F[0]) <= 1e-12


def test_spectral_int_order2_zero_mean_and_nonzero_mean():
    # Zero-mean residual
    n = 4096
    dx = 1.0 / n
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    k = 3
    r = np.cos(2.0 * np.pi * k * x)
    omega = _rfft_omega(n, dx)

    # Double integral via implementation
    F2 = ResidualCorrection.spectral(r, omega, kind="int", order=2, n=n, x=x)

    # Robust check: differentiate back twice -> should recover r
    r_back = ResidualCorrection.spectral(F2, omega, kind="diff", order=2, n=n, x=x)
    np.testing.assert_allclose(r_back, r, atol=3e-8, rtol=0)

    # Non-zero mean residual: r(x) = 1 → parabolic with zero endpoints
    r_const = np.ones_like(x)
    F2_const = ResidualCorrection.spectral(r_const, omega, kind="int", order=2, n=n, x=x)
    q_true = 0.5 * (x - x[0]) * (x - x[-1])
    np.testing.assert_allclose(F2_const, q_true, atol=5e-12, rtol=0)
    assert abs(F2_const[0]) <= 1e-12 and abs(F2_const[-1]) <= 1e-12


def test_spectral_raises_on_invalid_kind_or_order():
    n = 64
    dx = 1.0 / n
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    omega = _rfft_omega(n, dx)
    r = np.random.default_rng(1).standard_normal(n)

    with pytest.raises(ValueError, match="must be 'diff' or 'int'"):
        ResidualCorrection.spectral(r, omega, kind="weird", order=1, n=n, x=x)

    with pytest.raises(ValueError, match="Only int orders 1 and 2 are supported"):
        ResidualCorrection.spectral(r, omega, kind="int", order=3, n=n, x=x)
