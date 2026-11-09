"""Tensor Chebyshev transforms and Euclidean adjoints, using FFT/DCT-I.

The last two axes are reference coordinates in increasing [-1, 1] order.
No collocation differentiation matrices are assembled.
"""
import numpy as np
from scipy.fft import dct


def nodes(n):
    if n < 2:
        raise ValueError("At least two Lobatto nodes are required")
    return -np.cos(np.pi * np.arange(n) / (n - 1))


def _ends(a, factor):
    a = a.copy()
    a[..., 0] *= factor
    a[..., -1] *= factor
    return a


def coefficients(a, axis=-1, adjoint=False):
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    if not adjoint:
        a = a[..., ::-1]
    a = dct(_ends(a, 1 / np.sqrt(2)), type=1, norm="ortho", workers=1)
    a = _ends(a, 1 / np.sqrt(2)) * np.sqrt(2 / (n - 1))
    if adjoint:
        a = a[..., ::-1]
    return np.moveaxis(a, -1, axis)


def evaluate(a, size, axis=-1, adjoint=False):
    a = np.moveaxis(a, axis, -1)
    if adjoint:
        m = a.shape[-1]
        if size > m:
            raise ValueError("Adjoint truncation size exceeds grid size")
        a = dct(_ends(a[..., ::-1], np.sqrt(2)), type=1, norm="ortho", workers=1)
        a = (_ends(a, np.sqrt(2)) * np.sqrt((m - 1) / 2))[..., :size]
    else:
        if size < a.shape[-1]:
            raise ValueError("Evaluation cannot truncate coefficients")
        p = np.zeros(a.shape[:-1] + (size,), dtype=a.dtype)
        p[..., :a.shape[-1]] = a
        a = dct(_ends(p, np.sqrt(2)), type=1, norm="ortho", workers=1)
        a = (_ends(a, np.sqrt(2)) * np.sqrt((size - 1) / 2))[..., ::-1]
    return np.moveaxis(a, -1, axis)


def derivative(a, axis=-1, adjoint=False):
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    b = np.zeros_like(a)
    if adjoint:
        a = a.copy()
        a[..., 0] *= .5
    for start in (1, 2):
        j = np.arange(start, n, 2)
        if adjoint:
            b[..., j] = 2 * j * np.cumsum(a[..., j - 1], axis=-1)
        else:
            b[..., j - 1] = np.cumsum((2 * j * a[..., j])[..., ::-1], axis=-1)[..., ::-1]
    if not adjoint:
        b[..., 0] *= .5
    return np.moveaxis(b, -1, axis)


def transform2(a, adjoint=False):
    return coefficients(coefficients(a, -1, adjoint), -2, adjoint)


def evaluate2(a, size, adjoint=False):
    return evaluate(evaluate(a, size, -1, adjoint), size, -2, adjoint)


def weights(n):
    k = np.arange(n)
    moments = np.zeros(n)
    moments[::2] = 2 / (1 - k[::2] ** 2)
    return coefficients(moments, adjoint=True)
