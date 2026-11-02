# SPDX-License-Identifier: GPL-3.0-or-later
"""Rounding error of bspf_cpp against a 40-digit evaluation of the same map
(finite-difference endpoint jets), compared with pybspf's own rounding error."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pybspf
import pytest

import mp_reference
from test_parity import field, pair, rel


@pytest.mark.parametrize("n,degree", [(48, 5), (128, 5), (128, 7)])
def test_rounds_like_pybspf(n, degree):
    x, line, plan = pair(n, dict(degree=degree))
    f = field(x, 1)
    for k in (1, 2, 4):
        exact = mp_reference.derivative(x, line.knots, f, k, degree=degree, q=degree - 1, p=degree)
        ours = rel(line.derivative(k)(f), exact)
        theirs = rel(pybspf.differentiate(plan, f, order=k), exact)
        # Within 3x pybspf's distance from the exact map (or plain double rounding).
        assert ours < max(3 * theirs, 1e-14), (k, ours, theirs)
