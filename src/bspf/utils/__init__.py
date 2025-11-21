"""
Utility functions and time steppers for BSPF.
"""

from .time_steppers import (
    TimeStepperState,
    time_step,
    rk4_step,
    rk45_step,
    rk23_step,
    bdf2_step,
    create_time_stepper
)

# Chebyshev utilities
from .chebyshev import (
    construct_chebyshev_nodes,
    chebyshev_derivative_from_values
)

# Chebyshev integration utilities
from .chebyshev_integral import (
    chebyshev_antiderivatives_fft
)

# Padé finite difference utilities
try:
    from .padefd import (
        padefd,
        derive_tridiag_compact_coeffs,
        build_schemes_table
    )
    _HAS_PADEFD = True
except ImportError:
    # Optional dependency (requires findiff and scipy.fft)
    padefd = None
    derive_tridiag_compact_coeffs = None
    build_schemes_table = None
    _HAS_PADEFD = False

__all__ = [
    "TimeStepperState",
    "time_step",
    "rk4_step",
    "rk45_step",
    "rk23_step",
    "bdf2_step",
    "create_time_stepper",
    "construct_chebyshev_nodes",
    "chebyshev_derivative_from_values",
    "chebyshev_antiderivatives_fft",
]

if _HAS_PADEFD:
    __all__.extend([
        "padefd",
        "derive_tridiag_compact_coeffs",
        "build_schemes_table"
    ])

