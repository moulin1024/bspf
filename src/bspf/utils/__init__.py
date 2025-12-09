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
    chebyshev_derivative_from_values,
    chebyshev_second_derivative_from_values
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

# Grid mapping utilities
from .grid_mapping import (
    logistic,
    build_multi_sigmoid_expr,
    transform_to_unit_interval,
    transform_from_unit_interval,
    validate_domain,
    build_expr_via_connections_with_values,
    create_adaptive_mapping,
    create_simple_mapping
)

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
    "chebyshev_second_derivative_from_values",
    "logistic",
    "build_multi_sigmoid_expr",
    "transform_to_unit_interval",
    "transform_from_unit_interval",
    "validate_domain",
    "build_expr_via_connections_with_values",
    "create_adaptive_mapping",
    "create_simple_mapping",
]

if _HAS_PADEFD:
    __all__.extend([
        "padefd",
        "derive_tridiag_compact_coeffs",
        "build_schemes_table"
    ])

