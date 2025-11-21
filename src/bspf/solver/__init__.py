"""
Application-specific solvers and utilities built on top of BSPF.

This submodule contains domain-specific solvers (e.g., Schrödinger equation, heat equation)
that use the core BSPF functionality but are not part of the core library.
"""

from .schrodinger import (
    create_schrodinger_rhs,
    create_dirichlet_bc_enforcer,
    create_neumann_bc_enforcer,
    solve_schrodinger
)

from .heat import (
    create_heat_rhs_1d,
    build_laplacian_matrix_1d,
    create_heat_jacobian_1d
)

__all__ = [
    "create_schrodinger_rhs",
    "create_dirichlet_bc_enforcer",
    "create_neumann_bc_enforcer",
    "solve_schrodinger",
    "create_heat_rhs_1d",
    "build_laplacian_matrix_1d",
    "create_heat_jacobian_1d"
]

