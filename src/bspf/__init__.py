"""Top-level package exports for the BSPF library."""

from .bspf1d import bspf1d
from .piecewise_bspf1d import PiecewiseBSPF1D
from .bspf2d import bspf2d
from .bspf3d import bspf3d
# Time steppers moved to utils submodule
try:
    from .utils.time_steppers import TimeStepperState, time_step, rk4_step, rk45_step, rk23_step, bdf2_step, create_time_stepper
except ImportError:
    # Fallback if utils submodule doesn't exist
    TimeStepperState = None
    time_step = None
    rk4_step = None
    rk45_step = None
    rk23_step = None
    bdf2_step = None
    create_time_stepper = None

# Schrödinger solver moved to solver submodule
try:
    from .solver.schrodinger import (
        create_schrodinger_rhs, 
        create_dirichlet_bc_enforcer,
        create_neumann_bc_enforcer,
        solve_schrodinger
    )
except ImportError:
    create_schrodinger_rhs = None
    create_dirichlet_bc_enforcer = None
    create_neumann_bc_enforcer = None
    solve_schrodinger = None
from .boundary_conditions import enforce_zero_flux_neumann_bc
from .animation_1d import SchrodingerAnimator1D, create_comparison_plot_config

# Try to import GPU version if available
try:
    from .bspf1d_gpu import bspf1d as bspf1d_gpu
    __all__ = [
        "bspf1d", "bspf2d", "bspf3d", "bspf1d_gpu", "PiecewiseBSPF1D",
        "TimeStepperState", "time_step", "rk4_step", "rk45_step", "rk23_step", "bdf2_step", "create_time_stepper",
        "create_schrodinger_rhs", "create_dirichlet_bc_enforcer", "create_neumann_bc_enforcer", "solve_schrodinger",
        "enforce_zero_flux_neumann_bc",
        "SchrodingerAnimator1D", "create_comparison_plot_config"
    ]
except ImportError:
    __all__ = [
        "bspf1d", "bspf2d", "bspf3d", "PiecewiseBSPF1D",
        "TimeStepperState", "time_step", "rk4_step", "rk45_step", "rk23_step", "bdf2_step", "create_time_stepper",
        "create_schrodinger_rhs", "create_dirichlet_bc_enforcer", "create_neumann_bc_enforcer", "solve_schrodinger",
        "enforce_zero_flux_neumann_bc",
        "SchrodingerAnimator1D", "create_comparison_plot_config"
    ]
