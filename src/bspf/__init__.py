"""Top-level package exports for the BSPF library."""

from .bspf1d import bspf1d
from .bspf2d import bspf2d
from .bspf3d import bspf3d
from .time_steppers import TimeStepperState, time_step, rk4_step, rk45_step, rk23_step, bdf2_step, create_time_stepper
from .schrodinger_solver import (
    create_schrodinger_rhs, 
    create_dirichlet_bc_enforcer,
    create_neumann_bc_enforcer,
    solve_schrodinger
)
from .animation_1d import SchrodingerAnimator1D, create_comparison_plot_config

# Try to import GPU version if available
try:
    from .bspf1d_gpu import bspf1d as bspf1d_gpu
    __all__ = [
        "bspf1d", "bspf2d", "bspf3d", "bspf1d_gpu",
        "TimeStepperState", "time_step", "rk4_step", "rk45_step", "rk23_step", "bdf2_step", "create_time_stepper",
        "create_schrodinger_rhs", "create_dirichlet_bc_enforcer", "create_neumann_bc_enforcer", "solve_schrodinger",
        "SchrodingerAnimator1D", "create_comparison_plot_config"
    ]
except ImportError:
    __all__ = [
        "bspf1d", "bspf2d", "bspf3d",
        "TimeStepperState", "time_step", "rk4_step", "rk45_step", "rk23_step", "bdf2_step", "create_time_stepper",
        "create_schrodinger_rhs", "create_dirichlet_bc_enforcer", "create_neumann_bc_enforcer", "solve_schrodinger",
        "SchrodingerAnimator1D", "create_comparison_plot_config"
    ]
