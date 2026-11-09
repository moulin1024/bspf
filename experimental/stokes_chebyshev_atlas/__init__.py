"""General 2D Stokes-guided Chebyshev block-map prototype."""
from .atlas import StokesGuide, Settings, Atlas, Result, generate, diagnose, smooth_atlas
from .topology import Geometry
from .adapters import from_arnoldi_solver
from .laplacian import MappedLaplacian

__all__ = ['Geometry', 'StokesGuide', 'Settings', 'Atlas', 'Result', 'generate', 'diagnose',
           'from_arnoldi_solver', 'smooth_atlas', 'MappedLaplacian']
