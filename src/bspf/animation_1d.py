"""
1D animation utilities for Schrödinger equation simulations.

Provides reusable functions for creating animations and saving plots
for 1D quantum simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Callable, Optional, Dict, List, Tuple
from tqdm import tqdm


class SchrodingerAnimator1D:
    """
    Modular animator for 1D Schrödinger equation simulations.
    
    Supports:
    - Video animations (MP4)
    - Static plot saving
    - Customizable plot configurations
    - Multiple plot types (|ψ|², error, real/imaginary parts, potential)
    """
    
    def __init__(self, x: np.ndarray, 
                 figsize: Tuple[float, float] = (10, 8),
                 fps: int = 30,
                 dpi: int = 100):
        """
        Initialize animator.
        
        Parameters:
        -----------
        x : array
            Spatial grid
        figsize : tuple, optional
            Figure size (width, height). Default: (10, 8)
        fps : int, optional
            Frames per second for video. Default: 30
        dpi : int, optional
            DPI for saved plots. Default: 100
        """
        self.x = x
        self.figsize = figsize
        self.fps = fps
        self.dpi = dpi
        self.fig = None
        self.axes = None
        self.lines = {}
        self.plots = []
        
    def setup_plot(self, plot_config: Dict):
        """
        Setup plot configuration.
        
        Parameters:
        -----------
        plot_config : dict
            Configuration dictionary with keys:
            - 'layout': 'single' or 'double' (single or two subplots)
            - 'plots': list of plot specifications, each with:
                - 'type': 'density', 'error', 'real', 'imag', 'potential', 'custom'
                - 'ax': which axis (0 or 1 for double layout)
                - 'label': plot label
                - 'color': line color
                - 'style': line style ('-', '--', etc.)
                - 'linewidth': line width
                - 'alpha': transparency
                - 'ylim': y-axis limits (optional)
                - 'yscale': y-axis scale ('linear' or 'log', optional)
                - 'xlabel': x-axis label (optional)
                - 'ylabel': y-axis label (optional)
                - 'title_template': title template with {t} placeholder (optional)
                - 'callback': custom function to compute plot data (for 'custom' type)
        """
        self.plot_config = plot_config
        layout = plot_config.get('layout', 'single')
        
        if layout == 'double':
            self.fig, self.axes = plt.subplots(2, 1, figsize=self.figsize)
            self.axes = list(self.axes)
        else:
            self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.axes = [ax]
        
        self.lines = {}
        self.plots = plot_config.get('plots', [])
        
        # Setup each plot
        for i, plot_spec in enumerate(self.plots):
            ax_idx = plot_spec.get('ax', 0)
            if ax_idx >= len(self.axes):
                continue
                
            ax = self.axes[ax_idx]
            plot_type = plot_spec.get('type', 'density')
            
            # Create line object
            line, = ax.plot([], [], 
                           label=plot_spec.get('label', ''),
                           color=plot_spec.get('color', 'blue'),
                           linestyle=plot_spec.get('style', '-'),
                           linewidth=plot_spec.get('linewidth', 2),
                           alpha=plot_spec.get('alpha', 1.0))
            self.lines[i] = line
            
            # Set axis properties
            ax.set_xlim(self.x[0], self.x[-1])
            if 'ylim' in plot_spec:
                ax.set_ylim(plot_spec['ylim'])
            if 'yscale' in plot_spec:
                ax.set_yscale(plot_spec['yscale'])
            if 'xlabel' in plot_spec:
                ax.set_xlabel(plot_spec['xlabel'])
            if 'ylabel' in plot_spec:
                ax.set_ylabel(plot_spec['ylabel'])
            else:
                # Default ylabel based on type
                if plot_type == 'density':
                    ax.set_ylabel('|ψ|²')
                elif plot_type == 'error':
                    ax.set_ylabel('Error')
                elif plot_type == 'real':
                    ax.set_ylabel('Re[ψ]')
                elif plot_type == 'imag':
                    ax.set_ylabel('Im[ψ]')
                elif plot_type == 'potential':
                    ax.set_ylabel('V(x)')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def update_frame(self, t: float, psi: np.ndarray, 
                    psi_analytical: Optional[np.ndarray] = None,
                    V: Optional[np.ndarray] = None,
                    custom_data: Optional[Dict] = None):
        """
        Update animation frame with new data.
        
        Parameters:
        -----------
        t : float
            Current time
        psi : complex array
            Current wavefunction
        psi_analytical : array, optional
            Analytical solution for comparison
        V : array, optional
            Potential for plotting
        custom_data : dict, optional
            Custom data for custom plot types
        """
        custom_data = custom_data or {}
        
        for i, plot_spec in enumerate(self.plots):
            if i not in self.lines:
                continue
                
            line = self.lines[i]
            plot_type = plot_spec.get('type', 'density')
            
            # Compute data based on plot type
            if plot_type == 'density':
                data = np.abs(psi)**2
            elif plot_type == 'error':
                if psi_analytical is not None:
                    data = np.abs(psi - psi_analytical)
                    # Optionally hide boundary errors
                    if plot_spec.get('hide_boundaries', False):
                        data[0] = np.nan
                        data[-1] = np.nan
                else:
                    data = np.zeros_like(psi)
            elif plot_type == 'real':
                data = psi.real
                # Normalize if requested
                if plot_spec.get('normalize', False):
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val
            elif plot_type == 'imag':
                data = psi.imag
                # Normalize if requested
                if plot_spec.get('normalize', False):
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val
            elif plot_type == 'potential':
                if V is not None:
                    # Normalize if requested
                    if plot_spec.get('normalize', False):
                        V_max = np.max(np.abs(V))
                        if V_max > 0:
                            data = V / V_max
                        else:
                            data = V
                    else:
                        data = V
                else:
                    data = np.zeros_like(self.x)
            elif plot_type == 'custom':
                callback = plot_spec.get('callback')
                if callback:
                    data = callback(t, psi, psi_analytical, V, custom_data)
                else:
                    data = np.zeros_like(psi)
            else:
                data = np.zeros_like(psi)
            
            # Update line data
            line.set_data(self.x, data)
            
            # Update title if template provided
            ax_idx = plot_spec.get('ax', 0)
            if ax_idx < len(self.axes):
                ax = self.axes[ax_idx]
                title_template = plot_spec.get('title_template')
                if title_template:
                    ax.set_title(title_template.format(t=t))
    
    def animate_simulation(self, 
                          rhs_func: Callable,
                          psi_init: np.ndarray,
                          dt: float,
                          n_steps: int,
                          time_method: str = 'rk45',
                          output_path: str = 'animation.mp4',
                          enforce_bc: Optional[Callable] = None,
                          psi_analytical_func: Optional[Callable] = None,
                          V: Optional[np.ndarray] = None,
                          progress_callback: Optional[Callable] = None,
                          save_static: bool = False,
                          save_interval: int = 1,
                          static_output_dir: Optional[str] = None):
        """
        Animate a simulation.
        
        Parameters:
        -----------
        rhs_func : callable
            RHS function for time stepping: rhs_func(psi) -> dpsi_dt
        psi_init : array
            Initial wavefunction
        dt : float
            Time step
        n_steps : int
            Number of time steps
        time_method : str, optional
            Time stepping method. Default: 'rk45'
        output_path : str, optional
            Output path for video. Default: 'animation.mp4'
        enforce_bc : callable, optional
            Function to enforce boundary conditions: enforce_bc(psi) -> psi_bc
        psi_analytical_func : callable, optional
            Function to compute analytical solution: psi_analytical_func(t) -> psi_ana
        V : array, optional
            Potential array for plotting
        progress_callback : callable, optional
            Callback function: progress_callback(n, t, psi, error)
        save_static : bool, optional
            Whether to save static plots. Default: False
        save_interval : int, optional
            Interval for saving static plots. Default: 1
        static_output_dir : str, optional
            Directory for static plots. Default: None (uses current directory)
        """
        from bspf import TimeStepperState, time_step
        
        # Initialize
        psi = psi_init.copy()
        if enforce_bc:
            psi = enforce_bc(psi)
        
        # Setup video writer (only if output_path is provided)
        if output_path:
            writer = FFMpegWriter(fps=self.fps)
            writer_context = writer.saving(self.fig, output_path, self.dpi)
        else:
            # Dummy context manager if no video output
            from contextlib import nullcontext
            writer_context = nullcontext()
            writer = None
        
        with writer_context:
            # First frame: initial condition
            t = 0.0
            psi_ana = None
            if psi_analytical_func:
                psi_ana = psi_analytical_func(t)
            
            self.update_frame(t, psi, psi_ana, V)
            if writer:
                writer.grab_frame()
            
            if save_static:
                self._save_static_plot(t, psi, psi_ana, V, 0, static_output_dir)
            
            # Initialize time stepper
            T_final = n_steps * dt
            with TimeStepperState(psi.copy(), t_init=0.0, dt=dt, method=time_method,
                                 t_final=T_final, show_progress=False) as state:
                for n in tqdm(range(n_steps), desc="Rendering animation"):
                    t_current = (n + 1) * dt
                    
                    # Time step
                    psi_next = time_step(state, dt, rhs_func, method=time_method)
                    psi = state.get_current()
                    
                    # Enforce BCs
                    if enforce_bc:
                        psi = enforce_bc(psi)
                        state.psi_now = psi.copy()
                    
                    # Compute analytical solution if provided
                    psi_ana = None
                    if psi_analytical_func:
                        psi_ana = psi_analytical_func(t_current)
                    
                    # Update frame
                    self.update_frame(t_current, psi, psi_ana, V)
                    if writer:
                        writer.grab_frame()
                    
                    # Save static plot if requested
                    if save_static and (n + 1) % save_interval == 0:
                        self._save_static_plot(t_current, psi, psi_ana, V, n + 1, static_output_dir)
                    
                    # Progress callback
                    if progress_callback:
                        error = None
                        if psi_ana is not None:
                            error = np.abs(psi - psi_ana)
                        progress_callback(n + 1, t_current, psi, error)
        
        if output_path:
            print(f"Saved animation: {output_path}")
        return psi  # Return final wavefunction
    
    def _save_static_plot(self, t: float, psi: np.ndarray,
                         psi_analytical: Optional[np.ndarray],
                         V: Optional[np.ndarray],
                         step: int,
                         output_dir: Optional[str]):
        """Save a static plot snapshot."""
        import os
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create a copy of the figure for saving
        fig_save = plt.figure(figsize=self.figsize)
        if len(self.axes) == 2:
            ax1 = fig_save.add_subplot(2, 1, 1)
            ax2 = fig_save.add_subplot(2, 1, 2)
            axes_save = [ax1, ax2]
        else:
            ax1 = fig_save.add_subplot(1, 1, 1)
            axes_save = [ax1]
        
        # Recreate plots
        for i, plot_spec in enumerate(self.plots):
            ax_idx = plot_spec.get('ax', 0)
            if ax_idx >= len(axes_save):
                continue
                
            ax = axes_save[ax_idx]
            plot_type = plot_spec.get('type', 'density')
            
            # Compute data (same logic as update_frame)
            if plot_type == 'density':
                data = np.abs(psi)**2
                if psi_analytical is not None:
                    data_ana = np.abs(psi_analytical)**2
                    ax.plot(self.x, data_ana, '--', 
                           label=plot_spec.get('label', '').replace('Numerical', 'Analytical'),
                           color='black', linewidth=2, alpha=0.7)
            elif plot_type == 'error':
                if psi_analytical is not None:
                    data = np.abs(psi - psi_analytical)
                else:
                    data = np.zeros_like(psi)
            elif plot_type == 'real':
                data = psi.real
                # Normalize if requested
                if plot_spec.get('normalize', False):
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val
            elif plot_type == 'imag':
                data = psi.imag
                # Normalize if requested
                if plot_spec.get('normalize', False):
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val
            elif plot_type == 'potential':
                if V is not None:
                    # Normalize if requested
                    if plot_spec.get('normalize', False):
                        V_max = np.max(np.abs(V))
                        if V_max > 0:
                            data = V / V_max
                        else:
                            data = V
                    else:
                        data = V
                else:
                    data = np.zeros_like(self.x)
            elif plot_type == 'custom':
                callback = plot_spec.get('callback')
                if callback:
                    custom_data = {}
                    data = callback(t, psi, psi_analytical, V, custom_data)
                else:
                    data = np.zeros_like(psi)
            else:
                data = np.zeros_like(psi)
            
            # For potential plots, add fill_between (always, for visibility)
            if plot_type == 'potential' and V is not None:
                ax.fill_between(self.x, 0, data,
                               where=(data > 0),
                               color=plot_spec.get('color', 'red'),
                               alpha=0.4, zorder=0)
            
            ax.plot(self.x, data, 
                   label=plot_spec.get('label', ''),
                   color=plot_spec.get('color', 'blue'),
                   linestyle=plot_spec.get('style', '-'),
                   linewidth=plot_spec.get('linewidth', 2),
                   alpha=plot_spec.get('alpha', 1.0))
            
            ax.set_xlim(self.x[0], self.x[-1])
            if 'ylim' in plot_spec:
                ax.set_ylim(plot_spec['ylim'])
            if 'yscale' in plot_spec:
                ax.set_yscale(plot_spec['yscale'])
            if 'xlabel' in plot_spec:
                ax.set_xlabel(plot_spec['xlabel'])
            if 'ylabel' in plot_spec:
                ax.set_ylabel(plot_spec['ylabel'])
            
            title_template = plot_spec.get('title_template')
            if title_template:
                ax.set_title(title_template.format(t=t))
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Save
        filename = f"wavefunction_step_{step:06d}_t{t:.4f}.png"
        if output_dir:
            filename = os.path.join(output_dir, filename)
        fig_save.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig_save)
        print(f'Saved plot: {filename}')


def create_comparison_plot_config(title: str = "Schrödinger Solver",
                                 xlabel: str = "x",
                                 hide_boundary_errors: bool = False) -> Dict:
    """
    Create a standard comparison plot configuration (numerical vs analytical).
    
    Parameters:
    -----------
    title : str, optional
        Title template (use {t} for time). Default: "Schrödinger Solver"
    xlabel : str, optional
        X-axis label. Default: "x"
    hide_boundary_errors : bool, optional
        Whether to hide boundary errors in error plot. Default: False
    
    Returns:
    --------
    plot_config : dict
        Plot configuration dictionary
    """
    return {
        'layout': 'double',
        'plots': [
            {
                'type': 'density',
                'ax': 0,
                'label': 'Numerical',
                'color': 'blue',
                'style': '-',
                'linewidth': 2,
                'ylim': None,
                'xlabel': '',
                'ylabel': '|ψ|²',
                'title_template': f'{title}   t={{t:.3f}}'
            },
            {
                'type': 'error',
                'ax': 1,
                'label': '|ψ_num - ψ_ana|',
                'color': 'red',
                'style': '-',
                'linewidth': 2,
                'yscale': 'log',
                'ylim': (1e-16, 1e-1),
                'xlabel': xlabel,
                'ylabel': 'Error (log scale)',
                'hide_boundaries': hide_boundary_errors
            }
        ]
    }

