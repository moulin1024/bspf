# SPDX-License-Identifier: GPL-3.0-or-later
"""Compressible Navier--Stokes in a closed adiabatic tube using BSPF Galerkin.

Evolve nodal coefficients of (rho, rho*u, total energy density) in the BSPF
cardinal trial space. Composite Gauss quadrature evaluates nonlinear fluxes.
Physical stress tau=4*mu*u_x/3 and conduction kappa*T_x; no mass diffusion,
artificial viscosity, filtering, finite-volume fluxes or shock enrichment.
Rigid insulated walls: momentum coefficients zero at both endpoints; natural
zero mass/energy boundary fluxes. Wall forces can change total momentum.
Sod primitive states are joined by a fixed-width tanh, not an initial jump.

The default backend uses structured FFT quadrature and low-rank mass inverses.
--backend dense retains the original quadrature/Cholesky reference. Both use
the same BSPF trial space and quadrature rule, including spline-knot splits.
Requires numpy, scipy and the compiled bspf_cpp module (matplotlib for plots).
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from scipy.sparse.linalg import LinearOperator, eigsh

import bspf_cpp as bspf

# No fit penalty: exact constant reproduction is needed for equilibrium and
# conservation in a nonlinear Galerkin solve (regularized fits need not do so).
OPTIONS = dict(degree=9, n_basis=18, lam=0., endpoint_method="chebyshev",
               boundary_points=16, chebyshev_modes=12, chebyshev_alpha=1e-12,
               constraint_order=4)


class NonphysicalState(ValueError):
    pass


def primitives(state, gamma=1.4):
    if not np.all(np.isfinite(state)) or np.min(state[..., 0]) <= 0:
        raise NonphysicalState("nonpositive density or nonfinite state")
    rho = state[..., 0]
    velocity = state[..., 1]/rho
    pressure = (gamma-1)*(state[..., 2]-.5*state[..., 1]*velocity)
    if np.min(pressure) <= 0:
        raise NonphysicalState("nonpositive pressure")
    return rho, velocity, pressure, pressure/rho  # R=1


def initial_state(x, width=.02, gamma=1.4):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("initial width must be finite and positive")
    left = .5*(1-np.tanh((x-.5)/width))
    rho = .125+.875*left
    pressure = .1+.9*left
    return np.column_stack((rho, np.zeros_like(x), pressure/(gamma-1)))


def fluxes(state, gradient, mu, kappa, gamma=1.4):
    """Physical inviscid and viscous conservative fluxes at quadrature nodes."""
    rho, u, p, temperature = primitives(state, gamma)
    ux = (gradient[:, 1]-u*gradient[:, 0])/rho
    tx = (gamma-1)*((gradient[:, 2]-state[:, 2]*gradient[:, 0]/rho)/rho-u*ux)
    tau = (4/3)*mu*ux
    convective = np.column_stack((state[:, 1], state[:, 1]*u+p, u*(state[:, 2]+p)))
    viscous = np.column_stack((np.zeros_like(rho), tau, u*tau+kappa*tx))
    return convective, viscous


class BSPFNavierStokes:
    def __init__(self, n=65, mu=.01, prandtl=.72, gamma=1.4, quadrature=8, backend="fft", line_options=None):
        if n < 25 or quadrature < 4:
            raise ValueError("n >=25 and quadrature >=4 required")
        if not all(np.isfinite(v) and v > 0 for v in (mu, prandtl)) or not np.isfinite(gamma) or gamma <= 1:
            raise ValueError("positive finite mu, Pr and gamma >1 required")
        if backend not in ("fft", "dense"):
            raise ValueError("backend must be fft or dense")
        start = perf_counter()
        self.n, self.mu, self.prandtl, self.gamma = n, mu, prandtl, gamma
        self.kappa = mu*gamma/((gamma-1)*prandtl)  # cp=gamma/(gamma-1), R=1
        self.x = np.linspace(0., 1., n)
        options = dict(OPTIONS)
        options.update(line_options or {})
        self.line_options = options
        self.line = bspf.Line(self.x, **options)
        self.backend = backend
        if backend == "fft":
            self.galerkin = bspf.GalerkinLine(self.line, quadrature_order=quadrature)
            self.q = self.galerkin.quadrature_points
            self.weights = self.galerkin.quadrature_weights
            def operator(action):
                return LinearOperator((n, n), matvec=lambda v: action(np.asarray(v).reshape(n, 1)).ravel(), dtype=float)
            # Generalized Lanczos, using only matrix-free M, K and M^-1 actions.
            self.diffusion_radius = float(eigsh(operator(self.galerkin.stiffness), k=1,
                M=operator(self.galerkin.mass), Minv=operator(self.galerkin.solve_mass),
                which="LA", tol=1e-10, v0=np.random.default_rng(0).normal(size=n),
                return_eigenvectors=False)[0])
            self.integral = self.project(np.ones((len(self.q), 1)))[:, 0]
        else:
            edges = np.unique(np.concatenate((self.x, self.line.knots)))
            z, weights = np.polynomial.legendre.leggauss(quadrature)
            width = np.diff(edges)
            self.q = (edges[:-1, None]+width[:, None]*(z+1)/2).ravel()
            self.weights = (width[:, None]*weights/2).ravel()
            self.V = self.line.evaluate(self.q).matrix
            self.G = self.line.evaluate(self.q, derivative=1).matrix
            self.Gw = self.G.T*self.weights
            self.M = self.V.T@(self.weights[:, None]*self.V)
            self.M = (self.M+self.M.T)/2
            stiffness = self.Gw@self.G
            stiffness = (stiffness+stiffness.T)/2
            self.mass_factor = cho_factor(self.M, lower=True)
            self.momentum_factor = cho_factor(self.M[1:-1, 1:-1], lower=True)
            self.diffusion_radius = float(eigvalsh(stiffness, self.M, subset_by_index=[n-1, n-1])[0])
            self.integral = self.weights@self.V
        wall_load = np.zeros((n, 2))
        wall_load[[0, -1]] = np.eye(2)
        wall_response = self.solve_mass(wall_load)
        self.wall_lift = np.linalg.solve(wall_response[[0, -1]].T, wall_response.T).T
        self.quadrature = quadrature
        self.setup_seconds = perf_counter()-start

    def evaluate(self, state, derivative=0):
        if self.backend == "fft":
            return self.galerkin.evaluate(state, derivative)
        return (self.V if derivative == 0 else self.G)@state

    def project(self, values, derivative=0):
        if self.backend == "fft":
            return self.galerkin.project(values, derivative)
        return (self.V.T*self.weights if derivative == 0 else self.Gw)@values

    def solve_mass(self, rhs, interior=False):
        if self.backend == "fft":
            return self.galerkin.solve_mass(rhs, interior)
        return cho_solve(self.momentum_factor if interior else self.mass_factor, rhs, check_finite=False)

    def rhs(self, state, sampled=None):
        primitives(state, self.gamma)
        if sampled is None:
            sampled = self.evaluate(state)
        convective, viscous = fluxes(sampled, self.evaluate(state, 1), self.mu, self.kappa, self.gamma)
        residual = self.project(convective-viscous, 1)
        rate = self.solve_mass(residual)
        rate[:, 1] -= self.wall_lift@rate[[0, -1], 1]
        rate[[0, -1], 1] = 0.
        return rate

    def timestep(self, state, cfl, sampled=None):
        if sampled is None:
            sampled = self.evaluate(state)
        rho, u, p, _ = primitives(sampled, self.gamma)
        wavespeed = float(np.max(np.abs(u)+np.sqrt(self.gamma*p/rho)))
        diffusivity = max(4*self.mu/3, self.kappa*(self.gamma-1))/float(np.min(rho))
        # Conservative heuristic combining RK4 imaginary/negative-real limits.
        # State checks/rejection supplement this estimate; no nonlinear stability theorem.
        return cfl/(wavespeed*np.sqrt(self.diffusion_radius)/2.83
                    + diffusivity*self.diffusion_radius/2.785293563405282)

    def diagnostics(self, state, sampled=None):
        if sampled is None:
            sampled = self.evaluate(state)
        rho, _, p, _ = primitives(sampled, self.gamma)
        entropy = float(self.weights@(rho*np.log(p/rho**self.gamma)/(self.gamma-1)))
        totals = self.integral@state
        return dict(mass=float(totals[0]), momentum=float(totals[1]), energy=float(totals[2]),
                    entropy=entropy, min_density=float(rho.min()), min_pressure=float(p.min()))

    def initial_coefficients(self, width):
        return initial_state(self.x, width, self.gamma)

    def solve(self, final_time=.2, width=.02, cfl=.5, progress=False):
        if not np.isfinite(final_time) or final_time <= 0 or not np.isfinite(cfl) or not 0 < cfl <= 1:
            raise ValueError("positive finite final time and 0<cfl<=1 required")
        state = self.initial_coefficients(width)
        sampled = self.evaluate(state)
        initial = self.diagnostics(state, sampled)
        history = [dict(time=0., **initial)]
        t, steps, rejected = 0., 0, 0
        min_dt, max_dt = float("inf"), 0.
        next_record = final_time/40
        start = perf_counter()
        last_report = start
        # Preserve overrides used by the regularization experiments. The plain
        # solver can reuse accepted samples for both CFL and the first RK stage.
        plain_rhs = type(self).rhs is BSPFNavierStokes.rhs
        plain_timestep = type(self).timestep is BSPFNavierStokes.timestep
        while t < final_time:
            stable_dt = self.timestep(state, cfl, sampled) if plain_timestep else self.timestep(state, cfl)
            dt = min(stable_dt, final_time-t)
            for attempt in range(13):
                try:
                    k1 = self.rhs(state, sampled) if plain_rhs else self.rhs(state)
                    k2 = self.rhs(state+.5*dt*k1)
                    k3 = self.rhs(state+.5*dt*k2)
                    k4 = self.rhs(state+dt*k3)
                    trial = state+dt*(k1+2*k2+2*k3+k4)/6
                    primitives(trial, self.gamma)
                    trial_samples = self.evaluate(trial)
                    primitives(trial_samples, self.gamma)
                    break
                except NonphysicalState:
                    rejected += 1
                    dt *= .5
            else:
                raise RuntimeError(f"nonphysical stages persist at t={t:g}; refine the mesh or initial layer")
            if t+dt == t:
                raise RuntimeError("time step underflow")
            state, sampled = trial, trial_samples
            t = min(final_time, t+dt)
            steps += 1
            min_dt, max_dt = min(min_dt, dt), max(max_dt, dt)
            if t >= next_record or t == final_time:
                history.append(dict(time=t, **self.diagnostics(state, sampled)))
                next_record += final_time/40
            if progress and perf_counter()-last_report > 15:
                print(f"n={self.n}: t={t:.5f}/{final_time:g}, steps={steps}, dt={dt:.2e}", flush=True)
                last_report = perf_counter()
        elapsed = perf_counter()-start
        final = self.diagnostics(state, sampled)
        probes = np.linspace(0, 1, 2001)
        output = self.line.evaluate(probes)(state)
        primitives(output, self.gamma)
        mass_drift = max(abs(h["mass"]-initial["mass"]) for h in history)/abs(initial["mass"])
        energy_drift = max(abs(h["energy"]-initial["energy"]) for h in history)/abs(initial["energy"])
        summary = dict(n=self.n, backend=self.backend, mu=self.mu, prandtl=self.prandtl, gamma=self.gamma, kappa=self.kappa,
                       mass_rank=self.galerkin.mass_rank if self.backend == "fft" else None,
                       final_time=final_time, initial_width=width, cfl=cfl, quadrature=self.quadrature,
                       steps=steps, rejected_stages=rejected, setup_seconds=self.setup_seconds,
                       solve_seconds=elapsed, dt_min=min_dt, dt_max=max_dt,
                       diffusion_radius=self.diffusion_radius, initial=initial, final=final,
                       max_relative_mass_drift=mass_drift, max_relative_energy_drift=energy_drift,
                       entropy_change=final["entropy"]-initial["entropy"],
                       min_recorded_entropy_increment=float(np.min(np.diff([h["entropy"] for h in history]))))
        return dict(summary=summary, history=history, x=self.x, state=state, probes=probes, output=output)


def plot_results(results, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for result in results:
        summary, x = result["summary"], result["probes"]
        rho, velocity, pressure, temperature = primitives(result["output"], summary["gamma"])
        for ax, values, title in zip(axes[:4], (rho, velocity, pressure, temperature), ("Density", "Velocity", "Pressure", "Temperature")):
            ax.plot(x, values, label=f"n={summary['n']}")
            ax.set(xlabel="x", title=title)
        history = result["history"]
        times = [h["time"] for h in history]
        axes[4].plot(times, [h["entropy"]-history[0]["entropy"] for h in history], label=f"n={summary['n']}")
        drift = [max(abs(h["mass"]/history[0]["mass"]-1), abs(h["energy"]/history[0]["energy"]-1), 1e-17) for h in history]
        axes[5].semilogy(times, drift, label=f"n={summary['n']}")
    gamma = results[-1]["summary"]["gamma"]
    initial = initial_state(results[-1]["probes"], results[-1]["summary"]["initial_width"], gamma)
    for ax, values in zip(axes[:4], primitives(initial, gamma)):
        ax.plot(results[-1]["probes"], values, "k--", lw=.8, label="initial")
    axes[4].set(xlabel="Time", ylabel="Integrated physical entropy increase", title="Entropy production")
    axes[5].set(xlabel="Time", ylabel="Max relative mass / energy drift", title="Closed-tube conservation")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    s = results[-1]["summary"]
    fig.suptitle(f"BSPF Galerkin ({s['backend']}) compressible Navier–Stokes: t={s['final_time']:g}, μ={s['mu']:g}, Pr={s['prandtl']:g}\nRigid adiabatic walls; physical viscosity and heat conduction only")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[513])
    parser.add_argument("--mu", type=float, default=.00001)
    parser.add_argument("--prandtl", type=float, default=.72)
    parser.add_argument("--width", type=float, default=.02)
    parser.add_argument("--final-time", type=float, default=.1)
    parser.add_argument("--cfl", type=float, default=.5)
    parser.add_argument("--quadrature", type=int, default=8)
    parser.add_argument("--backend", choices=("fft", "dense"), default="fft")
    parser.add_argument("--json")
    parser.add_argument("--npz")
    parser.add_argument("--plot")
    args = parser.parse_args()
    results = []
    for n in sorted(set(args.sizes)):
        solver = BSPFNavierStokes(n, args.mu, args.prandtl, quadrature=args.quadrature, backend=args.backend)
        result = solver.solve(args.final_time, args.width, args.cfl, progress=True)
        results.append(result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    comparisons = []
    if len(results) > 1:
        reference = np.column_stack(primitives(results[-1]["output"]))
        for result in results[:-1]:
            primitive = np.column_stack(primitives(result["output"]))
            l1 = np.trapezoid(np.abs(primitive-reference), result["probes"], axis=0)
            comparisons.append(dict(n=result["summary"]["n"], reference_n=results[-1]["summary"]["n"],
                                    primitive_l1_difference=dict(zip(("rho", "u", "p", "T"), l1.tolist()))))
        print("Differences from finest grid (not exact-solution errors):", json.dumps(comparisons), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(model=__doc__, bspf_options=OPTIONS,
            runs=[dict(summary=r["summary"], history=r["history"]) for r in results], comparisons=comparisons), indent=2)+"\n")
    if args.npz:
        np.savez_compressed(args.npz, **{f"{key}_{r['summary']['n']}": r[key] for r in results for key in ("x", "state", "probes", "output")})
    if args.plot:
        plot_results(results, args.plot)


if __name__ == "__main__":
    main()
