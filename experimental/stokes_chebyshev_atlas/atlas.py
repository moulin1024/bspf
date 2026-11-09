"""Stokes-guided, jointly optimized polynomial coordinate atlas (2D).

For unforced constant-viscosity Stokes, Phi=p/mu-i*omega is holomorphic.
q=Phi' supplies a four-way coordinate direction and a regularized scale.
It is a guide, not an invertible global coordinate map. At q=0, alignment
vanishes smoothly and the metric stays positive. Geometry and connectivity
enter independently; neither is inferred uniquely from a Stokes solution.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
from numpy.polynomial.chebyshev import chebval2d
from numpy.polynomial.legendre import leggauss
from .spectral import nodes, weights, transform2, evaluate2, derivative
from .smoothness import MapSmoothness
from .operator_barrier import OperatorBarrier
from .deformation import ReducedDeformation
from .topology import (quadrangulate, nodal_layout, insert_critical_points,
                       one_block_seed, rectilinear_seed, paired_seed, region_seed, coarsen, subdivide, bisect_single)


@dataclass
class StokesGuide:
    """Vectorized complex callbacks q=Phi' and dq=Phi'', with physical units.

    For a Goursat solution these are 4*f'' and 4*f'''. Include logarithmic
    terms in multiply connected domains. For conservative forcing, subtract
    its pressure potential first. Nonconservative forcing is not supported by
    this holomorphic construction. scale regularizes |q|; it is a model choice.
    """
    q: object
    dq: object
    scale: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Guide scale must be finite and positive")

    @classmethod
    def from_goursat(cls, fsecond, fthird, scale=1.):
        return cls(lambda z: 4 * fsecond(z), lambda z: 4 * fthird(z), scale)

    def sample(self, z):
        q = np.broadcast_to(np.asarray(self.q(z), complex), np.shape(z))
        dq = np.broadcast_to(np.asarray(self.dq(z), complex), np.shape(z))
        with np.errstate(over='ignore', invalid='ignore'):
            q, dq = q / self.scale, dq / self.scale
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise ValueError("Stokes guide is not finite on the current atlas")
        return q, dq


@dataclass
class Settings:
    degree: int = 8
    solution_degree: int = 24
    iterations: int = 200
    distortion_weight: float = 1.
    area_weight: float = .15
    alignment_weight: float = .05
    spectral_weight: float = .01
    interface_weight: float = 1.
    wall_orthogonality_weight: float = .1
    interface_spacing_weight: float = 0.
    operator_barrier_weight: float = 0.
    deformation_degree: int | None = None
    density_contrast: float = 3.
    jacobian_floor_fraction: float = .02
    max_condition: float = 20.
    max_scale_variation: float = 30.
    spectral_tolerance: float = 1e-5
    boundary_tolerance: float = 1e-8
    gradient_tolerance: float = 1e-6
    search_critical_points: bool = True
    quality_slack: float = 1.05
    max_refinements: int = 1
    max_blocks: int = 256
    prefer_affine_rectilinear: bool = True

    def __post_init__(self):
        if not isinstance(self.degree, int) or self.degree < 3:
            raise ValueError("degree must be an integer >= 3")
        if self.deformation_degree is not None and (not isinstance(self.deformation_degree,int)
                                                    or not 2<=self.deformation_degree<=self.degree):
            raise ValueError('deformation_degree must be an integer from 2 to map degree')
        if not isinstance(self.solution_degree, int) or self.solution_degree < self.degree:
            raise ValueError("solution_degree must be an integer >= map degree")
        if not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if not isinstance(self.max_refinements, int) or self.max_refinements < 0:
            raise ValueError('max_refinements must be a nonnegative integer')
        if not isinstance(self.max_blocks, int) or self.max_blocks < 1:
            raise ValueError('max_blocks must be a positive integer')
        if self.quality_slack < 1:
            raise ValueError('quality_slack must be >= 1')
        for key, val in asdict(self).items():
            if isinstance(val, (float, int)) and not np.isfinite(val):
                raise ValueError(f"{key} must be finite")
        for key in ('distortion_weight', 'area_weight', 'alignment_weight',
                    'spectral_weight', 'density_contrast', 'interface_weight',
                    'wall_orthogonality_weight', 'interface_spacing_weight', 'operator_barrier_weight'):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0 < self.jacobian_floor_fraction < 1:
            raise ValueError("jacobian_floor_fraction must lie in (0,1)")
        if self.max_condition < 1 or self.max_scale_variation < 1:
            raise ValueError("Condition and scale bounds must be >= 1")
        if min(self.spectral_tolerance, self.boundary_tolerance, self.gradient_tolerance) <= 0:
            raise ValueError("Tolerances must be positive")


class Atlas:
    def __init__(self, seed, degree):
        self.seed = seed
        self.degree = degree
        (self.positions, self.ids, self.free, self.edge_ids,
         self.edge_curves) = nodal_layout(seed, degree + 1)

    @property
    def coefficients(self):
        return transform2(self.positions[self.ids])

    def evaluate(self, count=None):
        """Return complex X, X_xi, X_eta on a Lobatto tensor grid."""
        c = self.coefficients
        m = count or self.degree + 1
        return (evaluate2(c, m), evaluate2(derivative(c, -2), m),
                evaluate2(derivative(c, -1), m))

    def save(self, path, report=None):
        """Portable coefficients/connectivity and embedded JSON diagnostics."""
        np.savez_compressed(path, coefficients=self.coefficients,
                            nodes=nodes(self.degree + 1), global_nodes=self.positions,
                            node_ids=self.ids, quads=self.seed.quads,
                            fixed=~self.free,
                            report=json.dumps(report or {}, allow_nan=False))


class AtlasObjective:
    """One global objective; interfaces share unknowns, not penalty copies.

    All map evaluation/differentiation and reverse accumulation use DCTs.
    Local constitutive derivatives use complex-step on real algebra (six
    independent local variables), then exact spectral adjoints. Stokes is
    evaluated only at physical points; its supplied dq supplies spatial forces.
    """
    def __init__(self, atlas, guide, settings):
        self.atlas, self.guide, self.settings = atlas, guide, settings
        self.n = atlas.degree + 1
        self.m = 2 * self.n + 3
        w = weights(self.m)
        self.w = w[:, None] * w[None, :] / (4 * len(atlas.ids))
        self.base = atlas.positions.copy()
        self.length = np.hypot(np.ptp(self.base.real), np.ptp(self.base.imag))
        z, a, b = atlas.evaluate(self.m)
        self.smoothness = MapSmoothness(atlas,a,b)
        det = np.imag(np.conj(a) * b)
        if not np.all(np.isfinite(det)) or np.min(det) <= 0:
            raise ValueError("Initial polynomial atlas folds; refine the geometry triangulation or boundary representation")
        self.floor = settings.jacobian_floor_fraction * float(np.min(det))
        _, hi, lo = _quality(a, b)
        # If the initial seed meets the requested bounds, every accepted step
        # must meet them. Otherwise allow recovery, but never worsen its initial
        # worst bounds; the final diagnostic still applies the requested limits.
        self.condition_bound = max(settings.max_condition, float(np.max(hi/lo))*(1+1e-10))
        self.variation_bound = max(settings.max_scale_variation,
                                   float(np.max(hi.max(axis=(1, 2))/lo.min(axis=(1, 2))))*(1+1e-10))
        q, _ = guide.sample(z)
        r = abs(q)**2
        density = 1 + settings.density_contrast * r / (1 + r)
        self.target = float(np.sum(self.w * det * density))
        k = np.arange(self.n)
        # Penalize high map modes; do not penalize affine/bilinear maps.
        order = np.maximum(k[:, None], k[None, :])
        self.modal_weights = np.maximum(order - 1, 0).astype(float)**6
        self.operator_barrier = (OperatorBarrier(atlas.coefficients,settings)
                                 if settings.operator_barrier_weight else None)

    def pack(self, positions):
        z = positions[self.atlas.free] / self.length
        return np.r_[z.real, z.imag]

    def unpack(self, x):
        z = self.base.copy()
        k = np.sum(self.atlas.free)
        z[self.atlas.free] = self.length * (x[:k] + 1j * x[k:])
        return z

    def local_energy(self, v):
        ax, ay, bx, by, qx, qy = v
        s = self.settings
        d = ax*by - ay*bx
        aa, bb = ax*ax + ay*ay, bx*bx + by*by
        r = qx*qx + qy*qy
        density = 1 + s.density_contrast * r / (1 + r)
        log_area = np.log(density * d / self.target)
        energy = s.distortion_weight * ((aa + bb) / (2*d) - 1)
        energy = energy + .5 * s.area_weight * log_area**2
        # sin^2(2 arg(q*a)) with confidence (|q|^2/(1+|q|^2))^2.
        # The rational form is smooth at q=0 and avoids choosing a branch of arg.
        for x, y, length2 in ((ax, ay, aa), (bx, by, bb)):
            u, v = qx*x - qy*y, qx*y + qy*x
            energy = energy + .5 * s.alignment_weight * 4*u*u*v*v / ((1+r)**2 * length2**2)
        return energy

    def __call__(self, x):
        positions = self.unpack(x)
        c = transform2(positions[self.atlas.ids])
        barrier_value,barrier_gradient = (self.operator_barrier(c) if self.operator_barrier
                                        else (0.,np.zeros_like(c)))
        if not np.isfinite(barrier_value):
            return np.inf,np.zeros_like(x)
        z = evaluate2(c, self.m)
        a = evaluate2(derivative(c, -2), self.m)
        b = evaluate2(derivative(c, -1), self.m)
        det, hi, lo = _quality(a, b)
        if (not np.all(np.isfinite(det)) or np.min(det) <= self.floor
                or np.max(hi/lo) > self.condition_bound
                or np.max(hi.max(axis=(1, 2))/lo.min(axis=(1, 2))) > self.variation_bound):
            return np.inf, np.zeros_like(x)
        q, dq = self.guide.sample(z)
        local = [a.real, a.imag, b.real, b.imag, q.real, q.imag]
        value = barrier_value + float(np.sum(self.w * self.local_energy(local)))
        stress = []
        for k in range(6):
            perturbed = list(local)
            perturbed[k] = local[k].astype(complex) + 1e-25j
            stress.append(self.w * self.local_energy(perturbed).imag / 1e-25)
        # Cauchy-Riemann chain rule through the physical guide q(X).
        f = ((stress[4]*dq.real + stress[5]*dq.imag)
             + 1j*(-stress[4]*dq.imag + stress[5]*dq.real))
        transpose = lambda v: evaluate2(v, self.n, adjoint=True)
        gc = barrier_gradient + transpose(f)
        gc += derivative(transpose(stress[0] + 1j*stress[1]), -2, adjoint=True)
        gc += derivative(transpose(stress[2] + 1j*stress[3]), -1, adjoint=True)
        seam_value,ga,gb = self.smoothness.energy(a,b,self.settings.interface_weight,
                                                self.settings.wall_orthogonality_weight,
                                                self.settings.interface_spacing_weight)
        value += seam_value
        gc += derivative(transpose(ga),-2,adjoint=True)+derivative(transpose(gb),-1,adjoint=True)
        weight = self.settings.spectral_weight / (len(c) * self.length**2)
        value += .5 * weight * float(np.sum(self.modal_weights * abs(c)**2))
        gc += weight * self.modal_weights * c
        gblocks = transform2(gc, adjoint=True)
        global_gradient = np.zeros(len(positions), dtype=complex)
        np.add.at(global_gradient, self.atlas.ids.ravel(), gblocks.ravel())
        g = global_gradient[self.atlas.free] * self.length
        return value, np.r_[g.real, g.imag]


def _quality(a, b):
    det = np.imag(np.conj(a)*b)
    total = abs(a)**2 + abs(b)**2
    largest2 = .5 * (total + np.sqrt(np.maximum(total**2 - 4*det**2, 0)))
    largest = np.sqrt(largest2)
    smallest = abs(det) / np.maximum(largest, np.finfo(float).tiny)
    return det, largest, smallest


def _tail(values, cutoff):
    c = transform2(values)
    mask = np.maximum.outer(np.arange(c.shape[-2]), np.arange(c.shape[-1])) >= cutoff
    return np.sqrt(np.sum(abs(c[..., mask])**2, axis=-1)
                   / np.maximum(np.sum(abs(c)**2, axis=(-2, -1)), 1e-300))


def diagnose(atlas, settings):
    """Oversampled and independent Gauss-grid checks, not interval certificates.

    Modal tails measure resolution of map/operator coefficients; they do not
    prove complex analyticity or spectral accuracy of a future PDE solution.
    """
    n = atlas.degree + 1
    c = atlas.coefficients
    m = max(3*n + 2, 2*settings.solution_degree + 5)
    z, a, b = atlas.evaluate(m)
    det, hi, lo = _quality(a, b)
    q, _ = leggauss(2*n + 3)
    xi, eta = np.meshgrid(q, q, indexing='ij')
    ag = np.array([chebval2d(xi, eta, v) for v in derivative(c, -2)])
    bg = np.array([chebval2d(xi, eta, v) for v in derivative(c, -1)])
    dg, hg, lg = _quality(ag, bg)
    detmin = np.minimum(det.min(axis=(1, 2)), dg.min(axis=(1, 2)))
    minimum_scale = np.minimum(lo.min(axis=(1, 2)), lg.min(axis=(1, 2)))
    maximum_scale = np.maximum(hi.max(axis=(1, 2)), hg.max(axis=(1, 2)))
    skew = np.max(abs(np.real(np.conj(a)*b))/np.maximum(abs(a)*abs(b),1e-300),axis=(1,2))
    gram = np.stack([abs(a)**2,np.sqrt(2)*np.real(np.conj(a)*b),abs(b)**2],axis=1)
    mean_gram = np.mean(gram,axis=(2,3),keepdims=True)
    metric_variation = np.max(np.sqrt(np.sum((gram-mean_gram)**2,axis=1)),axis=(1,2))
    metric_variation /= np.maximum(np.sqrt(np.sum(mean_gram**2,axis=1)).ravel(),1e-300)
    with np.errstate(divide='ignore', invalid='ignore'):
        condition = np.maximum((hi/lo).max(axis=(1, 2)), (hg/lg).max(axis=(1, 2)))
        variation = maximum_scale / minimum_scale
        # Weak Laplace coefficients det(J)*J^-1 J^-T; plus mass density det(J).
        operator = np.stack([det, abs(b)**2/det, -np.real(np.conj(a)*b)/det, abs(a)**2/det], axis=1)
    # Normalize mass and stiffness separately: they have different physical
    # units. Combine the three stiffness components so a zero cross coefficient
    # cannot turn roundoff into a failed relative test.
    oc = transform2(operator)
    cutoff = settings.solution_degree + 1
    mask = np.maximum.outer(np.arange(m), np.arange(m)) >= cutoff
    mass_tail = np.sqrt(np.sum(abs(oc[:, 0, mask])**2, axis=1)
                        / np.maximum(np.sum(abs(oc[:, 0])**2, axis=(1, 2)), 1e-300))
    stiffness_tail = np.sqrt(np.sum(abs(oc[:, 1:, mask])**2, axis=(1, 2))
                             / np.maximum(np.sum(abs(oc[:, 1:])**2, axis=(1, 2, 3)), 1e-300))
    metric_tail = np.maximum(mass_tail, stiffness_tail)
    # Remove translation before normalizing geometry coefficient tails.
    centered = z - c[:, :1, :1]
    map_tail = _tail(centered, cutoff)
    boundary_error = np.zeros(len(atlas.ids))
    trace_error = 0.
    edge_owners = {}
    t = (nodes(m) + 1) / 2
    for k, quad in enumerate(atlas.seed.quads):
        traces = [z[k, :, 0], z[k, -1, :], z[k, ::-1, -1], z[k, 0, ::-1]]
        for (u, v), trace in zip(zip(quad, np.roll(quad, -1)), traces):
            edge = tuple(sorted((u, v)))
            oriented = trace if u < v else trace[::-1]
            if edge in edge_owners:
                trace_error = max(trace_error, float(np.max(abs(oriented-edge_owners[edge]))))
            else:
                edge_owners[edge] = oriented
            if (u, v) in atlas.seed.boundary_curves:
                exact = atlas.seed.boundary_curves[u, v](t)
            elif (v, u) in atlas.seed.boundary_curves:
                exact = atlas.seed.boundary_curves[v, u](1-t)
            else:
                continue
            boundary_error[k] = max(boundary_error[k], float(np.max(abs(trace-exact))))
    length = np.hypot(np.ptp(atlas.positions.real), np.ptp(atlas.positions.imag))
    blocks = []
    for k in range(len(c)):
        failures = []
        if detmin[k] <= 0: failures.append('nonpositive_sampled_jacobian')
        if not np.isfinite(condition[k]) or condition[k] > settings.max_condition: failures.append('distortion')
        if not np.isfinite(variation[k]) or variation[k] > settings.max_scale_variation: failures.append('scale_variation')
        if not np.isfinite(map_tail[k]) or map_tail[k] > settings.spectral_tolerance: failures.append('map_tail')
        if not np.isfinite(metric_tail[k]) or metric_tail[k] > settings.spectral_tolerance: failures.append('operator_tail')
        if boundary_error[k] > settings.boundary_tolerance * length: failures.append('boundary_resolution')
        blocks.append(dict(block=k, minimum_jacobian=float(detmin[k]),
                           maximum_condition=float(condition[k]), scale_variation=float(variation[k]),
                           map_tail=float(map_tail[k]), operator_tail=float(metric_tail[k]),
                           mass_tail=float(mass_tail[k]), stiffness_tail=float(stiffness_tail[k]),
                           maximum_skew_cosine=float(skew[k]), relative_metric_variation=float(metric_variation[k]),
                           boundary_error=float(boundary_error[k]), failures=failures,
                           minimum_scale=float(minimum_scale[k])))
    accepted = bool(all(not b['failures'] for b in blocks) and trace_error < 1e-10 * length)
    return dict(accepted=accepted, validation='sampled, not certified',
                blocks=blocks, interface_trace_error=trace_error,
                smoothness=MapSmoothness(atlas,a,b).diagnose(a,b),
                map_degree=atlas.degree, solution_degree=settings.solution_degree,
                lobatto_endpoint_spacing=float(1-np.cos(np.pi/settings.solution_degree)),
                note='Geometry suitability only; no PDE error or stable RK step is inferred.')


@dataclass
class Result:
    atlas: Atlas
    report: dict

    def require_accepted(self):
        if not self.report['quality']['accepted']:
            failed = [b['block'] for b in self.report['quality']['blocks'] if b['failures']]
            raise RuntimeError(f"Atlas needs refinement or different topology; failing blocks: {failed}")
        return self.atlas


def _quality_guard(candidate, initial, settings):
    """Protect resolution and the worst local scales, not just average energy."""
    slack = settings.quality_slack
    for new, old in zip(candidate['blocks'], initial['blocks']):
        if new['minimum_jacobian'] <= 0:
            return False
        for name in ('maximum_condition', 'scale_variation'):
            if not np.isfinite(new[name]) or new[name] > old[name]*slack:
                return False
        if new['minimum_scale'] < old['minimum_scale']/slack:
            return False
        # A resolved initial map must stay resolved. An unresolved initial map
        # may improve, but the optimizer may not worsen its tail.
        for name in ('operator_tail', 'map_tail'):
            if not np.isfinite(new[name]) or new[name] > max(old[name], settings.spectral_tolerance)*(1+1e-10):
                return False
    return not initial['accepted'] or candidate['accepted']


def _optimize_seed(seed, guide, settings, initial_positions=None):
    atlas = Atlas(seed, settings.degree)
    if initial_positions is not None:
        if np.shape(initial_positions) != atlas.positions.shape:
            raise ValueError('Initial polynomial nodes do not match this topology and degree')
        atlas.positions = np.asarray(initial_positions,dtype=complex).copy()
    initial_quality = diagnose(atlas, settings)
    objective = AtlasObjective(atlas, guide, settings)
    if settings.deformation_degree is not None:
        objective = ReducedDeformation(objective,settings.deformation_degree)
    x = objective.pack(atlas.positions)
    initial, gradient = objective(x)
    history = [initial]
    # Explicit feasible Armijo line search, retaining L-BFGS curvature pairs.
    # Reject inverted trials before evaluating the physical Stokes guide.
    pairs = []
    status = 'iteration_limit'
    value = initial
    iterations = 0
    quality = initial_quality
    quality_rejections = 0
    for iteration in range(settings.iterations):
        if np.linalg.norm(gradient, np.inf) <= settings.gradient_tolerance:
            status = 'gradient_converged'
            break
        direction = gradient.copy()
        alphas = []
        for s, y, rho in reversed(pairs):
            alpha = rho * np.dot(s, direction)
            alphas.append(alpha)
            direction -= alpha * y
        if pairs:
            s, y, _ = pairs[-1]
            direction *= np.dot(s, y) / np.dot(y, y)
        for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
            direction += s * (alpha - rho * np.dot(y, direction))
        direction = -direction
        slope = np.dot(gradient, direction)
        if slope >= 0:
            pairs.clear()
            direction = -gradient
            slope = -np.dot(gradient, gradient)
        step = 1.
        for _ in range(45):
            trial_value, trial_gradient = objective(x + step*direction)
            if np.isfinite(trial_value) and trial_value <= value + 1e-4*step*slope:
                atlas.positions = objective.unpack(x + step*direction)
                trial_quality = diagnose(atlas, settings)
                if _quality_guard(trial_quality, initial_quality, settings):
                    break
                quality_rejections += 1
            step *= .5
        else:
            status = 'quality_limited' if quality_rejections else 'line_search_failed'
            break
        if np.linalg.norm(step*direction, np.inf) < 1e-12:
            status = 'quality_limited'
            break
        s = step*direction
        y = trial_gradient-gradient
        curvature = np.dot(s, y)
        if curvature > 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
            pairs.append((s, y, 1/curvature))
            pairs = pairs[-10:]
        x += s
        value, gradient = trial_value, trial_gradient
        quality = trial_quality
        history.append(value)
        iterations = iteration + 1
    atlas.positions = objective.unpack(x)
    quality = diagnose(atlas, settings)
    report = dict(settings=asdict(settings), initial_quality=initial_quality,
                  optimization=dict(status=status, iterations=iterations, initial_objective=initial,
                                    final_objective=value, gradient_inf=float(np.linalg.norm(gradient, np.inf)),
                                    history=history, quality_rejections=quality_rejections), quality=quality)
    return Result(atlas, report)


def smooth_atlas(atlas, guide, settings=None):
    """Continue joint optimization of an existing atlas without changing cuts.

    Geometry boundary values stay fixed. Interface direction matching is G1
    grid-line compatibility away from tapered endpoints, not a global C1
    certificate. Existing Jacobian and spectral acceptance guards still apply.
    """
    settings = settings or Settings(degree=atlas.degree,
                                    solution_degree=max(24,atlas.degree))
    if settings.degree != atlas.degree:
        raise ValueError('Smoothing must use the existing polynomial map degree')
    return _optimize_seed(atlas.seed,guide,settings,initial_positions=atlas.positions)


def generate(geometry, guide, settings=None):
    """Prefer affine aligned charts on rectilinear boundaries; otherwise search.

    Boundary reconstruction can discard all input interior triangle edges.
    Stokes zeros do not by themselves force a chart cut. Candidate layouts are
    otherwise ranked by block count. Map optimization cannot sacrifice resolution.
    Search is bounded and heuristic, not a global minimum-block certificate.
    """
    settings = settings or Settings()
    attempts, successful, rejected = [], [], []
    critical = []
    if settings.search_critical_points:
        _, critical = insert_critical_points(geometry, guide, insert=False)
        critical = [dict(r, inserted=False, role='diagnostic_only') for r in critical]

    def acceptable(seed):
        try:
            return diagnose(Atlas(seed, settings.degree), settings)['accepted']
        except (ValueError, FloatingPointError):
            return False

    def try_seed(seed, name, merges=()):
        if len(seed.quads) > settings.max_blocks:
            attempts.append(dict(layout=name, blocks=len(seed.quads), accepted=False, reason='block_budget'))
            return None
        try:
            probe = Atlas(seed,settings.degree)
            quality = diagnose(probe,settings)
            affine_aligned = (quality['accepted'] and all(b['maximum_skew_cosine']<1e-10 and
                              b['relative_metric_variation']<1e-10 for b in quality['blocks']))
            if settings.prefer_affine_rectilinear and affine_aligned:
                result = Result(probe,dict(settings=asdict(settings),initial_quality=quality,quality=quality,
                    optimization=dict(status='affine_aligned_preserved',iterations=0,initial_objective=None,
                                      final_objective=None,gradient_inf=None,history=[],quality_rejections=0,
                                      guide_used_for_map_optimization=False)))
            else:
                result = _optimize_seed(seed, guide, settings)
        except ValueError as exc:
            # Only an invalid geometric initializer is a recoverable candidate
            # failure. Invalid user-supplied Stokes callbacks must remain errors.
            if 'Initial polynomial atlas folds' not in str(exc):
                raise
            attempts.append(dict(layout=name, blocks=len(seed.quads), accepted=False, reason=str(exc)))
            return None
        passed = result.report['quality']['accepted']
        attempts.append(dict(layout=name, blocks=len(seed.quads), accepted=passed,
                             merges=list(merges), failed_criteria=sorted({f for b in result.report['quality']['blocks'] for f in b['failures']})))
        result.report['selected_layout'] = name
        (successful if passed else rejected).append(result)
        return result

    def finish(result):
        result.report.update(topology_rule='prefer affine aligned charts on rectilinear boundaries; otherwise coarsest accepted searched layout',
                             topology_attempts=attempts, critical_points=critical,
                             critical_search='centroid-seeded Newton diagnostics; not exhaustive; no forced cuts')
        return result

    # Aligned affine charts have zero shear and constant metric coefficients.
    # Prefer this structure to fewer skewed charts, and do not distort it merely
    # to reduce the secondary Stokes-guidance energy.
    if settings.prefer_affine_rectilinear:
        aligned = rectilinear_seed(geometry,settings.max_blocks)
        if aligned is not None:
            result = try_seed(aligned,'rectilinear_affine')
            if result is not None and result.report['quality']['accepted']:
                return finish(result)

    # One chart is the absolute lower bound. Do not construct a finer topology
    # when this candidate is already admissible, even if its guide has a zero.
    simple = one_block_seed(geometry)
    failed_seeds = []
    if simple is not None:
        result = try_seed(simple, 'boundary_one_block')
        if result is not None and result.report['quality']['accepted']:
            return finish(result)
        failed_seeds.append(('boundary_one_block', simple))

    seeds = []
    regions, region_search = region_seed(geometry, acceptable)
    if regions is not None:
        merged, changes = coarsen(regions, acceptable)
        seeds.append(('region_cover_and_merged', merged, changes))
    paired = paired_seed(geometry)
    if paired is not None:
        merged, changes = coarsen(paired, acceptable)
        seeds.append(('paired_and_merged', merged, changes))
    # Subdivision is now a fallback, not the definition of the requested mesh.
    fallback, changes = coarsen(quadrangulate(geometry), acceptable)
    seeds.append(('subdivision_and_merged', fallback, changes))
    for name, seed, merges in sorted(seeds, key=lambda v: len(v[1].quads)):
        if simple is not None and len(seed.quads) == 1:
            continue  # same boundary chart was already attempted
        if successful and len(seed.quads) >= min(len(r.atlas.ids) for r in successful):
            continue
        result = try_seed(seed, name, merges)
        if result is None or not result.report['quality']['accepted']:
            failed_seeds.append((name, seed))

    # Refine only rejected layouts and only while they could improve the best
    # accepted block count. Uniform four-way refinement preserves conformity.
    for level in range(settings.max_refinements):
        next_failed = []
        for name, seed in sorted(failed_seeds, key=lambda v: len(v[1].quads)):
            if len(seed.quads) == 1 and settings.max_blocks >= 2:
                if not successful or min(len(r.atlas.ids) for r in successful) > 2:
                    for axis in (0, 1):
                        try_seed(bisect_single(seed, axis), name+f'/bisected_{axis}')
            count = 4*len(seed.quads)
            if count > settings.max_blocks or (successful and count >= min(len(r.atlas.ids) for r in successful)):
                continue
            refined = subdivide(seed)
            label = name+'/refined'
            result = try_seed(refined, label)
            if result is None or not result.report['quality']['accepted']:
                next_failed.append((label, refined))
        failed_seeds = next_failed

    if successful:
        result = min(successful, key=lambda r: (len(r.atlas.ids), r.report['optimization']['final_objective'] or 0.))
        result.report['region_search'] = region_search
        return finish(result)
    if rejected:
        # Return the smallest rejected candidate transparently, never silently
        # relabel an unresolved map as suitable for the target solver.
        return finish(min(rejected, key=lambda r: len(r.atlas.ids)))
    raise ValueError('No valid initial atlas within the block/refinement budget; inspect geometry and topology_attempts')
