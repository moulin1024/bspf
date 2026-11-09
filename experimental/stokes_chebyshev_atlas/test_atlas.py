"""Implementation invariants; no geometry-specific physical benchmark."""
import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval2d
from . import Geometry, StokesGuide, Settings, Atlas, generate, diagnose, from_arnoldi_solver
from .atlas import AtlasObjective
from .topology import (quadrangulate, insert_critical_points, QuadSeed, one_block_seed,
                       coarsen, subdivide, paired_seed, region_seed, rectilinear_seed, bisect_single)
from .spectral import nodes, weights, transform2, evaluate2, derivative


def geometry(curved=False):
    curves = {(0, 1): lambda t: np.asarray(t) - .08j*np.sin(np.pi*np.asarray(t))} if curved else {}
    return Geometry([0, 1, 1+1j, 1j], np.array([[0, 1, 2], [0, 2, 3]]), curves)


def guide():
    return StokesGuide(lambda z: .2 + .3*np.asarray(z), lambda z: np.full(np.shape(z), .3), scale=.5)


def test_fft_polynomial_and_adjoint():
    rng = np.random.default_rng(721)
    for n in (4, 7, 10):
        m = 2*n+3
        a = rng.normal(size=(2, n, n)) + 1j*rng.normal(size=(2, n, n))
        c = transform2(a)
        assert np.max(abs(evaluate2(c, n)-a)) < 2e-13
        q = nodes(m)
        x, y = np.meshgrid(q, q, indexing='ij')
        assert np.max(abs(evaluate2(c, m)-np.array([chebval2d(x, y, p) for p in c]))) < 2e-13
        v = rng.normal(size=(2, m, m))
        lhs = np.vdot(evaluate2(c, m), v)
        rhs = np.vdot(c, evaluate2(v, n, adjoint=True))
        assert abs(lhs-rhs) < 1e-11
        b = rng.normal(size=a.shape)
        assert abs(np.vdot(transform2(a), b)-np.vdot(a, transform2(b, adjoint=True))) < 1e-11
        for axis in (-1, -2):
            assert abs(np.vdot(derivative(c, axis), b)-np.vdot(c, derivative(b, axis, True))) < 1e-10
        assert abs(weights(m).sum()-2) < 1e-14
        assert np.all(weights(m)>0)
    x, y = np.meshgrid(nodes(9), nodes(9), indexing='ij')
    c = transform2(x**3*y**2)
    assert np.max(abs(evaluate2(derivative(c, -2), 9)-3*x*x*y*y)) < 1e-12
    assert np.max(abs(evaluate2(derivative(c, -1), 9)-2*x**3*y)) < 1e-12


def test_topology_and_reversed_interface():
    a = Atlas(quadrangulate(geometry()), 6)
    assert a.ids.shape == (6, 7, 7)
    report = diagnose(a, Settings(degree=6))
    assert report['interface_trace_error'] < 1e-13
    # Perturbing shared unknowns preserves complete traces, not just endpoints.
    rng = np.random.default_rng(8)
    a.positions[a.free] += .001*(rng.normal(size=a.free.sum())+1j*rng.normal(size=a.free.sum()))
    report = diagnose(a, Settings(degree=6))
    assert report['interface_trace_error'] < 1e-13


def test_curved_boundary_and_no_boundary_motion():
    g = geometry(curved=True)
    settings = Settings(degree=8, iterations=8, search_critical_points=False)
    initial = Atlas(one_block_seed(g), settings.degree)
    result = generate(g, guide(), settings)
    assert len(result.atlas.ids) == 1
    assert result.report['quality']['accepted']
    assert np.array_equal(initial.positions[~initial.free], result.atlas.positions[~initial.free])
    assert max(b['boundary_error'] for b in result.report['quality']['blocks']) < settings.boundary_tolerance
    assert result.report['optimization']['final_objective'] < result.report['optimization']['initial_objective']


def test_complete_objective_gradient_including_spatial_guide():
    settings = Settings(degree=4, search_critical_points=False,
                        interface_weight=1.3,wall_orthogonality_weight=.7,interface_spacing_weight=.2)
    atlas = Atlas(quadrangulate(geometry()), settings.degree)
    objective = AtlasObjective(atlas, guide(), settings)
    x = objective.pack(atlas.positions)
    rng = np.random.default_rng(43)
    x += 1e-4*rng.normal(size=len(x))
    value, grad = objective(x)
    assert np.isfinite(value)
    for _ in range(4):
        direction = rng.normal(size=len(x))
        direction /= np.linalg.norm(direction)
        h = 2e-7
        fd = (objective(x+h*direction)[0]-objective(x-h*direction)[0])/(2*h)
        assert abs(fd-np.dot(grad, direction)) < 2e-7*max(1, abs(fd))


def test_zero_guide_is_finite_and_not_infinite_defects():
    zero = StokesGuide(lambda z: np.zeros_like(z), lambda z: np.zeros_like(z))
    result = generate(geometry(), zero, Settings(degree=4, iterations=12))
    assert result.report['critical_points'] == []
    assert result.report['optimization']['status'] == 'affine_aligned_preserved'
    assert min(b['minimum_jacobian'] for b in result.report['quality']['blocks']) > 0


def test_critical_point_search_does_not_force_a_topology_change():
    g = Geometry([0, 1, 1j], np.array([[0, 1, 2]]))
    root = .23+.31j
    f = StokesGuide(lambda z: np.asarray(z)-root, lambda z: np.ones_like(z))
    refined, records = insert_critical_points(g, f)
    assert len(refined.triangles) == 3
    assert len(records) == 1 and records[0]['inserted']
    assert abs(complex(*records[0]['point'])-root) < 1e-12
    result = generate(g, f, Settings(degree=4, iterations=4))
    assert len(result.atlas.ids) == 3  # triangle needs three charts, not nine
    assert np.isfinite(result.report['optimization']['final_objective'])
    assert not result.report['critical_points'][0]['inserted']


def test_stokes_field_changes_optimization():
    options = Settings(degree=8, iterations=8, search_critical_points=False)
    zero = StokesGuide(lambda z: 0., lambda z: 0.)
    a = generate(geometry(curved=True), zero, options)
    b = generate(geometry(curved=True), guide(), options)
    assert np.max(abs(a.atlas.positions-b.atlas.positions)) > 1e-4


def test_inverted_trial_is_rejected_before_field_evaluation():
    atlas = Atlas(quadrangulate(geometry()), 4)
    objective = AtlasObjective(atlas, guide(), Settings(degree=4))
    x = objective.pack(atlas.positions)
    x[0] += 10
    value, _ = objective(x)
    assert np.isinf(value)


def test_failure_is_explicit_and_save_is_portable(tmp_path):
    result = generate(geometry(curved=True), guide(), Settings(degree=4, iterations=0,
                      max_condition=1.01, search_critical_points=False, max_refinements=0))
    assert not result.report['quality']['accepted']
    with pytest.raises(RuntimeError, match='failing blocks'):
        result.require_accepted()
    path = tmp_path/'atlas.npz'
    result.atlas.save(path, result.report)
    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data['coefficients'], result.atlas.coefficients)
        assert 'sampled, not certified' in str(data['report'])


def test_bad_inputs():
    with pytest.raises(ValueError, match='Degenerate'):
        Geometry([0, 1, 2], np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match='Duplicate'):
        Geometry([0, 1, 1j], np.array([[0, 1, 2], [0, 2, 1]]))
    with pytest.raises(ValueError, match='integers'):
        Geometry([0, 1, 1j], np.array([[0., 1., 2.]]))
    with pytest.raises(ValueError, match='positive'):
        StokesGuide(lambda z: z, lambda z: 1, scale=0)
    with pytest.raises(ValueError, match='degree'):
        Settings(degree=2)


def test_affine_chart_acceptance_and_sampling_degree():
    g = geometry()
    p = g.vertices
    edges = {(i, (i+1)%4): g.curve(i, (i+1)%4) for i in range(4)}
    atlas = Atlas(QuadSeed(p, np.array([[0, 1, 2, 3]]), edges), 4)
    quality = diagnose(atlas, Settings(degree=4, solution_degree=12))
    assert quality['accepted']
    import json
    assert json.loads(json.dumps(quality))['accepted'] is True
    assert abs(quality['blocks'][0]['maximum_condition']-1) < 2e-7
    assert quality['blocks'][0]['operator_tail'] < 1e-12
    assert quality['solution_degree'] == 12


def test_hole_topology_is_preserved():
    # Topological fixture: four quads around a hole, each split into triangles.
    p = np.array([-2-2j, 2-2j, 2+2j, -2+2j, -.5-.5j, .5-.5j, .5+.5j, -.5+.5j])
    triangles = []
    for i in range(4):
        j = (i+1)%4
        triangles.extend([[i, j, 4+j], [i, 4+j, 4+i]])
    seed = quadrangulate(Geometry(p, np.array(triangles)))
    edges = set()
    for quad in seed.quads:
        edges.update(tuple(sorted(e)) for e in zip(quad, np.roll(quad, -1)))
    assert len(seed.vertices)-len(edges)+len(seed.quads) == 0  # annular Euler characteristic
    assert len(seed.boundary_curves) == 16
    atlas = Atlas(seed, 4)
    assert diagnose(atlas, Settings(degree=4))['interface_trace_error'] < 1e-12


def test_arnoldi_adapter_polynomial_rational_and_logarithmic_jets():
    from types import SimpleNamespace
    h = lambda k: np.r_[np.zeros(k+1), 1.]
    poly = SimpleNamespace(polynomial=True, scale=1., H=[h(k) for k in range(3)])
    rational = SimpleNamespace(polynomial=False, poles=np.array([3j, 3j]), H=[h(k) for k in range(2)])
    a, b, c, loga = .3+.2j, -.4+.7j, .2-.1j, .5+.8j
    solver = SimpleNamespace(groups=[poly, rational], f=np.array([1., 2., 0., a, b, c]),
                             log_coeff=np.array([loga.real, loga.imag, 7.]), laurent_center=-3j)
    field = from_arnoldi_solver(solver, chunk_size=3)
    z = np.array([.1+.2j, .7+.4j, .3+.1j, -.2-.5j]).reshape(2, 2)
    q, dq = field.sample(z)
    exactq = 4*(6*a*z+2*b/(z-3j)**3+6*c/(z-3j)**4-loga/(z+3j)**2)
    exactdq = 4*(6*a-6*b/(z-3j)**4-24*c/(z-3j)**5+2*loga/(z+3j)**3)
    assert np.max(abs(q-exactq)) < 1e-13
    assert np.max(abs(dq-exactdq)) < 1e-13
    z[0, 0] += .01
    changed, _ = field.sample(z)
    assert abs(changed[0, 0]-q[0, 0]) > 1e-3


def test_single_chart_independent_of_input_triangulation_and_numbering():
    # Refined boundary and interior discretization must not prescribe blocks.
    x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 5), indexing='ij')
    p = (x+1j*y).ravel()
    ids = np.arange(len(p)).reshape(x.shape)
    tris = []
    for i in range(3):
        for j in range(4):
            a,b,c,d = ids[i,j],ids[i+1,j],ids[i+1,j+1],ids[i,j+1]
            tris.extend([[a,b,c],[a,c,d]])
    rng = np.random.default_rng(93)
    permutation = rng.permutation(len(p))
    inverse = np.argsort(permutation)
    g = Geometry(p[permutation], inverse[np.asarray(tris)][:, ::-1])
    result = generate(g, guide(), Settings(degree=4, iterations=3))
    assert len(result.atlas.ids) == 1
    assert result.report['quality']['accepted']
    assert result.report['selected_layout'] == 'rectilinear_affine'


def test_guide_zero_inside_regular_single_chart_is_not_a_required_cut():
    z0 = .31+.22j
    g = StokesGuide(lambda z: np.asarray(z)-z0, lambda z: np.ones_like(z))
    result = generate(geometry(), g, Settings(degree=4, iterations=3))
    assert len(result.atlas.ids) == 1
    assert result.report['quality']['accepted']
    assert len(result.report['critical_points']) == 1
    assert result.report['critical_points'][0]['role'] == 'diagnostic_only'


def test_quality_filter_preserves_spectral_resolution_and_spacing():
    result = generate(geometry(curved=True), guide(), Settings(iterations=20))
    initial = result.report['initial_quality']['blocks'][0]
    final = result.report['quality']['blocks'][0]
    assert result.report['quality']['accepted']
    assert final['operator_tail'] <= 1e-5
    assert final['maximum_condition'] <= 1.05*initial['maximum_condition']
    assert final['scale_variation'] <= 1.05*initial['scale_variation']
    assert final['minimum_scale'] >= initial['minimum_scale']/1.05


def test_merging_and_refinement_preserve_boundary_and_conformity():
    seed = subdivide(one_block_seed(geometry(curved=True)))
    assert len(seed.quads) == 4
    settings = Settings(degree=8, iterations=0)
    def acceptable(candidate):
        return diagnose(Atlas(candidate, 8), settings)['accepted']
    merged, history = coarsen(seed, acceptable)
    assert len(merged.quads) == 1
    assert history
    assert acceptable(merged)
    for axis in (0, 1):
        split = bisect_single(merged, axis)
        quality = diagnose(Atlas(split, 8), settings)
        assert quality['accepted']
        assert quality['interface_trace_error'] < 1e-12


def test_triangle_pairing_and_merging_cannot_remove_a_hole():
    p = np.array([-2-2j, 2-2j, 2+2j, -2+2j, -.5-.5j, .5-.5j, .5+.5j, -.5+.5j])
    triangles = []
    for i in range(4):
        j = (i+1)%4
        triangles.extend([[i,j,4+j], [i,4+j,4+i]])
    g = Geometry(p, np.asarray(triangles))
    assert one_block_seed(g) is None
    paired = paired_seed(g)
    assert len(paired.quads) == 4
    merged, _ = coarsen(paired, lambda seed: True)
    assert len(merged.quads) == 4
    result = generate(g, StokesGuide(lambda z: 0., lambda z: 0.), Settings(degree=4, iterations=0))
    assert len(result.atlas.ids) == 8
    assert result.report['quality']['accepted']
    coarse = generate(g, StokesGuide(lambda z: 0., lambda z: 0.),
                      Settings(degree=4,iterations=0,prefer_affine_rectilinear=False))
    assert len(coarse.atlas.ids) == 4  # explicit old block-count-first policy


def test_hanging_interfaces_are_not_introduced_by_merging():
    p = np.array([0,1,2,1j,1+1j,2+1j,2j,1+2j])
    faces = np.array([[0,1,4,3], [1,2,5,4], [3,4,7,6]])
    edges = {}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            edges.setdefault(tuple(sorted((a,b))), []).append((a,b))
    from .topology import line
    boundary = {use[0]:line(p[use[0][0]],p[use[0][1]]) for use in edges.values() if len(use)==1}
    merged, history = coarsen(QuadSeed(p, faces, boundary), lambda seed: True)
    assert len(merged.quads) == 3
    assert not history


def test_split_only_after_the_one_block_candidate_fails():
    g = Geometry([0,4,4+1j,1j], np.array([[0,1,2],[0,2,3]]))
    zero = StokesGuide(lambda z: 0., lambda z: 0.)
    result = generate(g, zero, Settings(degree=4, iterations=0, max_condition=2.5))
    assert len(result.atlas.ids) == 2
    assert result.report['quality']['accepted']
    attempts = result.report['topology_attempts']
    assert any(a['layout']=='boundary_one_block' and not a['accepted'] for a in attempts)
    assert not attempts[0]['accepted']
    assert 'bisected' in result.report['selected_layout']


def test_operator_resolution_does_not_depend_on_length_units():
    g = geometry()
    atlas = Atlas(one_block_seed(g), 6)
    rng = np.random.default_rng(7)
    atlas.positions[atlas.free] += .002*(rng.normal(size=atlas.free.sum())+1j*rng.normal(size=atlas.free.sum()))
    settings = Settings(degree=6, solution_degree=12)
    first = diagnose(atlas, settings)['blocks'][0]
    other = Atlas(one_block_seed(Geometry(1000*g.vertices, g.triangles)), 6)
    other.positions = 1000*atlas.positions
    second = diagnose(other, settings)['blocks'][0]
    assert first['operator_tail'] > 1e-8
    assert abs(second['operator_tail']/first['operator_tail']-1) < 1e-7


def test_curved_hole_region_cover_does_not_identify_distinct_arcs():
    from scipy.spatial import Delaunay
    # Boundary-only unstructured triangulation; no radial cuts or quad layout.
    corners = np.array([-3-1j,3-1j,3+1j,-3+1j])
    outer = []
    for a,b in zip(corners,np.roll(corners,-1)):
        count = int(np.ceil(abs(b-a)/.8))
        outer.extend(a+(b-a)*np.arange(count)/count)
    center,radius = .12j,.35
    theta = 2*np.pi*np.arange(12)/12
    points = np.r_[outer,center+radius*np.exp(1j*theta)]
    triangles = Delaunay(np.column_stack([points.real,points.imag])).simplices
    triangles = triangles[abs(np.mean(points[triangles],axis=1)-center)>radius]
    curves = {(len(outer)+i,len(outer)+(i+1)%12):
              (lambda t,angle=angle:center+radius*np.exp(1j*(angle+np.pi*np.asarray(t)/6)))
              for i,angle in enumerate(theta)}
    geometry = Geometry(points,triangles,curves)
    settings = Settings(degree=10,iterations=0)
    seed, info = region_seed(geometry,lambda candidate:diagnose(Atlas(candidate,10),settings)['accepted'])
    assert seed is not None and len(seed.quads)<len(triangles)//2
    assert info['admissible_regions']>len(seed.quads)
    atlas = Atlas(seed,10)
    assert diagnose(atlas,settings)['accepted']
    z,_,_ = atlas.evaluate(61)
    assert np.min(abs(z-center)-radius)>-1e-8
    keys = [tuple(sorted(e)) for e in seed.boundary_curves]
    assert len(set(keys))==len(keys)  # two semicircles must remain distinct edges
    edges = {tuple(sorted((a,b))) for face in seed.quads for a,b in zip(face,np.roll(face,-1))}
    assert len(np.unique(seed.quads))-len(edges)+len(seed.quads)==0  # hole preserved
    for vertex in np.unique(seed.quads):
        if vertex >= len(outer):
            assert np.count_nonzero(seed.quads==vertex)==2


def test_operator_barrier_gradient_through_shared_coordinates():
    settings = Settings(degree=4,solution_degree=7,spectral_tolerance=1e-4,
                        operator_barrier_weight=.3,search_critical_points=False)
    atlas = Atlas(quadrangulate(geometry()),settings.degree)
    objective = AtlasObjective(atlas,guide(),settings)
    barrier_value,barrier_gradient = objective.operator_barrier(atlas.coefficients)
    assert barrier_value>1e-6 and np.max(abs(barrier_gradient))>1e-2
    x = objective.pack(atlas.positions)
    rng = np.random.default_rng(29)
    x += 1e-6*rng.normal(size=len(x))
    value,grad = objective(x)
    assert np.isfinite(value)
    for _ in range(4):
        direction = rng.normal(size=len(x));direction /= np.linalg.norm(direction)
        h = 1e-7
        fd = (objective(x+h*direction)[0]-objective(x-h*direction)[0])/(2*h)
        assert abs(fd-np.dot(grad,direction)) < 2e-6*max(1,abs(fd))


@pytest.mark.parametrize('angle',[0.,.37])
def test_aligned_eight_block_partition_is_frame_invariant_and_preserved(angle):
    from scipy.spatial import Delaunay
    p = np.array([-3-1j,3-1j,3+1j,-3+1j,-.35-.23j,.35-.23j,.35+.47j,-.35+.47j])
    tri = Delaunay(np.column_stack([p.real,p.imag])).simplices
    mid = np.mean(p[tri],axis=1)-.12j
    tri = tri[(abs(mid.real)>.35)|(abs(mid.imag)>.35)]
    rotation = np.exp(1j*angle)
    g = Geometry(2+3j+rotation*p,tri)
    result = generate(g,guide(),Settings(degree=6,iterations=20,search_critical_points=False))
    assert len(result.atlas.ids)==8
    assert result.report['selected_layout']=='rectilinear_affine'
    assert result.report['quality']['accepted']
    assert result.report['optimization']['iterations']==0
    assert not result.report['optimization']['guide_used_for_map_optimization']
    for b in result.report['quality']['blocks']:
        assert b['maximum_skew_cosine']<1e-11
        assert b['relative_metric_variation']<1e-11
    chart_vertices = result.atlas.positions[np.unique(result.atlas.seed.quads)]
    for corner in g.vertices[4:]:
        assert min(abs(chart_vertices-corner))<1e-12


def test_rectilinear_rule_handles_multiple_holes_without_an_encoded_count():
    from scipy.spatial import Delaunay
    p = np.r_[[-3-2j,3-2j,3+2j,-3+2j],
              -1+.3*np.array([-1-1j,1-1j,1+1j,-1+1j]),
              1+.3*np.array([-1-1j,1-1j,1+1j,-1+1j])]
    tri = Delaunay(np.column_stack([p.real,p.imag])).simplices
    mid = np.mean(p[tri],axis=1)
    keep = np.ones(len(tri),dtype=bool)
    for center in [-1,1]:
        keep &= (abs(mid.real-center)>.3)|(abs(mid.imag)>.3)
    result = generate(Geometry(p,tri[keep]),guide(),Settings(degree=4,iterations=0))
    assert len(result.atlas.ids)==13
    assert result.report['quality']['accepted']


def test_curved_boundary_is_not_mistaken_for_rectilinear_geometry():
    assert rectilinear_seed(geometry(curved=True)) is None


def test_interface_direction_is_independent_of_normal_chart_scale():
    from .smoothness import MapSmoothness
    from .topology import line
    vertices=np.array([0,1,3,1j,1+1j,3+1j])
    faces=np.array([[0,1,4,3],[1,2,5,4]])
    boundary={(u,v):line(vertices[u],vertices[v]) for u,v in [(0,1),(1,2),(2,5),(5,4),(4,3),(3,0)]}
    atlas=Atlas(QuadSeed(vertices,faces,boundary),4)
    z,a,b=atlas.evaluate(21)
    smooth=MapSmoothness(atlas,a,b)
    report=smooth.diagnose(a,b)
    assert report['interface_direction_rms_deg']<1e-10
    assert report['boundary_normal_deviation_rms_deg']<1e-10
    assert abs(report['interface_log_spacing_rms']-np.log(2))<1e-12
    assert smooth.energy(a,b,1.,1.,0.)[0]<1e-22


def test_genuine_interface_kink_is_detected_despite_shared_trace():
    from .smoothness import MapSmoothness
    from .topology import line
    vertices=np.array([0,1,2+.4j,1j,1+1j,2+1.4j])
    faces=np.array([[0,1,4,3],[1,2,5,4]])
    boundary={(u,v):line(vertices[u],vertices[v]) for u,v in [(0,1),(1,2),(2,5),(5,4),(4,3),(3,0)]}
    atlas=Atlas(QuadSeed(vertices,faces,boundary),4)
    z,a,b=atlas.evaluate(21)
    report=MapSmoothness(atlas,a,b).diagnose(a,b)
    assert abs(report['interface_direction_rms_deg']-np.degrees(np.arctan(.4)))<1e-10
    assert diagnose(atlas,Settings(degree=4))['interface_trace_error']<1e-12
