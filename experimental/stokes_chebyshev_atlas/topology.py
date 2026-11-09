"""Boundary reconstruction, triangle pairing, conforming merging/refinement.

Input triangulation edges do not prescribe block interfaces. Three-quads-per-
triangle subdivision remains only a fallback. CAD curves stay shared objects.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import root
from .spectral import nodes


def line(a, b):
    return lambda t: a + (b - a) * np.asarray(t)


@dataclass
class Geometry:
    """Nonoverlapping conforming triangles; arbitrary components and holes.

    vertices: complex physical coordinates. triangles: integer (T,3) indices.
    curves[(a,b)](t): optional analytic physical boundary edge, t in [0,1].
    Curves must meet their endpoint vertices; unspecified boundary edges are
    straight. The caller supplies a valid planar triangulation of the domain,
    not a quad decomposition. Curved edges must preserve its topology.
    """
    vertices: np.ndarray
    triangles: np.ndarray
    curves: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=complex).copy()
        raw = np.asarray(self.triangles)
        if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
            raise ValueError("triangles must be a nonempty (T,3) array")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Triangle indices must be integers")
        self.triangles = raw.astype(int, copy=True)
        if (self.vertices.ndim != 1 or not np.all(np.isfinite(self.vertices))
                or raw.min() < 0 or raw.max() >= len(self.vertices)):
            raise ValueError("Invalid vertices or triangle indices")
        if len(set(map(tuple, np.sort(raw, axis=1)))) != len(raw):
            raise ValueError("Duplicate triangles")
        z = self.vertices[self.triangles]
        area = np.imag(np.conj(z[:, 1] - z[:, 0]) * (z[:, 2] - z[:, 0]))
        scale = np.ptp(self.vertices.real) ** 2 + np.ptp(self.vertices.imag) ** 2
        if scale == 0 or np.any(abs(area) < 1e-14 * scale):
            raise ValueError("Degenerate triangle")
        flip = area < 0
        self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]
        uses = {}
        for tri in self.triangles:
            for a, b in zip(tri, np.roll(tri, -1)):
                uses.setdefault(tuple(sorted((a, b))), []).append((a, b))
        for edge, directions in uses.items():
            if len(directions) > 2 or (len(directions) == 2 and directions[0] == directions[1]):
                raise ValueError(f"Nonmanifold or overlapping edge {edge}")
        self.boundary = {e for e, use in uses.items() if len(use) == 1}
        self.edges = uses
        for (a, b), curve in self.curves.items():
            if tuple(sorted((a, b))) not in self.boundary:
                raise ValueError("A CAD curve can only be assigned to a boundary edge")
            if (b, a) in self.curves:
                raise ValueError("A boundary edge has two curve definitions")
            ends = np.asarray(curve(np.array([0., 1.])))
            if ends.shape != (2,) or np.max(abs(ends - self.vertices[[a, b]])) > 1e-10 * np.sqrt(scale):
                raise ValueError("Boundary curve does not match its vertices")

    def curve(self, a, b):
        if (a, b) in self.curves:
            return self.curves[a, b]
        if (b, a) in self.curves:
            f = self.curves[b, a]
            return lambda t: f(1 - np.asarray(t))
        return line(self.vertices[a], self.vertices[b])


def insert_critical_points(geometry, guide, relative_tolerance=1e-8, *, insert=True):
    """Newton-search Phi'=0 from triangle centroids, then split containing faces.

    With insert=False, return diagnostics without modifying the triangulation.
    The generator uses this mode: a guide zero alone does not require a cut.
    This is a search, not a certified enumeration. Uniform/constant Phi is
    detected as a degenerate guide and is not interpreted as infinitely many
    isolated defects. Roots on existing edges are reported but not inserted.
    """
    vertices = list(geometry.vertices)
    triangles = []
    records = []
    scale = max(np.ptp(geometry.vertices.real), np.ptp(geometry.vertices.imag))
    for tri in geometry.triangles:
        p = geometry.vertices[tri]
        seed = np.mean(p)
        if abs(guide.dq(np.array(seed))) * scale < relative_tolerance * guide.scale:
            triangles.append(tri.tolist())
            continue
        def fun(x):
            q = complex(guide.q(x[0] + 1j * x[1]))
            return [q.real, q.imag]
        def jac(x):
            q = complex(guide.dq(x[0] + 1j * x[1]))
            return [[q.real, -q.imag], [q.imag, q.real]]
        sol = root(fun, [seed.real, seed.imag], jac=jac)
        z = complex(*sol.x)
        mat = np.array([(p[1:] - p[0]).real, (p[1:] - p[0]).imag])
        uv = np.linalg.solve(mat, [z.real - p[0].real, z.imag - p[0].imag])
        bary = np.r_[1 - uv.sum(), uv]
        isroot = (np.isfinite(z) and abs(guide.q(z)) < relative_tolerance * guide.scale
                  and np.min(bary) >= -relative_tolerance)
        if isroot:
            interior = bool(np.min(bary) > relative_tolerance)
            if all(abs(z - complex(*r['point'])) > relative_tolerance * scale for r in records):
                records.append(dict(point=[z.real, z.imag], inserted=bool(interior and insert)))
            if interior and insert:
                j = len(vertices)
                vertices.append(z)
                triangles.extend([[int(a), int(b), j] for a, b in zip(tri, np.roll(tri, -1))])
                continue
        triangles.append(tri.tolist())
    return Geometry(vertices, np.array(triangles), geometry.curves), records


@dataclass
class QuadSeed:
    vertices: np.ndarray
    quads: np.ndarray
    boundary_curves: dict
    interior_curves: dict = field(default_factory=dict)


def boundary_cycles(faces):
    """Oriented boundary cycles of a union of consistently oriented faces."""
    edges = set()
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            if (b, a) in edges:
                edges.remove((b, a))
            else:
                edges.add((a, b))
    outgoing = {}
    for a, b in edges:
        if a in outgoing:
            return []  # vertex-only connection / nonmanifold union
        outgoing[a] = b
    cycles = []
    while outgoing:
        start = min(outgoing)
        cycle, a = [], start
        while a in outgoing:
            cycle.append(a)
            a = outgoing.pop(a)
        if a != start:
            return []
        cycles.append(cycle)
    return cycles


def seed_curve(seed, a, b):
    for table in (seed.boundary_curves, seed.interior_curves):
        if (a, b) in table:
            return table[a, b]
        if (b, a) in table:
            f = table[b, a]
            return lambda t, f=f: f(1-np.asarray(t))
    return line(seed.vertices[a], seed.vertices[b])


def _tangent(curve, end):
    # Fourth-order endpoint derivative; scale-free corner classification.
    h = 1e-3
    t = h*np.arange(5)
    if end:
        t = 1-t
    v = np.asarray(curve(t))
    d = (-25*v[0]+48*v[1]-36*v[2]+16*v[3]-3*v[4])/(12*h)
    return -d if end else d


def _join(curves):
    if len(curves) == 1:
        return curves[0]
    samples = [np.asarray(f(np.linspace(0, 1, 65))) for f in curves]
    lengths = np.array([np.sum(abs(np.diff(v))) for v in samples])
    if np.min(lengths) <= 0:
        raise ValueError('Degenerate boundary segment')
    a, b = samples[0][0], samples[-1][-1]
    # Exactly straight chains get a single affine parametrization. In particular,
    # a boundary subdivision does not change the recovered one-block map.
    points = np.concatenate(samples)
    if abs(b-a) > 0:
        r = (points-a)/(b-a)
        if np.max(abs(r.imag)) < 1e-11 and np.min(np.diff(r.real)) >= -1e-11:
            return line(a, b)
    breaks = np.r_[0., np.cumsum(lengths)/sum(lengths)]
    def joined(t):
        t = np.asarray(t)
        index = np.clip(np.searchsorted(breaks, t, side='right')-1, 0, len(curves)-1)
        out = np.empty(t.shape, dtype=complex)
        for i, f in enumerate(curves):
            mask = index == i
            out[mask] = f((t[mask]-breaks[i])/(breaks[i+1]-breaks[i]))
        return out
    return joined


def four_sided_region(vertices, faces, curve, protected=()):
    """Recognize a regular four-sided region, preserving required junctions.

    Only tangent-continuous joins may be removed. No corner is invented on a
    smooth closed wall; a disk does not become a nonsingular square chart.
    """
    loops = boundary_cycles(faces)
    if len(loops) != 1:
        return None  # holes / disconnected regions cannot be one square chart
    loop = loops[0]
    edges = [curve(a, b) for a, b in zip(loop, np.roll(loop, -1))]
    corners = []
    for i, vertex in enumerate(loop):
        left, right = _tangent(edges[i-1], True), _tangent(edges[i], False)
        if abs(left) == 0 or abs(right) == 0:
            return None
        if vertex in protected or abs(np.angle(right/left)) > 1e-6:
            corners.append(i)
    if len(corners) != 4:
        return None
    # The signed boundary area must be positive (not the hole's exterior).
    p = np.concatenate([f(np.linspace(0, 1, 17)[:-1]) for f in edges])
    if np.sum(np.imag(np.conj(p)*np.roll(p, -1))) <= 0:
        return None
    quad = [loop[i] for i in corners]
    sides = {}
    for j, i in enumerate(corners):
        end = corners[(j+1)%4]
        count = (end-i) % len(loop)
        sides[quad[j], quad[(j+1)%4]] = _join([edges[(i+k)%len(loop)] for k in range(count)])
    return quad, sides


def one_block_seed(geometry):
    """Recover a four-sided domain boundary, independently of its triangles."""
    region = four_sided_region(geometry.vertices, geometry.triangles, geometry.curve)
    if region is None:
        return None
    quad, curves = region
    return QuadSeed(geometry.vertices.copy(), np.array([quad]), curves)


def rectilinear_seed(geometry, max_blocks=256):
    """Recover the boundary's orthogonal frame and its aligned partition.

    Coordinates come only from true boundary corners, not sampling vertices.
    An even/odd boundary test removes solid cells. The same construction handles
    rotated/translated orthogonal polygons, holes, and disconnected components;
    neither a square obstacle nor a block count is encoded.
    """
    loops = boundary_cycles(geometry.triangles)
    if not loops:
        return None
    a,b = max(geometry.boundary, key=lambda e:abs(geometry.vertices[e[1]]-geometry.vertices[e[0]]))
    origin = geometry.vertices[a]
    direction = geometry.vertices[b]-origin
    direction /= abs(direction)
    points = (geometry.vertices-origin)/direction
    scale = np.hypot(np.ptp(points.real),np.ptp(points.imag))
    tolerance = 1e-11*scale
    corners = []
    for loop in loops:
        p = points[loop]
        edges = np.roll(p,-1)-p
        if np.any(np.minimum(abs(edges.real),abs(edges.imag)) > tolerance):
            return None
        for i,(u,v) in enumerate(zip(loop,np.roll(loop,-1))):
            sample = (geometry.curve(u,v)(np.linspace(0,1,33))-origin)/direction
            ratio = (sample-p[i])/edges[i]
            if np.max(abs(ratio.imag))*abs(edges[i]) > tolerance or np.min(np.diff(ratio.real)) < -1e-12:
                return None  # a curved edge is not rectilinear
            previous = edges[i-1]
            if abs(np.imag(np.conj(previous)*edges[i])) > 1e-10*abs(previous)*abs(edges[i]):
                corners.append(p[i])
    if len(corners)<4:
        return None
    def levels(values):
        values = np.sort(values)
        return values[np.r_[True,np.diff(values)>64*np.finfo(float).eps*scale]]
    corners = np.asarray(corners)
    xs,ys = levels(corners.real),levels(corners.imag)
    if (len(xs)-1)*(len(ys)-1) > max(4096,4*max_blocks):
        return None
    x,y = np.meshgrid(xs,ys,indexing='ij')
    coordinates = x+1j*y
    ids = np.arange(coordinates.size).reshape(coordinates.shape)
    cx,cy = np.meshgrid((xs[:-1]+xs[1:])/2,(ys[:-1]+ys[1:])/2,indexing='ij')
    inside = np.zeros(cx.shape,dtype=bool)
    for loop in loops:
        p = points[loop]
        for u,v in zip(p,np.roll(p,-1)):
            if abs(v.imag-u.imag) <= tolerance:
                continue
            crosses = (u.imag>cy)!=(v.imag>cy)
            crossing_x = u.real+(v.real-u.real)*(cy-u.imag)/(v.imag-u.imag)
            inside ^= crosses & (cx<crossing_x)
    if not np.any(inside) or np.sum(inside)>max_blocks:
        return None
    faces = np.asarray([[ids[i,j],ids[i+1,j],ids[i+1,j+1],ids[i,j+1]]
                        for i,j in zip(*np.nonzero(inside))])
    vertices = origin+direction*coordinates.ravel()
    uses = {}
    for face in faces:
        for u,v in zip(face,np.roll(face,-1)):
            uses.setdefault(tuple(sorted((u,v))),[]).append((u,v))
    boundary = {edges[0]:line(vertices[edges[0][0]],vertices[edges[0][1]])
                for edges in uses.values() if len(edges)==1}
    return QuadSeed(vertices,faces,boundary)


def paired_seed(geometry):
    """Maximum-cardinality, shape-weighted triangle pairing; require full cover.

    Binary matching prevents a greedy early pair from stranding triangles.
    Unmatched triangulations use the conforming subdivision fallback instead.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    owners = {}
    for i, tri in enumerate(geometry.triangles):
        for a, b in zip(tri, np.roll(tri, -1)):
            owners.setdefault(tuple(sorted((a, b))), []).append(i)
    pairs = []
    for use in owners.values():
        if len(use) != 2:
            continue
        loops = boundary_cycles(geometry.triangles[use])
        if len(loops) != 1 or len(loops[0]) != 4:
            continue
        quad = loops[0]
        z = geometry.vertices[quad]
        edges = np.roll(z, -1)-z
        cross = np.imag(np.conj(edges)*np.roll(edges, -1))
        if np.min(cross) <= 1e-12 * np.max(abs(edges))**2:
            continue
        # Cheap geometric ranking; acceptance is checked on actual curved maps.
        score = float(np.max(abs(edges))/np.min(abs(edges))
                      + np.max(abs(np.real(np.conj(edges)*np.roll(edges, -1)))/cross))
        pairs.append((use, quad, score))
    if not pairs:
        return None
    incidence = lil_matrix((len(geometry.triangles), len(pairs)))
    for j, (use, _, _) in enumerate(pairs):
        incidence[use, j] = 1
    # Every feasible solution covers all triangles; minimize shape cost among them.
    result = milp(np.array([p[2] for p in pairs]), integrality=np.ones(len(pairs)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(incidence.tocsr(), 1, 1),
                  options={'time_limit': 10.})
    if result.x is None or np.max(abs(incidence@np.rint(result.x)-1)) > 1e-8:
        return None
    quads = [p[1] for p, selected in zip(pairs, result.x) if selected > .5]
    curves = {(a, b): geometry.curve(a, b) for a, b in geometry.boundary}
    return QuadSeed(geometry.vertices.copy(), np.asarray(quads), curves)


def region_seed(geometry, acceptable, max_states=4000, max_region_triangles=16):
    """Cover triangles by larger admissible four-sided regions automatically.

    Smooth CAD boundary vertices can disappear only when all their incident
    triangles belong to the region. This is necessary for curved holes, where
    pairwise triangle matching alone often cannot produce regular quad charts.
    The connected-region enumeration and binary exact-cover solve are bounded;
    this is a topology heuristic, not a minimum-block proof.
    """
    from collections import deque
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix, hstack, csr_matrix
    triangles = geometry.triangles
    vertex_faces, edge_faces = {}, {}
    for i, tri in enumerate(triangles):
        for v in tri:
            vertex_faces.setdefault(v, set()).add(i)
        for a,b in zip(tri,np.roll(tri,-1)):
            edge_faces.setdefault(tuple(sorted((a,b))), []).append(i)
    neighbors = [set() for _ in triangles]
    for use in edge_faces.values():
        if len(use)==2:
            a,b=use
            neighbors[a].add(b); neighbors[b].add(a)
    queue = deque(frozenset([i]) for i in range(len(triangles)))
    seen = set(queue)
    candidates = []
    explored = 0
    while queue and explored < max_states:
        group = queue.popleft()
        explored += 1
        if len(group) >= 2:
            protected = {v for v,use in vertex_faces.items() if use-group}
            region = four_sided_region(geometry.vertices, triangles[sorted(group)], geometry.curve, protected)
            if region is not None:
                quad, sides = region
                if acceptable(QuadSeed(geometry.vertices, np.asarray([quad]), sides)):
                    candidates.append((group,quad,sides))
        if len(group) < max_region_triangles:
            frontier = set().union(*(neighbors[i] for i in group))-group
            for j in sorted(frontier):
                extended = group | {j}
                if extended not in seen:
                    seen.add(extended); queue.append(extended)
    info = dict(explored_regions=explored, admissible_regions=len(candidates),
                enumeration_truncated=bool(queue), max_region_triangles=max_region_triangles)
    if not candidates:
        return None, info
    incidence = lil_matrix((len(triangles),len(candidates)))
    costs = []
    physical_keys = {}
    for j,(group,quad,_) in enumerate(candidates):
        incidence[list(group),j] = 1
        z = geometry.vertices[quad]
        aspect = np.max(abs(np.roll(z,-1)-z))/np.min(abs(np.roll(z,-1)-z))
        costs.append(1+1e-6*aspect/(1+aspect))
        loop = boundary_cycles(triangles[sorted(group)])[0]
        for a,b in zip(quad,np.roll(quad,-1)):
            first,last = loop.index(a),loop.index(b)
            chain = [tuple(sorted((loop[k%len(loop)],loop[(k+1)%len(loop)])))
                     for k in range(first,first+(last-first)%len(loop))]
            if all(e in geometry.boundary for e in chain):
                physical_keys.setdefault(tuple(sorted((a,b))), []).append(j)
    # The atlas identifies edges by endpoint pair. Distinct arcs of the same
    # closed wall must not collapse onto that same key (e.g. two semicircles).
    unique_boundary = lil_matrix((len(physical_keys),len(candidates)))
    for row, uses in enumerate(physical_keys.values()):
        unique_boundary[row, uses] = 1
    # Do not place extraordinary three-chart junctions on a smooth curved wall.
    # A retained smooth wall vertex has two incident charts; a removed sample
    # vertex has none. Defects may still be necessary elsewhere in the atlas.
    regular = []
    for loop in boundary_cycles(triangles):
        for i,v in enumerate(loop):
            left=geometry.curve(loop[i-1],v)
            right=geometry.curve(v,loop[(i+1)%len(loop)])
            tl,tr=_tangent(left,True),_tangent(right,False)
            curved = any(abs(f(.5)-.5*(f(0.)+f(1.))) > 1e-9*abs(f(1.)-f(0.)) for f in (left,right))
            if curved and abs(np.angle(tr/tl))<1e-6:
                regular.append(v)
    nc,nv = len(candidates),len(regular)
    constraint = lil_matrix((nv,nc+nv))
    for row,v in enumerate(regular):
        for j,(_,quad,_) in enumerate(candidates):
            if v in quad: constraint[row,j]=1
        constraint[row,nc+row]=-2
    incidence_full=hstack([incidence.tocsr(),csr_matrix((len(triangles),nv))]).tocsr()
    boundary_full=hstack([unique_boundary.tocsr(),csr_matrix((len(physical_keys),nv))]).tocsr()
    constraints=[LinearConstraint(incidence_full,1,1),LinearConstraint(boundary_full,0,1)]
    if nv:constraints.append(LinearConstraint(constraint.tocsr(),0,0))
    info['regular_curved_boundary_samples']=nv
    solution = milp(np.r_[costs,np.zeros(nv)],integrality=np.ones(nc+nv),bounds=Bounds(0,1),
                    constraints=constraints,options={'time_limit':10.})
    if solution.x is None or np.max(abs(incidence@np.rint(solution.x[:nc])-1))>1e-8:
        return None, info
    quads, boundary, interior = [], {}, {}
    for (group,quad,sides), chosen in zip(candidates,solution.x[:nc]):
        if chosen < .5: continue
        quads.append(quad)
        loop = boundary_cycles(triangles[sorted(group)])[0]
        for (a,b), curve in sides.items():
            i,j = loop.index(a),loop.index(b)
            chain = [tuple(sorted((loop[k%len(loop)],loop[(k+1)%len(loop)])))
                     for k in range(i,i+(j-i)%len(loop))]
            target = boundary if all(e in geometry.boundary for e in chain) else interior
            target[a,b] = curve
    info['selected_regions'] = len(quads)
    return QuadSeed(geometry.vertices.copy(),np.asarray(quads),boundary,interior), info


def coarsen(seed, acceptable):
    """Greedy conforming patch merges, gated by actual map quality.

    Try face pairs and vertex stars. A removed boundary vertex may not belong
    to any outside face, so merges cannot introduce hanging interfaces.
    """
    history = []
    while True:
        faces = seed.quads
        vertex_faces, edge_faces = {}, {}
        for i, face in enumerate(faces):
            for v in face:
                vertex_faces.setdefault(v, set()).add(i)
            for a, b in zip(face, np.roll(face, -1)):
                edge_faces.setdefault(tuple(sorted((a, b))), set()).add(i)
        groups = {tuple(sorted(v)) for v in edge_faces.values() if len(v) == 2}
        groups.update(tuple(sorted(v)) for v in vertex_faces.values() if len(v) > 2)
        merged = False
        for group in sorted(groups, key=lambda g: (-len(g), g)):
            inside = set(group)
            protected = {v for v, use in vertex_faces.items() if use-inside}
            region = four_sided_region(seed.vertices, faces[list(group)],
                                       lambda a, b: seed_curve(seed, a, b), protected)
            if region is None:
                continue
            quad, sides = region
            probe = QuadSeed(seed.vertices, np.array([quad]), sides)
            if not acceptable(probe):
                continue
            # New long edges are physical boundaries only when all their old
            # segments were physical. An unchanged shared edge remains free.
            boundary = dict(seed.boundary_curves)
            interior = dict(seed.interior_curves)
            old_loop = boundary_cycles(faces[list(group)])[0]
            outside_edges = {tuple(sorted((u,v))) for i,face in enumerate(faces) if i not in inside
                             for u,v in zip(face,np.roll(face,-1))}
            collision = False
            for (a, b), curve in sides.items():
                i, j = old_loop.index(a), old_loop.index(b)
                chain = [(old_loop[k % len(old_loop)], old_loop[(k+1) % len(old_loop)])
                         for k in range(i, i+(j-i) % len(old_loop))]
                physical = all((u, v) in boundary or (v, u) in boundary for u, v in chain)
                if physical and tuple(sorted((a,b))) in outside_edges:
                    collision = True
                    break
                (boundary if physical else interior)[a, b] = curve
            if collision:
                continue
            remaining = [face.tolist() for i, face in enumerate(faces) if i not in inside]
            seed = QuadSeed(seed.vertices, np.asarray(remaining+[quad]), boundary, interior)
            history.append(dict(before=len(faces), after=len(seed.quads), merged_faces=list(group)))
            merged = True
            break
        if not merged:
            return seed, history


def subdivide(seed):
    """Conforming four-way refinement, preserving exact physical edge curves."""
    vertices = list(seed.vertices)
    midpoints, boundary, interior = {}, {}, {}
    for face in seed.quads:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((a, b)))
            if key in midpoints:
                continue
            u, v = key
            curve = seed_curve(seed, u, v)
            mid = len(vertices)
            vertices.append(complex(curve(.5)))
            midpoints[key] = mid
            target = boundary if (u, v) in seed.boundary_curves or (v, u) in seed.boundary_curves else interior
            target[u, mid] = lambda t, f=curve: f(np.asarray(t)/2)
            target[mid, v] = lambda t, f=curve: f((1+np.asarray(t))/2)
    quads = []
    for a, b, c, d in seed.quads:
        bottom, right, top, left = [seed_curve(seed, u, v) for u, v in ((a,b), (b,c), (d,c), (a,d))]
        p = seed.vertices[[a,b,c,d]]
        def mapping(s, t, bottom=bottom, right=right, top=top, left=left, p=p):
            s, t = np.asarray(s), np.asarray(t)
            return ((1-t)*bottom(s)+t*top(s)+(1-s)*left(t)+s*right(t)
                    - ((1-s)*(1-t)*p[0]+s*(1-t)*p[1]+s*t*p[2]+(1-s)*t*p[3]))
        center = len(vertices)
        vertices.append(complex(mapping(.5,.5)))
        ab, bc, cd, da = [midpoints[tuple(sorted(e))] for e in ((a,b),(b,c),(c,d),(d,a))]
        interior[ab, center] = lambda t, m=mapping: m(.5, np.asarray(t)/2)
        interior[center, cd] = lambda t, m=mapping: m(.5, (1+np.asarray(t))/2)
        interior[da, center] = lambda t, m=mapping: m(np.asarray(t)/2, .5)
        interior[center, bc] = lambda t, m=mapping: m((1+np.asarray(t))/2, .5)
        quads.extend([[a,ab,center,da], [ab,b,bc,center], [center,bc,c,cd], [da,center,cd,d]])
    return QuadSeed(np.asarray(vertices), np.asarray(quads), boundary, interior)


def bisect_single(seed, axis):
    """Try either two-chart split before four-way refinement of one chart."""
    if len(seed.quads) != 1 or axis not in (0, 1):
        raise ValueError('Bisection requires one chart and axis 0 or 1')
    vertices = list(seed.vertices)
    a, b, c, d = seed.quads[0]
    bottom, right, top, left = [seed_curve(seed, u, v) for u, v in ((a,b),(b,c),(d,c),(a,d))]
    p = seed.vertices[[a,b,c,d]]
    def mapping(s, t):
        return ((1-t)*bottom(s)+t*top(s)+(1-s)*left(t)+s*right(t)
                - ((1-s)*(1-t)*p[0]+s*(1-t)*p[1]+s*t*p[2]+(1-s)*t*p[3]))
    edges = ((a,b,bottom), (d,c,top)) if axis == 0 else ((a,d,left), (b,c,right))
    boundary = dict(seed.boundary_curves)
    interior = dict(seed.interior_curves)
    mids = []
    for u, v, curve in edges:
        mid = len(vertices)
        vertices.append(complex(curve(.5)))
        mids.append(mid)
        target = boundary if (u,v) in boundary or (v,u) in boundary else interior
        target[u,mid] = lambda t, f=curve: f(np.asarray(t)/2)
        target[mid,v] = lambda t, f=curve: f((1+np.asarray(t))/2)
    e, f = mids
    if axis == 0:
        interior[e,f] = lambda t: mapping(.5, np.asarray(t))
        quads = [[a,e,f,d], [e,b,c,f]]
    else:
        interior[e,f] = lambda t: mapping(np.asarray(t), .5)
        quads = [[a,b,f,e], [e,f,c,d]]
    return QuadSeed(np.asarray(vertices), np.asarray(quads), boundary, interior)


def quadrangulate(geometry):
    """Three quads per triangle, shared midpoint per edge, curved CAD traces."""
    vertices = list(geometry.vertices)
    midpoints = {}
    boundary_curves = {}
    for a, b in geometry.edges:
        curve = geometry.curve(a, b)
        j = len(vertices)
        vertices.append(complex(curve(.5)))
        midpoints[a, b] = j
        if (a, b) in geometry.boundary:
            boundary_curves[a, j] = lambda t, f=curve: f(np.asarray(t) / 2)
            boundary_curves[j, b] = lambda t, f=curve: f((1 + np.asarray(t)) / 2)
    quads = []
    for tri in geometry.triangles:
        center = len(vertices)
        vertices.append(np.mean(geometry.vertices[tri]))
        for i in range(3):
            a, b, c = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
            quads.append([a, midpoints[tuple(sorted((a, b)))], center,
                          midpoints[tuple(sorted((c, a)))]])
    return QuadSeed(np.asarray(vertices), np.asarray(quads), boundary_curves)


def nodal_layout(seed, n):
    """Shared Lobatto degrees of freedom and polynomial Coons initialization.

    Axis 0 is xi, axis 1 eta; corners are (-,-),(+,-),(+,+),(-,+).
    The same global edge nodes give identical polynomial traces, including
    orientation reversals. Every physical boundary degree of freedom is fixed.
    """
    t = (nodes(n) + 1) / 2
    positions = list(seed.vertices)
    fixed = set()
    edge_ids = {}
    edge_curves = {}
    for quad in seed.quads:
        for a, b in zip(quad, np.roll(quad, -1)):
            key = tuple(sorted((a, b)))
            if key in edge_ids:
                continue
            a, b = key
            curve = seed.boundary_curves.get((a, b))
            if curve is None and (b, a) in seed.boundary_curves:
                f = seed.boundary_curves[b, a]
                curve = lambda t, f=f: f(1 - np.asarray(t))
            boundary = curve is not None
            curve = curve or seed_curve(seed, a, b)
            ids = np.r_[a, np.arange(len(positions), len(positions) + n - 2), b].astype(int)
            positions.extend(curve(t[1:-1]))
            edge_ids[key] = ids
            edge_curves[key] = curve
            if boundary:
                fixed.update(ids)
    def oriented(a, b):
        ids = edge_ids[tuple(sorted((a, b)))]
        return ids if a < b else ids[::-1]
    blocks = []
    for a, b, c, d in seed.quads:
        ids = np.empty((n, n), dtype=int)
        ids[:, 0], ids[-1, :], ids[:, -1], ids[0, :] = (
            oriented(a, b), oriented(b, c), oriented(d, c), oriented(a, d))
        p = np.asarray(positions)
        s, v = t[:, None], t[None, :]
        blend = ((1-s)*(1-v)*p[a] + s*(1-v)*p[b] + s*v*p[c] + (1-s)*v*p[d])
        z = ((1-v)*p[ids[:, 0]][:, None] + v*p[ids[:, -1]][:, None]
             + (1-s)*p[ids[0, :]][None, :] + s*p[ids[-1, :]][None, :] - blend)
        interior = np.arange(len(positions), len(positions) + (n - 2)**2).reshape(n - 2, n - 2)
        ids[1:-1, 1:-1] = interior
        positions.extend(z[1:-1, 1:-1].ravel())
        blocks.append(ids)
    positions = np.asarray(positions)
    free = np.ones(len(positions), dtype=bool)
    free[list(fixed)] = False
    # Input triangulations may contain unused vertices; never optimize them.
    used = np.unique(blocks)
    free[np.setdiff1d(np.arange(len(positions)), used)] = False
    return positions, np.asarray(blocks), free, edge_ids, edge_curves
