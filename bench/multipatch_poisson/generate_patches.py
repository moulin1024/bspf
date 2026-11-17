"""Reproducible all-quad cubic patch meshes of local TCV/ITER field domains.

Dependencies: numpy scipy h5py contourpy shapely gmsh matplotlib.
The flux classifier is reconstructed from Parallax, not linked to its Fortran
library. Geometry is an approximation to cubic interpolation of the NetCDF.
"""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
import contourpy
import gmsh
from scipy.interpolate import RectBivariateSpline
from scipy.special import comb
from shapely.geometry import Polygon, LineString, box, mapping, shape
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

HERE = Path(__file__).resolve().parent
CASES = dict(tcv=('run_genex/TCV-X21.nc', .74, .045),
             iter=('tools/iter_strong_scaling/inputs/ITER_q10.nc', .9, .035))


def scalar(a, key):
    return float(np.asarray(a[key]).ravel()[0])


class Equilibrium:
    def __init__(self, name, root):
        path, self.rmin, self.h = CASES[name]
        self.name, self.path = name, root/path
        with h5py.File(self.path) as f:
            g, p, lim = f['Magnetic_geometry'], f['Psi_limits'], f['Grid_limits']
            self.R, self.Z = g['R'][:], -g['Z'][:][::-1]
            psi = g['psi'][:][::-1]
            p0, px = scalar(p.attrs, 'psi_axis'), scalar(p.attrs, 'psi_seperatrix')
            self.spline = RectBivariateSpline(self.R, self.Z, ((psi-p0)/(px-p0)).T)
            self.rmax = scalar(p.attrs, 'rho_max')
            self.bounds = (scalar(lim.attrs,'grid_rmin'), -scalar(lim.attrs,'grid_zmax'),
                           scalar(lim.attrs,'grid_rmax'), -scalar(lim.attrs,'grid_zmin'))
            def polygon(k):
                assert scalar(f[k].attrs, 'invert') == 0
                return Polygon(np.c_[f[k]['R_points'][:], -f[k]['Z_points'][:]])
            self.wall, self.exclusion = polygon('divertor_polygon'), polygon('exclusion_polygon')
            if name == 'tcv':
                self.exclusion = box(.675, -.55, 1.26250217, .84)
            self.private = polygon('Psi_limits/private_flux')
            a = f['Psi_limits/private_flux'].attrs
            assert scalar(a,'use_local_min') == 1 and scalar(a,'use_local_max') == 0
            self.pmin = max(self.rmin, scalar(a,'local_rho_min'))
            self.scale = scalar(g.attrs, 'magnetic_axis_R')
            self.xpoint = [scalar(g.attrs,'x_point_R')/self.scale,
                           -scalar(g.attrs,'x_point_Z')/self.scale]

    def domain(self, n):
        x = np.linspace(self.bounds[0], self.bounds[2], n)
        y = np.linspace(self.bounds[1], self.bounds[3], round(n*1.5))
        cg = contourpy.contour_generator(x=x, y=y, z=self.spline(x,y).T, fill_type='OuterOffset')
        def band(lo, hi):
            points, offsets = cg.filled(lo*lo, hi*hi)
            polygons = []
            for a, cuts in zip(points, offsets):
                rings = [a[i:j] for i,j in zip(cuts[:-1],cuts[1:])]
                polygons.append(Polygon(rings[0], rings[1:]))
            return unary_union(polygons)
        outer = band(self.rmin, self.rmax).difference(self.private)
        inner = band(self.pmin, self.rmax).intersection(self.private)
        d = outer.union(inner).intersection(self.exclusion).intersection(self.wall).intersection(box(*self.bounds))
        assert d.is_valid and d.area > 0
        return d

    def classify_boundary(self, point):
        from shapely.geometry import Point
        pt = Point(point)
        rho = np.sqrt(max(0, self.spline.ev(*point)))
        candidates = [(abs(rho-self.rmin), 'core'), (abs(rho-self.rmax), 'outer_flux'),
                      (abs(rho-self.pmin), 'private_flux')]
        # Physical polygon boundaries take precedence if geometrically close.
        for geom, name in [(self.wall,'target'),(self.exclusion,'exclusion'),(box(*self.bounds),'grid_limit')]:
            if geom.boundary.distance(pt) < 3e-5:
                return name
        return min(candidates)[1]


def add_ring(ring, h):
    a = np.asarray(ring.coords)[:-1]
    prev, nxt = a-np.roll(a,1,axis=0), np.roll(a,-1,axis=0)-a
    angle = np.arccos(np.clip(np.sum(prev*nxt,axis=1)/np.linalg.norm(prev,axis=1)/np.linalg.norm(nxt,axis=1),-1,1))
    # Preserve small but genuine divertor polygon corners: smoothing a 10-degree
    # turn into a spline changes the domain even if the patch Jacobians are good.
    corners = list(np.flatnonzero(angle > np.deg2rad(5)))
    if len(corners) < 4:
        corners = sorted(set(corners+list(np.linspace(0,len(a),5,dtype=int)[:-1])))
    corner_tags = {i: gmsh.model.geo.addPoint(*a[i],0,h) for i in corners}
    curves = []
    for start, stop in zip(corners, corners[1:]+[corners[0]+len(a)]):
        ids = np.arange(start,stop+1)%len(a)
        line = LineString(a[ids])
        count = max(3, int(np.ceil(line.length/.003))+1)
        points = np.array([line.interpolate(s, normalized=True).coords[0] for s in np.linspace(0,1,count)])
        tags = [corner_tags[start]]+[gmsh.model.geo.addPoint(*p,0,h) for p in points[1:-1]]+[corner_tags[stop%len(a)]]
        curves.append(gmsh.model.geo.addSpline(tags))
    return gmsh.model.geo.addCurveLoop(curves)


def basis(q, derivative=0):
    x = np.linspace(0,1,4)
    values = []
    for j in range(4):
        p = np.poly1d([1.])
        for k in range(4):
            if k != j: p *= np.poly1d([1.,-x[k]])/(x[j]-x[k])
        values.append(np.polyder(p,derivative)(q))
    return np.array(values).T


def metrics(patches, q):
    B, D = basis(q), basis(q,1)
    du = np.einsum('ai,bj,pijc->pabc',D,B,patches,optimize=True)
    dv = np.einsum('ai,bj,pijc->pabc',B,D,patches,optimize=True)
    det = du[...,0]*dv[...,1]-du[...,1]*dv[...,0]
    scaled = det/(np.linalg.norm(du,axis=-1)*np.linalg.norm(dv,axis=-1))
    frob = np.sum(du*du+dv*dv,axis=-1)
    largest = .5*(frob+np.sqrt(np.maximum(0,frob*frob-4*det*det)))
    cond = largest/np.abs(det)
    return det, scaled, cond


def export(name, eq, domain, convergence, out, h, algorithm=11, curvature=0):
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads',1)
    gmsh.option.setNumber('General.Terminal',1)
    gmsh.model.add(name)
    polygons = [domain] if domain.geom_type == 'Polygon' else list(domain.geoms)
    surfaces = []
    for p in polygons:
        p = orient(p,sign=1)
        loops = [add_ring(r,h) for r in [p.exterior,*p.interiors]]
        surfaces.append(gmsh.model.geo.addPlaneSurface(loops))
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2,surfaces, name='plasma')
    groups={}
    for dim,curve in gmsh.model.getEntities(1):
        lo,hi=gmsh.model.getParametrizationBounds(1,curve)
        point=gmsh.model.getValue(1,curve,(lo+hi)/2)[:2]
        groups.setdefault(eq.classify_boundary(point),[]).append(curve)
    for kind,curves in groups.items():
        gmsh.model.addPhysicalGroup(1,curves,name=kind)
    gmsh.option.setNumber('Mesh.Algorithm',algorithm)
    gmsh.option.setNumber('Mesh.RecombinationAlgorithm',3)
    gmsh.option.setNumber('Mesh.RecombineAll',1)
    gmsh.option.setNumber('Mesh.MeshSizeMin',h/6)
    gmsh.option.setNumber('Mesh.MeshSizeMax',h)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',curvature)
    gmsh.option.setNumber('Mesh.Smoothing',20)
    gmsh.model.mesh.generate(2)
    types, tags, nodes = gmsh.model.mesh.getElements(2)
    if any(gmsh.model.mesh.getElementProperties(int(t))[0].startswith('Triangle') for t in types):
        raise RuntimeError('Quad mesher left triangles; do not export a mixed mesh')
    gmsh.model.mesh.setOrder(3)
    gmsh.model.mesh.optimize('HighOrder')
    types, tags, nodes = gmsh.model.mesh.getElements(2)
    assert len(types)==1
    label, dim, order, nn, local, primary = gmsh.model.mesh.getElementProperties(int(types[0]))
    assert primary == 4 and nn == 16 and order == 3
    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    order_nodes = np.argsort(node_tags)
    node_tags, xyz = node_tags[order_nodes], xyz.reshape(-1,3)[order_nodes,:2]
    connectivity = np.searchsorted(node_tags,nodes[0].reshape(-1,16))
    local = np.asarray(local).reshape(-1,2)
    ij = np.rint((local+1)*1.5).astype(int)
    grid_ids = np.empty((len(connectivity),4,4),dtype=np.int64)
    grid_ids[:,ij[:,0],ij[:,1]] = connectivity
    patches = xyz[grid_ids]
    det, scaled, cond = metrics(patches,np.linspace(0,1,25))
    # det(DF) for bicubic maps is at most bi-degree (5,5). Positive
    # Bernstein coefficients certify positivity throughout the whole patch.
    t = np.linspace(0,1,6)
    V = np.array([comb(5,j)*t**j*(1-t)**(5-j) for j in range(6)]).T
    inv = np.linalg.inv(V)
    coefficients = np.einsum('ia,pab,jb->pij', inv,metrics(patches,t)[0],inv)
    certified = coefficients.min(axis=(1,2)) > 1e-12*det.max(axis=(1,2))
    edges, interfaces, boundary = {}, [], []
    # Face order: u=0, u=1, v=0, v=1; each along increasing remaining coordinate.
    for p, ids in enumerate(grid_ids):
        for face, edge in enumerate([ids[0,:],ids[-1,:],ids[:,0],ids[:,-1]]):
            key = tuple(sorted((int(edge[0]),int(edge[-1]))))
            if key in edges:
                other, oface, oe = edges.pop(key)
                reverse = bool(edge[0] != oe[0])
                assert np.array_equal(edge,oe[::-1] if reverse else oe)
                interfaces.append((other,oface,p,face,int(reverse)))
            else: edges[key]=(p,face,edge)
    for p,face,edge in edges.values():
        midpoint=(basis(np.array([.5]))@xyz[edge])[0]
        boundary.append(dict(patch=p,face=face,kind=eq.classify_boundary(midpoint),bc='dirichlet_zero'))
    np.savez_compressed(out/'patches.npz',geometry=patches,geometry_nodes=np.linspace(0,1,4),
                        connectivity=grid_ids,points=xyz,interfaces=np.array(interfaces,dtype=np.int64))
    (out/'boundary.json').write_text(json.dumps(boundary,indent=2)+'\n')
    gmsh.write(str(out/'mesh.msh'))
    gmsh.write(str(out/'geometry.geo_unrolled'))
    # Boundary representation error of the exported cubic maps, not CAD only.
    B=basis(np.linspace(0,1,41)); boundary_curves=[]
    for p,face,_ in edges.values():
        a=patches[p]
        edge=[a[0,:],a[-1,:],a[:,0],a[:,-1]][face]
        boundary_curves.append(LineString(B@edge))
    boundary_error=unary_union(boundary_curves).hausdorff_distance(domain.boundary)
    q,w=np.polynomial.legendre.leggauss(8)
    area=float(np.einsum('pij,i,j->',metrics(patches,(q+1)/2)[0],w/2,w/2))
    quality=dict(patches=len(patches),interfaces=len(interfaces),boundary_faces=len(boundary),
                 min_jacobian=float(det.min()),min_scaled_jacobian=float(scaled.min()),
                 max_map_condition=float(cond.max()),p99_map_condition=float(np.quantile(cond,.99)),
                 min_bernstein_jacobian=float(coefficients.min()),certified_patches=int(certified.sum()),
                 boundary_hausdorff_normalized=float(boundary_error),relative_area_error=abs(area-domain.area)/domain.area,
                 area=area,domain_area=domain.area,domain_components=len(polygons),
                 holes=sum(len(p.interiors) for p in polygons),**convergence,
                 source=str(eq.path),source_sha256=hashlib.sha256(eq.path.read_bytes()).hexdigest(),
                 gmsh_version=gmsh.__version__,geometry_degree=3,coordinates='R/R_axis, flipped Z/R_axis',
                 algorithm=algorithm,curvature_sizing=curvature,target_size=h,
                 rho_min=eq.rmin,rho_max=eq.rmax,private_rho_min=eq.pmin,
                 status='candidate',classifier='Python reconstruction; not a direct Fortran comparison')
    if det.min()>0 and certified.all() and scaled.min()>.2 and cond.max()<12 and boundary_error<1e-4:
        quality['status']='geometry_checks_passed'
    (out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,9),constrained_layout=True)
    outlines=[]
    B=basis(np.linspace(0,1,12))
    for a in patches:
        outlines.append(np.concatenate([B@a[:,0],B@a[-1,:],(B@a[:,-1])[::-1],(B@a[0,:])[::-1]]))
    for ax,vals,title in zip(axes,[scaled.min(axis=(1,2)),cond.max(axis=(1,2))],
                             ['Minimum scaled Jacobian','Maximum map condition number']):
        pc=PolyCollection(outlines,array=vals,cmap='viridis',edgecolors='#222222',linewidths=.35)
        ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal')
        ax.plot(*eq.xpoint,'r+',ms=9,label='Magnetic X-point')
        ax.set_title(title);ax.set_xlabel('Normalized R');ax.set_ylabel('Normalized Z')
        fig.colorbar(pc,ax=ax,shrink=.6);ax.legend(loc='lower left')
    fig.suptitle(f'{name.upper()}: {len(patches)} conforming cubic quadrilateral patches')
    fig.savefig(out/'patch_quality.png',dpi=180);plt.close(fig)
    # Interior reference lines demonstrate the tensor-product patch maps.
    B=basis(np.linspace(0,1,17));C=basis(np.linspace(0,1,5))
    lines=[]
    for a in patches:
        lines.extend(np.einsum('ai,bj,ijc->bac',B,C,a))
        lines.extend(np.einsum('ai,bj,ijc->abc',C,B,a))
    fig,ax=plt.subplots(figsize=(7,10),constrained_layout=True)
    ax.add_collection(LineCollection(lines,color='#3b6f8f',linewidths=.25))
    ax.add_collection(PolyCollection(outlines,facecolors='none',edgecolors='black',linewidths=.6))
    ax.autoscale();ax.set_aspect('equal');ax.set_title(f'{name.upper()}: patch maps with 4 x 4 display subdivisions')
    ax.set_xlabel('Normalized R');ax.set_ylabel('Normalized Z')
    fig.savefig(out/'patch_grid.png',dpi=180);plt.close(fig)
    gmsh.finalize()
    print(json.dumps(quality,indent=2),flush=True)
    return quality


if __name__=='__main__':
    faulthandler.dump_traceback_later(90,repeat=True)
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=list(CASES),required=True)
    p.add_argument('--genex',type=Path,default=Path.home()/'genex')
    p.add_argument('--size',type=float)
    p.add_argument('--algorithm',type=int,default=11)
    p.add_argument('--curvature',type=float,default=0)
    p.add_argument('--output',type=Path)
    p.add_argument('--reuse-domain',type=Path)
    args=p.parse_args()
    print(f'Loading {args.case} equilibrium',flush=True)
    eq=Equilibrium(args.case,args.genex)
    out=args.output or HERE/'meshes'/args.case;out.mkdir(parents=True,exist_ok=True)
    if args.reuse_domain:
        domain=shape(json.loads((args.reuse_domain/'domain.geojson').read_text()))
        report=json.loads((args.reuse_domain/'quality.json').read_text())
        assert report['source_sha256']==hashlib.sha256(eq.path.read_bytes()).hexdigest()
        convergence={k:v for k,v in report.items() if k.startswith('extraction_')}
    else:
        print('Extracting domain on 800 and 1600 point R grids',flush=True)
        coarse=eq.domain(800); fine=eq.domain(1600)
        convergence=dict(extraction_hausdorff=float(fine.boundary.hausdorff_distance(coarse.boundary)),
                         extraction_relative_area_difference=abs(fine.area-coarse.area)/fine.area)
        print(f'Boundary extraction comparison: {convergence}',flush=True)
        domain=fine.simplify(2e-6,preserve_topology=True)
    (out/'domain.geojson').write_text(json.dumps(mapping(domain)))
    export(args.case,eq,domain,convergence,out,args.size or eq.h,args.algorithm,args.curvature)
