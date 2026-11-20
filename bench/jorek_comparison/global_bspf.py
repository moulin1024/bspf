"""Physical-coordinate BSPF least squares on a curved JOREK integration mesh."""
from time import perf_counter
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.linalg import cho_factor, cho_solve

from campaign import HybridAxis, rule


def geometry(ref, q, w):
    tables = rule(q)
    coef = ref.coeff[:2].reshape(2,16,-1).transpose(2,0,1)
    pos, ds, dt = (coef @ tab for tab in tables)
    jac = ds[:,0]*dt[:,1] - ds[:,1]*dt[:,0]
    if np.min(jac) <= 0 or np.min(pos[:,0]) <= 0:
        raise ValueError('expected positive R and positively oriented quadrilaterals')
    return pos[:,0].ravel(), pos[:,1].ravel(), (jac*np.outer(w,w).ravel()).ravel()


def exterior(ref, q, w):
    # Match the complete cubic edge using four samples, including reversed edges.
    coef = ref.coeff[:2].reshape(2,16,-1).transpose(2,0,1)
    sides = [([0],None,2),([1],None,2),(None,[0],1),(None,[1],1)]
    samples = np.linspace(0,1,4)
    edge_points = []
    for s,t,_ in sides:
        edge = coef @ rule(samples if s is None else s,samples if t is None else t)[0]
        edge_points.append(edge.transpose(0,2,1))
    points = np.concatenate(edge_points).round(11)
    noncollapsed = np.any(points != points[:,0:1,:],axis=(1,2))
    indices = np.flatnonzero(noncollapsed)
    forward = points[indices].reshape(-1,8)
    backward = points[indices,::-1].reshape(-1,8)
    first = np.argmax(forward != backward,axis=1)
    rows = np.arange(len(indices))
    reverse = forward[rows,first] > backward[rows,first]
    forward[reverse] = backward[reverse]
    _,locations,counts = np.unique(forward,axis=0,return_index=True,return_counts=True)
    if np.any(counts > 2):
        raise ValueError('nonmanifold edge')
    boundary = indices[locations[counts == 1]]
    if not len(boundary):
        raise ValueError("missing exterior boundary for the Dirichlet problem")
    coords, weights = [], []
    for side,(s,t,tangent) in enumerate(sides):
        elements = boundary[boundary//ref.count == side] % ref.count
        basis = rule(q if s is None else s,q if t is None else t)
        coords.append(coef[elements] @ basis[0])
        weights.append(np.linalg.norm(coef[elements] @ basis[tangent],axis=1)*w)
    pos = np.concatenate(coords)
    return pos[:,0].ravel(),pos[:,1].ravel(),np.concatenate(weights).ravel()


def tabulate(axis, x, start, width, order):
    # Build the Chebyshev table once for all cardinal functions and derivatives.
    coeff = axis.derivatives[:order+1]
    length = max(np.flatnonzero(np.max(abs(c),axis=1)>1e-17)[-1]+1 for c in coeff)
    basis = chebvander(2*(x-start)/width-1,length-1)
    return [(basis @ c[:length])/width**k for k,c in enumerate(coeff)]


def outer(a,b):
    return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)


class PhysicalBSPF:
    """Approximate the flux in one global tensor space restricted to the domain."""
    def __init__(self, ref, degree=(5,13), order=4, *, wall_weight=None):
        self.ref = ref
        self.order = order
        self.axes = [HybridAxis(1,degree=d,nodes='symmetric_qr') for d in degree]
        q,w = np.polynomial.legendre.leggauss(order)
        q,w = (q+1)/2,w/2
        self.volume = geometry(ref,q,w)
        self.wall = exterior(ref,q,w)
        # Cubic Bezier control-point bounds contain the entire physical mesh.
        transform = np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,0]])
        corner = ref.coeff[:2].reshape(2,4,4,-1)
        lattice = np.empty((2,4,4,ref.count))
        for vertex,(i,j) in enumerate([(0,0),(2,0),(2,2),(0,2)]):
            for v in range(4):
                lattice[:,i+(v%2),j+(v//2)] = corner[:,v,vertex]
        controls = np.einsum('ia,dabp,jb->dijp',transform,lattice,transform)
        self.controls = controls
        self.start = controls.min(axis=(1,2,3))
        self.width = controls.max(axis=(1,2,3))-self.start

        self.wall_weight = ((max(axis.n for axis in self.axes)/min(self.width))**3
                            if wall_weight is None else float(wall_weight))
        if not np.isfinite(self.wall_weight) or self.wall_weight <= 0:
            raise ValueError("wall weight must be finite and positive")

    def tables(self,R,Z,order):
        return [tabulate(axis,x,lo,width,order)
                for axis,x,lo,width in zip(self.axes,(R,Z),self.start,self.width)]

    def solve(self,current,wall_flux,*,keep_system=False):
        n = np.prod([axis.n for axis in self.axes])
        gram = np.zeros((n,n))
        rhs = np.zeros(n)
        # Dimensional balance between squared PDE and trace residual integrals.
        wall_weight = self.wall_weight
        for wall,(r,z,w) in [(False,self.volume),(True,self.wall)]:
            for first in range(0,len(r),2048):
                sl = slice(first,first+2048)
                R,Z = r[sl],z[sl]
                a,b = self.tables(R,Z,0 if wall else 2)
                if wall:
                    mat = outer(a[0],b[0])
                    target = wall_flux(R,Z)
                else:
                    mat = outer(a[2]-a[1]/R[:,None],b[0])+outer(a[0],b[2])
                    target = current(R,Z)
                weight = np.sqrt(w[sl]*(wall_weight if wall else 1))
                mat *= weight[:,None]
                gram += mat.T@mat
                rhs += mat.T@(target*weight)
        if keep_system:
            self.global_system = gram,rhs
        scale = np.sqrt(np.diag(gram))
        scaled = gram/scale[:,None]/scale
        start = perf_counter()
        coef = cho_solve(cho_factor(scaled),rhs/scale)/scale
        self.factor_and_solve_s = perf_counter()-start
        self.residual = np.linalg.norm(gram@coef-rhs)/np.linalg.norm(rhs)
        self.coef = coef.reshape(self.axes[0].n,self.axes[1].n)

    def evaluate(self,R,Z):
        """Return flux and its two physical derivatives at points inside the domain."""
        R,Z = np.broadcast_arrays(R,Z)
        a,b = self.tables(R.ravel(),Z.ravel(),1)
        def field(i,j):
            return np.sum((a[i]@self.coef)*b[j],axis=1).reshape(R.shape)
        return field(0,0),field(1,0),field(0,1)

    def score(self,ref,exact,order,observe=None):
        q,w = np.polynomial.legendre.leggauss(order)
        r,z,w = geometry(ref,(q+1)/2,w/2)
        sums = np.zeros(4)
        for first in range(0,len(r),8192):
            sl = slice(first,first+8192)
            R,Z,wt = r[sl],z[sl],w[sl]
            u,ur,uz = self.evaluate(R,Z)
            val,gr,gz,_ = exact(R,Z)
            if observe is not None:
                observe(R,Z,wt,ur,uz,gr,gz)
            sums += [wt@(u-val)**2,wt@val**2,
                     (wt/R**2)@((ur-gr)**2+(uz-gz)**2),
                     (wt/R**2)@(gr**2+gz**2)]
        return np.sqrt(sums[[0,2]]/sums[[1,3]])
