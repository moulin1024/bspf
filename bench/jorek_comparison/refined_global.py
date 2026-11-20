"""Residual-selected tensor spline enrichment of one global physical BSPF solve."""
import numpy as np
from math import comb
from scipy.interpolate import BSpline
from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import splu, spsolve

from campaign import rule
from global_bspf import PhysicalBSPF, exterior, outer


def prolong_cardinal(old,shape,degree):
    """Represent a tensor cardinal spline on the grid with half the knot spacing."""
    # Cardinal B-spline two-scale relation. Index zero represents knot -degree.
    mask=np.array([comb(degree+1,k)/2**degree for k in range(degree+2)])
    for axis in (0,1):
        old=np.moveaxis(old,axis,0)
        up=np.zeros((shape[axis],)+old.shape[1:])
        for k,weight in enumerate(mask):
            shift=k-degree
            begin=max(0,(-shift+1)//2)
            end=min(len(old),(len(up)-shift+1)//2)
            up[2*begin+shift:2*end+shift:2]+=weight*old[begin:end]
        old=np.moveaxis(up,0,axis)
    return old


class RefinedGlobalBSPF(PhysicalBSPF):
    def pilot(self,current,wall_flux,fraction=0.90):
        if not 0 < fraction <= 1:
            raise ValueError("marking fraction must lie in (0, 1]")
        self.current,self.wall_flux = current,wall_flux
        self.domain_level = False
        self.local_coef = None
        self.solve(current,wall_flux,keep_system=True)
        self.pilot_coef = self.coef.copy()
        self.mark(current,fraction)

    def mark(self,current,fraction):
        """Mark coarse physical bins carrying the specified share of squared PDE residual."""
        r,z,w = self.volume
        error = np.empty_like(r)
        for first in range(0,len(r),4096):
            sl = slice(first,first+4096)
            R,Z = r[sl],z[sl]
            a,b = self.tables(R,Z,2)
            radial = (a[2]-a[1]/R[:,None])@self.coef
            vertical = a[0]@self.coef
            defect = np.sum(radial*b[0]+vertical*b[2],axis=1)-current(R,Z)
            if self.local_coef is not None:
                near = self.near(R,Z)
                if near.any():
                    ids,a_local,b_local = self.stencil(R[near],Z[near],2)
                    lap = (outer(a_local[2]-a_local[1]/R[near,None],b_local[0])
                           +outer(a_local[0],b_local[2]))
                    defect[near] += np.sum(lap*self.local_coef[ids],axis=1)
            error[sl] = defect
        self.bin_width = min(self.width)/8
        bins = np.floor((np.column_stack((r,z))-self.start)/self.bin_width).astype(int)
        bins,ids = np.unique(bins,axis=0,return_inverse=True)
        energy = np.bincount(ids,weights=w*error**2)
        sort = np.argsort(-energy)
        count = np.searchsorted(np.cumsum(energy[sort]),fraction*energy.sum())+1
        self.marked = bins[sort[:count]]

    def remark(self,fraction=0.90):
        """Grow the existing footprint using the PDE residual of the refined solution."""
        if self.local_coef is None:
            raise ValueError("refine the pilot before remarking")
        if not 0 < fraction <= 1:
            raise ValueError("marking fraction must lie in (0, 1]")
        previous = self.marked
        self.mark(self.current,fraction)
        self.marked = np.unique(np.vstack((previous,self.marked)),axis=0)

    def prepare(self,h,support_padding=None):
        if support_padding is not None and (not np.isfinite(support_padding) or support_padding < 0):
            raise ValueError("support padding must be finite and nonnegative")
        self.h = h
        self.degree = 5
        self.cardinal = BSpline.basis_element(np.arange(self.degree+2),extrapolate=False)
        # Retain splines intersecting a marked bin. Each spline is C4 across its support boundary.
        blocks = []
        for cell in self.marked:
            if support_padding is None:
                lo = np.floor((cell-1)*self.bin_width/h).astype(int)-self.degree
                hi = np.ceil((cell+2)*self.bin_width/h).astype(int)
            else:
                # Only keep complete supports within a fixed physical box.
                lo = np.ceil((cell*self.bin_width-support_padding)/h-1e-10).astype(int)
                hi = np.floor(((cell+1)*self.bin_width+support_padding)/h+1e-10).astype(int)-self.degree
            if np.any(hi <= lo):
                continue
            block = np.indices(hi-lo).reshape(2,-1).T + lo
            blocks.append(block)
        if not blocks:
            raise ValueError("no complete spline support fits this spacing and padding")
        keys = np.unique(np.concatenate(blocks),axis=0)
        self.offset = keys.min(axis=0)
        self.lookup = np.full(keys.max(axis=0)-self.offset+1,-1,int)
        self.lookup[tuple((keys-self.offset).T)] = np.arange(len(keys))
        self.nlocal = len(keys)
        self.lower = self.start+h*keys.min(axis=0)
        self.upper = self.start+h*(keys.max(axis=0)+self.degree+1)

    def near(self,R,Z):
        return (R>=self.lower[0])&(R<=self.upper[0])&(Z>=self.lower[1])&(Z<=self.upper[1])

    def stencil(self,R,Z,order):
        axis=[]
        for x,lo in zip((R,Z),self.start):
            t = (x-lo)/self.h
            knots = np.floor(t).astype(int)[:,None]-np.arange(self.degree+1)
            values = [self.cardinal(t[:,None]-knots,nu=k)/self.h**k for k in range(order+1)]
            axis.append((knots,values))
        (i,a),(j,b) = axis
        i,j = i[:,:,None]-self.offset[0], j[:,None,:]-self.offset[1]
        ids = self.lookup[i.clip(0,self.lookup.shape[0]-1),j.clip(0,self.lookup.shape[1]-1)]
        valid = (i>=0)&(i<self.lookup.shape[0])&(j>=0)&(j<self.lookup.shape[1])
        return np.where(valid,ids,-1).reshape(len(R),-1),a,b

    def quadrature(self,points):
        """Composite Gauss integration with physical parametric spans bounded by h."""
        if points is None:
            yield False,self.volume
            yield True,self.wall
            return
        q,w = np.polynomial.legendre.leggauss(points)
        q,w = (q+1)/2,w/2
        controls = self.controls
        # A cubic Bezier derivative lies in the convex hull of its derivative controls.
        bounds = np.stack([3*np.abs(np.diff(controls,axis=axis)).max(axis=(0,1,2))
                           for axis in (1,2)],axis=1)
        intersects = ((controls.max(axis=(1,2)).T>=self.lower)
                      &(controls.min(axis=(1,2)).T<=self.upper)).all(axis=1)
        panels = np.maximum(1,np.ceil(bounds/self.h).astype(int))
        panels[~intersects] = 1
        # Round up to powers of two to batch cells with the same tensor rule.
        panels = 2**np.ceil(np.log2(panels)).astype(int)
        def composite(n):
            return ((np.arange(n)[:,None]+q)/n).ravel(),np.tile(w/n,n)
        coef = self.ref.coeff[:2].reshape(2,16,-1).transpose(2,0,1)
        for pair in np.unique(panels,axis=0):
            cells = np.flatnonzero(np.all(panels==pair,axis=1))
            qs,ws = composite(pair[0])
            qt,wt = composite(pair[1])
            if tuple(pair)==(1,1) and self.order>points:
                base,bw = np.polynomial.legendre.leggauss(self.order)
                qs=qt=(base+1)/2
                ws=wt=bw/2
            tables = rule(qs,qt)
            weights = np.outer(ws,wt).ravel()
            batch = max(1,65536//len(weights))
            for offset in range(0,len(cells),batch):
                c = coef[cells[offset:offset+batch]]
                pos,ds,dt = [c@tab for tab in tables]
                jac = ds[:,0]*dt[:,1]-ds[:,1]*dt[:,0]
                if np.min(jac)<=0:
                    raise ValueError('invalid curved element Jacobian')
                yield False,(pos[:,0].ravel(),pos[:,1].ravel(),(jac*weights).ravel())
        edge_q,edge_w = composite(panels.max())
        if len(edge_q)<self.order:
            edge_q,edge_w = np.polynomial.legendre.leggauss(self.order)
            edge_q,edge_w = (edge_q+1)/2,edge_w/2
        yield True,exterior(self.ref,edge_q,edge_w)

    def refine(self,h,*,support_padding=None,points_per_span=6):
        """Refine across the domain, or restrict enrichment via explicit support padding."""
        if not np.isfinite(h) or h <= 0:
            raise ValueError("knot spacing must be finite and positive")
        if points_per_span is not None and (not isinstance(points_per_span,int) or points_per_span<2):
            raise ValueError("points_per_span must be an integer >= 2 or None")
        if support_padding is None and points_per_span is not None:
            return self.refine_domain(h,points_per_span=points_per_span)
        self.domain_level = False
        self.prepare(h,support_padding)
        n = self.coef.size
        gram,global_rhs = self.global_system
        cross = np.zeros((self.nlocal,n))
        sparse = csr_matrix((self.nlocal,self.nlocal))
        rhs = np.zeros(n+self.nlocal)
        rhs[:n] = global_rhs
        penalty = self.wall_weight
        if points_per_span is not None:
            gram = np.zeros_like(gram)
            rhs[:n] = 0.
        self.assembly_points = 0
        for wall,(r,z,w) in self.quadrature(points_per_span):
            self.assembly_points += len(r)
            for first in range(0,len(r),2048):
                sl = slice(first,first+2048)
                R,Z = r[sl],z[sl]
                order = 0 if wall else 2
                a,b = self.tables(R,Z,order)
                def operator(x,y):
                    if wall:
                        return outer(x[0],y[0])
                    return outer(x[2]-x[1]/R[:,None],y[0])+outer(x[0],y[2])
                weight = np.sqrt(w[sl]*(penalty if wall else 1))
                target = weight*(self.wall_flux(R,Z) if wall else self.current(R,Z))
                G = operator(a,b)*weight[:,None]
                if points_per_span is not None:
                    target -= G@self.pilot_coef.ravel()
                    gram += G.T@G
                    rhs[:n] += G.T@target
                near = self.near(R,Z)
                if not np.any(near):
                    continue
                R,Z,G,weight,target = R[near],Z[near],G[near],weight[near],target[near]
                ids,ar,az = self.stencil(R,Z,order)
                row,col = np.nonzero(ids>=0)
                S = csr_matrix(((operator(ar,az)*weight[:,None])[row,col],(row,ids[row,col])),
                               shape=(len(R),self.nlocal))
                cross += S.T@G
                sparse += S.T@S
                rhs[n:] += S.T@target
        diag = np.r_[gram.diagonal(), sparse.diagonal()]
        active = diag>diag.max()*1e-16
        # Very small intersections of a support with the curved wall can be nearly dependent.
        ridge = 1e-12
        if points_per_span is not None:
            # A 2D second-derivative Gram diagonal scales as h^-2, while a
            # coefficient mass weight scales as h^2. Keep the physical bias
            # bounded as the number of spline translates grows.
            ridge = max(16*np.finfo(float).eps, ridge*min(1.,(32*h/min(self.width))**4))
        u = np.zeros(len(rhs))
        if self.nlocal < 16000:
            full = bmat([[csr_matrix(gram),csr_matrix(cross.T)],[csr_matrix(cross),sparse]],format='csr')
            inv = diags(1/np.sqrt(diag[active]))
            scaled = (inv@full[active][:,active]@inv).tocsc()
            u[active] = inv@spsolve(scaled+ridge*eye(scaled.shape[0],format='csc'),inv@rhs[active])
        else:
            # Eliminate compact splines first; a dense border in the sparse LU would cause fill.
            keep = active[n:]
            if not active[:n].all():
                raise ValueError('global basis vanished on the domain')
            g = 1/np.sqrt(diag[:n])
            l = 1/np.sqrt(diag[n:][keep])
            D = diags(l)
            local = (D@sparse[keep][:,keep]@D + ridge*eye(len(l))).tocsc()
            lu = splu(local)
            X = cross[keep]*l[:,None]*g
            schur = (gram*g[:,None])*g + ridge*np.eye(n)
            for first in range(0,n,16):
                sl = slice(first,first+16)
                schur[:,sl] -= X.T@lu.solve(X[:,sl])
            load = rhs[n:][keep]*l
            z = np.linalg.solve(schur,rhs[:n]*g-X.T@lu.solve(load))
            u[:n] = g*z
            u[n:][keep] = l*lu.solve(load-X@z)
        residual = np.r_[gram@u[:n]+cross.T@u[n:],cross@u[:n]+sparse@u[n:]]-rhs
        self.residual = np.linalg.norm(residual[active])/max(np.linalg.norm(rhs[active]),np.finfo(float).tiny)
        self.active_dofs = int(active.sum())
        self.coef = u[:n].reshape(self.coef.shape)
        if points_per_span is not None:
            self.coef += self.pilot_coef
        self.local_coef = np.r_[u[n:],0.]

    def refine_domain(self,h,*,degree=5,points_per_span=6):
        """Fit a domain-wide spline correction, prolonging the previous dyadic level."""
        if not np.isfinite(h) or h<=0 or not isinstance(points_per_span,int) or points_per_span<2:
            raise ValueError("expected positive knot spacing and at least two Gauss points per span")
        if not isinstance(degree,(int,np.integer)) or degree<2:
            raise ValueError("the strong-form correction needs spline degree at least two")
        shape=np.ceil(self.width/h).astype(int)+degree
        initial=np.zeros(shape)
        if (getattr(self,'domain_level',False) and self.local_coef is not None
                and self.degree==degree and np.isclose(self.h,2*h,rtol=2e-14,atol=0)):
            initial=prolong_cardinal(self.local_coef[:-1].reshape(self.lookup.shape),shape,degree)
        self.domain_level=False
        self.h=h
        self.degree=degree
        self.cardinal=BSpline.basis_element(np.arange(degree+2),extrapolate=False)
        self.offset=np.full(2,-degree)
        self.nlocal=int(np.prod(shape))
        self.lookup=np.arange(self.nlocal).reshape(shape)
        self.lower=self.start
        self.upper=self.start+self.width
        coeff=initial.ravel()
        load=np.zeros(self.nlocal)
        # Merge sparse block contributions pairwise to avoid copying the entire
        # assembled matrix for each quadrature batch.
        partial={}
        self.assembly_points=0
        for wall,(r,z,w) in self.quadrature(points_per_span):
            self.assembly_points+=len(r)
            for first in range(0,len(r),4096):
                sl=slice(first,first+4096)
                R,Z=r[sl],z[sl]
                order=0 if wall else 2
                a,b=self.tables(R,Z,order)
                ids,x,y=self.stencil(R,Z,order)
                if wall:
                    values=outer(x[0],y[0])
                    pilot=np.sum((a[0]@self.pilot_coef)*b[0],axis=1)
                    target=self.wall_flux(R,Z)-pilot
                    weight=np.sqrt(w[sl]*self.wall_weight)
                else:
                    values=outer(x[2]-x[1]/R[:,None],y[0])+outer(x[0],y[2])
                    radial=(a[2]-a[1]/R[:,None])@self.pilot_coef
                    pilot=np.sum(radial*b[0]+(a[0]@self.pilot_coef)*b[2],axis=1)
                    target=self.current(R,Z)-pilot
                    weight=np.sqrt(w[sl])
                row,col=np.nonzero(ids>=0)
                S=csr_matrix(((values*weight[:,None])[row,col],(row,ids[row,col])),
                             shape=(len(R),self.nlocal))
                load+=S.T@(weight*target-S@coeff)
                block=(S.T@S).tocsr()
                level=0
                while level in partial:
                    block+=partial.pop(level)
                    level+=1
                partial[level]=block
        gram=sum(partial.values())
        partial.clear()
        diag=gram.diagonal()
        active=diag>diag.max()*1e-16
        scaling=diags(1/np.sqrt(diag[active]))
        gram=gram[active][:,active].tocsr()
        system=(scaling@gram@scaling).tocsc()
        ridge=max(16*np.finfo(float).eps,1e-12*min(1.,(32*h/min(self.width))**4))
        delta=scaling@spsolve(system+ridge*eye(system.shape[0],format='csc'),scaling@load[active])
        defect=gram@delta-load[active]
        self.residual=np.linalg.norm(defect)/max(np.linalg.norm(load[active]),np.finfo(float).tiny)
        coeff[active]+=delta
        self.coef=self.pilot_coef.copy()
        self.local_coef=np.r_[coeff,0.]
        self.correction_unknowns=int(active.sum())
        # Count inherited nonzero boundary-extension coefficients conservatively.
        self.active_dofs=int(np.count_nonzero(active | (coeff!=0))+self.coef.size)
        self.domain_level=True

    def evaluate(self,R,Z):
        R,Z = np.broadcast_arrays(R,Z)
        base = super().evaluate(R,Z)
        if getattr(self,'local_coef',None) is None:
            return base
        near = self.near(R,Z)
        if np.any(near):
            ids,a,b = self.stencil(R[near],Z[near],1)
            weights = self.local_coef[ids]
            for v,(i,j) in zip(base,[(0,0),(1,0),(0,1)]):
                v[near] += np.sum(outer(a[i],b[j])*weights,axis=1)
        return base
