"""Conforming matrix-free Galerkin Dirichlet Poisson/reaction-diffusion solves.

Build directly from an axis and geometry: constructing the old mapped Plan
would already assemble dense patch matrices. Only one-dimensional basis arrays,
quadrature metrics, connectivity, and an optional sparse Q1 preconditioner are
stored here. No dense patch stiffness, inverse, or Schur complement is formed.
"""
import time
import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator,cg,splu


class MatrixFreeDirichlet:
    """Solve -Delta u + sigma*u=f with conforming patchwise Galerkin CG.

    Geometry supplies count, map(patch,s,t), and boundary_edges (side order:
    bottom/right/top/left). Optional quad_ids identifies mesh vertices; optional
    edge_ids (patch, side) distinguishes different edges with the same endpoints.
    Edge orientation follows increasing vertex ID. One axis
    is shared by both directions and all patches. sigma must be nonnegative.

    preconditioner: 'auto' selects 'tensor' for a single patch with all four
    Dirichlet edges, and 'q1_lu' otherwise. 'tensor' uses one-dimensional
    generalized eigenproblems; 'q1_lu' uses sparse low-order factorization;
    'q1_amg' uses optional PyAMG; 'jacobi' uses the Galerkin diagonal.
    operator_mode: 'auto' uses separable metric contractions when economical,
    with a roundoff-level truncation bound; 'quadrature' disables compression.
    quadrature_rule optionally supplies (nodes, weights) on [0,1]; weights must
    be positive and sum to one. Its length must equal quadrature_order.
    Boundary lifting and iteration use the same operator. Final defect
    correction uses the original quadrature operator and a preconditioned
    stopping measure to prevent compressed-operator error from accumulating
    when the solution is overresolved. No dense patch matrices are introduced.
    """
    def __init__(self,axis,geometry,quadrature_order=None,*,sigma=0.,preconditioner='auto',operator_mode='auto',quadrature_rule=None):
        start=time.perf_counter()
        if not np.isfinite(sigma) or sigma<0:raise ValueError('CG requires a nonnegative finite shift')
        if preconditioner not in ('auto','tensor','q1_lu','q1_amg','jacobi'):raise ValueError('unknown preconditioner')
        if operator_mode not in ('auto','quadrature'):raise ValueError('unknown operator mode')
        self.axis,self.geometry,self.n=axis,geometry,int(axis.n)
        self.sigma=float(sigma);self.operator_mode=operator_mode
        order=max(48,2*self.n+12) if quadrature_order is None else quadrature_order
        if not isinstance(order,(int,np.integer)) or order<self.n:raise ValueError('quadrature_order must be an integer >= axis.n')
        if quadrature_rule is None:
            z,w=leggauss(order);self.q=(z+1)/2;self.w=w/2
        else:
            # Optional positive nodal rule on [0,1], e.g. BSPF interpolatory weights.
            points,weights=(np.asarray(v,dtype=float) for v in quadrature_rule)
            if (points.shape!=(order,) or weights.shape!=(order,)
                or not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights))
                or np.any(np.diff(points)<=0) or points[0]<0 or points[-1]>1
                or np.any(weights<=0) or not np.isclose(weights.sum(),1.,rtol=1e-12,atol=1e-14)):
                raise ValueError('quadrature_rule requires ordered unit-interval nodes and positive normalized weights')
            self.q=points.copy();self.w=weights.copy()
        self.basis,self.derivative=axis.evaluate(self.q,1)
        self._topology()
        maps=[self._map(p,self.q[:,None],self.q[None,:]) for p in range(geometry.count)]
        self.points=np.stack([m[0] for m in maps])
        qweight=np.outer(self.w,self.w)
        self.weights=np.stack([qweight*m[3].reshape(order,order) for m in maps])
        self.hss=np.stack([(qweight.ravel()*np.sum(m[2]**2,axis=1)/m[3]).reshape(order,order) for m in maps])
        self.htt=np.stack([(qweight.ravel()*np.sum(m[1]**2,axis=1)/m[3]).reshape(order,order) for m in maps])
        self.hst=np.stack([(-qweight.ravel()*np.sum(m[1]*m[2],axis=1)/m[3]).reshape(order,order) for m in maps])
        self._tensor_terms=[None]*len(self.ids)
        self.metric_compression=[]
        if operator_mode=='auto':self._compress_metrics()
        self.operator=LinearOperator((len(self.free),len(self.free)),matvec=self._free_apply,dtype=float)
        self._factor=None;self._amg=None;self.low_order_matrix=None
        tensor_eligible=(len(self.ids)==1 and np.array_equal(
            self.free,self.ids[0].reshape(self.n,self.n)[1:-1,1:-1].ravel()))
        if preconditioner=='auto':preconditioner='tensor' if tensor_eligible else 'q1_lu'
        self.preconditioner_kind=preconditioner
        if preconditioner=='tensor':
            if not tensor_eligible:raise ValueError('tensor preconditioning requires one patch with four Dirichlet edges')
            b,d=self.basis,self.derivative
            mass=b.T@(self.w[:,None]*b);stiffness=d.T@(self.w[:,None]*d)
            eigenvalues,self._tensor_vectors=la.eigh(stiffness[1:-1,1:-1],mass[1:-1,1:-1])
            denominator=(self.hss.sum()*eigenvalues[:,None]+self.htt.sum()*eigenvalues[None,:]
                         +self.sigma*self.weights.sum())
            if np.any(denominator<=0):raise ValueError('nonpositive tensor preconditioner spectrum')
            self._tensor_inverse=1/denominator
            self.preconditioner=LinearOperator(self.operator.shape,matvec=self._tensor_precondition,dtype=float)
            self.preconditioner_nnz=self._tensor_vectors.size+self._tensor_inverse.size
        elif preconditioner=='jacobi':
            b,d=self.basis,self.derivative;b2,d2,bd=b*b,d*d,b*d
            diagonal=d2.T@self.hss@b2+b2.T@self.htt@d2+2*bd.T@self.hst@bd
            if self.sigma:diagonal+=self.sigma*(b2.T@self.weights@b2)
            self.diagonal=self._scatter(diagonal)[self.free]
            if np.any(self.diagonal<=0):raise ValueError('nonpositive operator diagonal')
            self.preconditioner=LinearOperator(self.operator.shape,matvec=lambda x:x/self.diagonal,dtype=float)
            self.preconditioner_nnz=len(self.diagonal)
        else:
            matrix=self._q1_matrix();self.low_order_matrix=matrix[self.free][:,self.free].tocsc()
            if preconditioner=='q1_lu':
                self._factor=splu(self.low_order_matrix,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.)
                self.preconditioner=LinearOperator(self.operator.shape,matvec=self._factor.solve,dtype=float)
                self.preconditioner_nnz=self._factor.L.nnz+self._factor.U.nnz
            else:
                try:import pyamg
                except ImportError as error:raise ImportError("q1_amg requires pyamg; use q1_lu or jacobi") from error
                smoother=('gauss_seidel',{'sweep':'symmetric'})
                self._amg=pyamg.smoothed_aggregation_solver(self.low_order_matrix.tocsr(),symmetry='symmetric',
                    presmoother=smoother,postsmoother=smoother,coarse_solver='splu',max_coarse=32)
                self.preconditioner=self._amg.aspreconditioner(cycle='V')
                self.preconditioner_nnz=sum(level.A.nnz for level in self._amg.levels)
        self.setup_seconds=time.perf_counter()-start
        self.operator_array_bytes=sum(a.nbytes for a in (self.q,self.w,self.basis,self.derivative,self.nodes,
            self.ids,self.boundary,self.free,self.points,self.weights,self.hss,self.htt,self.hst))
        self.operator_array_bytes+=sum(a.nbytes for terms in self._tensor_terms if terms is not None for a in terms)

    def _tensor_precondition(self,rhs):
        v=self._tensor_vectors;c=np.asarray(rhs).reshape(self.n-2,self.n-2)
        return (v@((v.T@c@v)*self._tensor_inverse)@v.T).ravel()

    def _compress_metrics(self):
        """Replace economical separable quadrature tensors by 1D contractions.

        Each weighted metric is fitted to a Frobenius error of at most 32
        eps times the larger diagonal metric norm. Cross terms are paired to
        preserve symmetry. Retain the original metrics for residual checks and
        fall back per patch for high ranks or strongly anisotropic metrics.
        No two-dimensional stiffness matrix or inverse is formed.
        """
        b,d=self.basis,self.derivative;n=self.n;q=len(self.q)
        for p in range(len(self.ids)):
            scale=max(np.linalg.norm(self.hss[p]),np.linalg.norm(self.htt[p]))
            tolerance=32*np.finfo(float).eps*scale
            weight=np.outer(self.w,self.w)
            a,c,t=self.hss[p]/weight,self.htt[p]/weight,self.hst[p]/weight
            low=.5*(a+c-np.hypot(a-c,2*t))
            high=.5*(a+c+np.hypot(a-c,2*t))
            # Compression is unnecessary when metric perturbations could be
            # significant relative to the weakest elliptic direction.
            eligible=np.min(low)>1e-10*np.max(high)
            factors=[];ranks=[];errors=[];cost=0
            metrics=[self.hss[p],self.htt[p],self.hst[p]]
            if self.sigma:metrics.append(self.sigma*self.weights[p])
            for i,h in enumerate(metrics):
                u,s,v=la.svd(h,full_matrices=False,check_finite=False)
                tol=tolerance if i<3 else 32*np.finfo(float).eps*np.linalg.norm(h)
                tails=np.r_[np.sqrt(np.cumsum((s[::-1]**2)))[::-1],0.]
                rank=int(np.flatnonzero(tails<=tol)[0])
                error=float(np.linalg.norm(h-(u[:,:rank]*s[:rank])@v[:rank]))
                eligible=eligible and error<=tol
                ranks.append(rank);errors.append(error);cost+=rank*(2 if i==2 else 1)
                factors.append((u[:,:rank],s[:rank],v[:rank]))
            # Approximate multiplication counts: 2*k*n^3 versus the original
            # four gradient contractions (six when a mass term is present).
            limit=(2+(1 if self.sigma else 0))*(q/n)*(1+q/n)
            eligible=bool(eligible and cost<.8*limit)
            self.metric_compression.append(dict(ranks=ranks,terms=cost,compressed=eligible,
                                                relative_fit_errors=[e/scale for e in errors]))
            if not eligible:continue
            terms=[]
            for i,(u,s,v) in enumerate(factors):
                for j in range(len(s)):
                    left=u[:,j,None];right=(s[j]*v[j])[:,None]
                    if i==2:
                        l=d.T@(left*b);r=b.T@(right*d)
                        terms.extend(((l,r.T),(l.T,r)))
                    else:
                        bl=d if i==0 else b;br=d if i==1 else b
                        l=bl.T@(left*bl);r=br.T@(right*br)
                        # Remove only floating-point antisymmetry.
                        terms.append(((l+l.T)*.5,(r+r.T)*.5))
            self._tensor_terms[p]=(np.stack([l for l,r in terms]),np.stack([r for l,r in terms]))

    def _map(self,patch,s,t):
        points,xs,xt,jac=(np.asarray(v,dtype=float) for v in self.geometry.map(patch,s,t))
        size=np.broadcast_arrays(s,t)[0].size
        if points.shape!=(size,2) or xs.shape!=points.shape or xt.shape!=points.shape or jac.shape!=(size,):
            raise ValueError('geometry.map returned incompatible shapes')
        if not all(np.isfinite(v).all() for v in (points,xs,xt,jac)) or np.any(jac<=0):
            raise ValueError('geometry requires finite metrics and positive Jacobians')
        determinant=xs[:,0]*xt[:,1]-xs[:,1]*xt[:,0]
        if not np.allclose(determinant,jac,rtol=1e-10,atol=1e-14):raise ValueError('inconsistent geometry Jacobian')
        return points,xs,xt,jac

    def _topology(self):
        n=self.n;local=np.arange(n*n).reshape(n,n)
        self.side_nodes=[local[:,0],local[-1,:],local[:,-1],local[0,:]]
        boundary_edges=getattr(self.geometry,'boundary_edges',None)
        if boundary_edges is None:
            if not hasattr(self.geometry,'outer_start'):raise ValueError('geometry must identify its Dirichlet boundary edges')
            boundary_edges=[(p,1) for p in range(self.geometry.outer_start,self.geometry.count)]
        sides={p:[] for p in range(self.geometry.count)}
        for p,side in boundary_edges:
            if p not in sides or side not in range(4):raise ValueError('invalid boundary edge')
            sides[p].append(side)
        points=[];lookup={};ids=[];boundary=set()
        z=np.asarray(self.axis.x)
        if z.shape!=(n,) or np.any(np.diff(z)<=0) or z[0]!=0 or z[-1]!=1:raise ValueError('axis nodes must increase from 0 to 1')
        for patch in range(self.geometry.count):
            coordinates=self._map(patch,z[:,None],z[None,:])[0];patch_ids=[]
            rounded=np.round(coordinates,12) if not hasattr(self.geometry,'quad_ids') else None
            for k,point in enumerate(coordinates):
                if hasattr(self.geometry,'quad_ids'):
                    i,j=divmod(k,n);v=self.geometry.quad_ids[patch]
                    if i in (0,n-1) and j in (0,n-1):
                        key=('vertex',int(v[{(0,0):0,(n-1,0):1,(n-1,n-1):2,(0,n-1):3}[i,j]]))
                    elif j==0 or i==n-1 or j==n-1 or i==0:
                        a,b,u=((v[0],v[1],i) if j==0 else (v[1],v[2],j) if i==n-1 else
                               (v[3],v[2],i) if j==n-1 else (v[0],v[3],j))
                        edge_ids=getattr(self.geometry,'edge_ids',None)
                        if edge_ids is None:
                            key=('edge',int(min(a,b)),int(max(a,b)),int(u if a<b else n-1-u))
                        else:
                            side=0 if j==0 else 1 if i==n-1 else 2 if j==n-1 else 3
                            key=('explicit_edge',int(edge_ids[patch,side]),int(u if a<b else n-1-u))
                    else:key=('interior',patch,i,j)
                else:key=tuple(rounded[k])
                if key not in lookup:
                    lookup[key]=len(points);points.append(point)
                elif not np.allclose(points[lookup[key]],point,rtol=0.,atol=1e-11):
                    raise ValueError('shared-node geometry mismatch')
                idx=lookup[key]
                patch_ids.append(idx)
            ids.append(np.asarray(patch_ids))
            for side in sides[patch]:boundary.update(ids[-1][self.side_nodes[side]])
        self.nodes=np.asarray(points);self.ids=np.asarray(ids)
        self.boundary=np.asarray(sorted(boundary),dtype=int)
        if not len(self.boundary):raise ValueError('Dirichlet boundary required')
        self.free=np.setdiff1d(np.arange(len(self.nodes)),self.boundary)

    def _scatter(self,local):
        return np.bincount(self.ids.ravel(),weights=np.asarray(local).ravel(),minlength=len(self.nodes))

    def boundary_geometry(self,patch,side,u):
        s,t=[(u,0),(1,u),(u,1),(0,u)][side]
        points,xs,xt,_=self._map(patch,s,t)
        tangent=(xs if side%2==0 else xt)*(1 if side<2 else -1)
        speed=np.linalg.norm(tangent,axis=1)
        return points,np.column_stack((tangent[:,1],-tangent[:,0]))/speed[:,None],speed

    def apply(self,coefficients):
        """Apply the full assembled weak operator by tensor contractions."""
        c=np.asarray(coefficients)
        if np.iscomplexobj(c) or c.shape!=(len(self.nodes),) or not np.isfinite(c).all():raise ValueError('expected a finite real global coefficient vector')
        cc=c[self.ids].reshape(len(self.ids),self.n,self.n)
        if all(terms is None for terms in self._tensor_terms):return self._quadrature_apply(c)
        local=np.empty_like(cc,dtype=float)
        for p,terms in enumerate(self._tensor_terms):
            if terms is None:local[p]=self._quadrature_local(cc[p],p)
            else:
                left,right=terms
                local[p]=np.sum(left@cc[p]@right,axis=0)
        return self._scatter(local)

    def _quadrature_local(self,cc,p):
        b,d=self.basis,self.derivative
        us=d@cc@b.T;ut=b@cc@d.T
        local=d.T@(self.hss[p]*us+self.hst[p]*ut)@b+b.T@(self.hst[p]*us+self.htt[p]*ut)@d
        if self.sigma:local+=self.sigma*(b.T@(self.weights[p]*(b@cc@b.T))@b)
        return local

    def _quadrature_apply(self,c):
        cc=c[self.ids].reshape(len(self.ids),self.n,self.n)
        local=self._quadrature_local(cc,slice(None))
        return self._scatter(local)

    def _free_apply(self,c):
        full=np.zeros(len(self.nodes));full[self.free]=c
        return self.apply(full)[self.free]

    def _q1_matrix(self):
        """Sparse bilinear FEM operator on the same nodal subcells; not the PDE matrix."""
        z,w=leggauss(2);z=(z+1)/2;w=w/2
        a,b=np.meshgrid(z,z,indexing='ij');a,b=a.ravel(),b.ravel()
        value=np.column_stack(((1-a)*(1-b),a*(1-b),a*b,(1-a)*b))
        da=np.column_stack((-(1-b),1-b,b,-b));db=np.column_stack((-(1-a),-a,a,1-a))
        h=np.diff(self.axis.x);hs,ht=np.meshgrid(h,h,indexing='ij');hs,ht=hs.ravel(),ht.ravel()
        ss,tt=np.meshgrid(self.axis.x[:-1],self.axis.x[:-1],indexing='ij')
        s=ss.ravel()[:,None]+hs[:,None]*a;t=tt.ravel()[:,None]+ht[:,None]*b
        ds=da[None,:,:]/hs[:,None,None];dt=db[None,:,:]/ht[:,None,None]
        weight=hs[:,None]*ht[:,None]*np.outer(w,w).ravel()
        rr=[];cc=[];vv=[]
        for patch,ids in enumerate(self.ids):
            _,xs,xt,jac=self._map(patch,s,t);xs=xs.reshape(-1,4,2);xt=xt.reshape(-1,4,2);jac=jac.reshape(-1,4)
            hss=weight*np.sum(xt*xt,axis=2)/jac;htt=weight*np.sum(xs*xs,axis=2)/jac
            hst=-weight*np.sum(xs*xt,axis=2)/jac
            local=(np.einsum('cqi,cq,cqj->cij',ds,hss,ds)+np.einsum('cqi,cq,cqj->cij',dt,htt,dt)
                   +np.einsum('cqi,cq,cqj->cij',ds,hst,dt)+np.einsum('cqi,cq,cqj->cij',dt,hst,ds))
            if self.sigma:local+=self.sigma*np.einsum('qi,cq,qj->cij',value,weight*jac,value)
            grid=ids.reshape(self.n,self.n)
            corners=np.column_stack((grid[:-1,:-1].ravel(),grid[1:,:-1].ravel(),grid[1:,1:].ravel(),grid[:-1,1:].ravel()))
            rr.append(np.repeat(corners,4,axis=1).ravel());cc.append(np.tile(corners,(1,4)).ravel());vv.append(local.ravel())
        return sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),
                                 shape=(len(self.nodes),len(self.nodes))).tocsr()

    def sample(self,forcing,boundary=0.):
        f=np.stack([np.broadcast_to(forcing(p) if callable(forcing) else forcing,(len(p),)) for p in self.points])
        p=self.nodes[self.boundary]
        g=np.broadcast_to(boundary(p,None) if callable(boundary) else boundary,(len(p),))
        return f,g

    def solve(self,forcing,boundary=0.,**kwargs):
        return self.solve_sampled(*self.sample(forcing,boundary),**kwargs)

    def solve_sampled(self,forcing,boundary,*,rtol=1e-11,atol=0.,maxiter=1000):
        start=time.perf_counter();f=np.asarray(forcing);g=np.asarray(boundary)
        if (np.iscomplexobj(f) or np.iscomplexobj(g) or f.shape!=self.points.shape[:2] or g.shape!=(len(self.boundary),)
                or not np.isfinite(f).all() or not np.isfinite(g).all()):raise ValueError('invalid real forcing/boundary samples')
        if not np.isfinite([rtol,atol]).all() or rtol<=0 or atol<0 or not isinstance(maxiter,(int,np.integer)) or maxiter<1:
            raise ValueError('invalid CG tolerances or iteration limit')
        load=self._scatter(self.basis.T@(self.weights*f.reshape(self.weights.shape))@self.basis)
        coefficients=np.zeros(len(self.nodes));coefficients[self.boundary]=g
        # Mixing quadrature boundary lifting with compressed interior action
        # amplifies roundoff near the boundary at high polynomial/Fourier order.
        rhs=(load-self.apply(coefficients))[self.free]
        original_scale=np.linalg.norm((load-self._quadrature_apply(coefficients))[self.free])
        iterations=0
        def count(_):
            nonlocal iterations
            iterations+=1
        inside,info=cg(self.operator,rhs,M=self.preconditioner,rtol=rtol,atol=atol,maxiter=maxiter,callback=count)
        coefficients[self.free]=inside
        if info:raise RuntimeError(f'CG failed: info={info}, iterations={iterations}')
        defect=(load-self._quadrature_apply(coefficients))[self.free]
        # ||P r|| is sensitive to low-mode solution error that a small raw
        # residual can hide. This estimator uses only the operator and RHS.
        def indicator(r,c):
            return float(np.linalg.norm(self.preconditioner@r)/max(np.linalg.norm(c),np.finfo(float).tiny))
        estimate=indicator(defect,coefficients);history=[estimate];refinements=0
        target=max(rtol,32*np.finfo(float).eps)
        while np.linalg.norm(defect)>atol and estimate>target and refinements<3 and iterations<maxiter:
            correction,info=cg(self.operator,defect,M=self.preconditioner,rtol=1e-2,atol=0.,
                               maxiter=maxiter-iterations,callback=count)
            if info:raise RuntimeError(f'Defect correction failed: info={info}, iterations={iterations}')
            candidate=coefficients.copy();candidate[self.free]+=correction
            next_defect=(load-self._quadrature_apply(candidate))[self.free]
            next_estimate=indicator(next_defect,candidate)
            if not np.isfinite(next_estimate) or next_estimate>=estimate:break
            coefficients,defect,estimate=candidate,next_defect,next_estimate
            refinements+=1;history.append(estimate)
        residual=np.linalg.norm(defect);scale=original_scale
        if not np.isfinite(coefficients).all() or residual>10*max(atol,rtol*scale,np.finfo(float).tiny) or (residual>atol and estimate>10*target):
            raise RuntimeError(f'CG failed: info={info}, iterations={iterations}, relative residual={residual/max(scale,1e-300):.3e}')
        return coefficients,dict(iterations=iterations,relative_residual=float(residual/max(scale,1e-300)),
            solve_seconds=time.perf_counter()-start,setup_seconds=self.setup_seconds,preconditioner=self.preconditioner_kind,
            operator_array_bytes=self.operator_array_bytes,preconditioner_nnz=self.preconditioner_nnz,
            free_dofs=len(self.free),dense_patch_matrices=0,operator_mode=self.operator_mode,
            compressed_patches=sum(terms is not None for terms in self._tensor_terms),
            metric_compression=self.metric_compression,residual_operator='original_quadrature',
            refinements=refinements,preconditioned_defect=estimate,defect_history=history)
