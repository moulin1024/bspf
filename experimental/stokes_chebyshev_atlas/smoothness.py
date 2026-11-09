"""Physical line-direction compatibility across polynomial block interfaces.

Shared positions enforce C0 only. This module compares *inward* cross-interface
derivatives, with orientation reversal handled by topology. Unit directions
avoid mistaking a chart-scale change for a kink. Optional spacing matching is
separate. Endpoint tapering acknowledges extraordinary junctions, where all
coordinate lines cannot in general be continued smoothly at once.
"""
import numpy as np
from .spectral import nodes, weights


def edge_trace(values, edge):
    return (values[:, 0], values[-1, :], values[::-1, -1], values[0, ::-1])[edge]


def _add_edge(values, edge, contribution):
    if edge == 0: values[:, 0] += contribution
    elif edge == 1: values[-1, :] += contribution
    elif edge == 2: values[::-1, -1] += contribution
    else: values[0, ::-1] += contribution


class MapSmoothness:
    def __init__(self, atlas, a, b):
        self.shape = a.shape
        m = a.shape[-1]
        q = nodes(m)
        base = weights(m)*(1-q*q)**2
        uses = {}
        self.incidence = {}
        for k, quad in enumerate(atlas.seed.quads):
            for vertex in quad:
                self.incidence.setdefault(int(vertex), []).append(k)
            for edge, (u,v) in enumerate(zip(quad, np.roll(quad,-1))):
                descriptor = (k, edge, bool(u>v))
                uses.setdefault(tuple(sorted((int(u),int(v)))), []).append(descriptor)
        self.seams, self.walls = [], []
        for key, owners in uses.items():
            _, tangent = self.jets(a,b,owners[0])
            w = base*abs(tangent)
            if len(owners)==2:
                self.seams.append((key,owners,w))
            elif len(owners)==1:
                self.walls.append((key,owners[0],w))
            else:
                raise ValueError('Nonmanifold block edge')
        for entries in (self.seams,self.walls):
            total = sum(float(v[-1].sum()) for v in entries)
            if total:
                for i, item in enumerate(entries):
                    entries[i] = (*item[:-1],item[-1]/total)

    @staticmethod
    def jets(a,b,descriptor):
        k,edge,reverse = descriptor
        # Faces are CCW. Tangents follow the boundary and normals point inward.
        normal_axis,normal_sign = ((1,1),(0,-1),(1,-1),(0,1))[edge]
        tangent_axis,tangent_sign = ((0,1),(1,1),(0,-1),(1,-1))[edge]
        normal = normal_sign*edge_trace((a,b)[normal_axis][k],edge)
        tangent = tangent_sign*edge_trace((a,b)[tangent_axis][k],edge)
        if reverse:
            normal,tangent = normal[::-1],-tangent[::-1]
        return normal,tangent

    @staticmethod
    def scatter(ga,gb,descriptor,normal=None,tangent=None):
        k,edge,reverse = descriptor
        for contribution,is_tangent in ((normal,False),(tangent,True)):
            if contribution is None: continue
            if reverse:
                contribution = contribution[::-1] * (-1 if is_tangent else 1)
            axis,sign = (((0,1),(1,1),(0,-1),(1,-1)) if is_tangent
                         else ((1,1),(0,-1),(1,-1),(0,1)))[edge]
            _add_edge((ga,gb)[axis][k],edge,sign*contribution)

    def energy(self,a,b,interface_weight=0.,wall_weight=0.,spacing_weight=0.):
        ga,gb = np.zeros_like(a),np.zeros_like(b)
        energy = 0.
        for _,owners,w in self.seams:
            d1,_ = self.jets(a,b,owners[0]);d2,_ = self.jets(a,b,owners[1])
            h1,h2 = abs(d1),abs(d2)
            u,v = d1/h1,d2/h2
            residual = u+v
            energy += .5*interface_weight*float(np.sum(w*abs(residual)**2))
            g1 = interface_weight*w*(residual-u*np.real(np.conj(u)*residual))/h1
            g2 = interface_weight*w*(residual-v*np.real(np.conj(v)*residual))/h2
            log_ratio = np.log(h1/h2)
            energy += .5*spacing_weight*float(np.sum(w*log_ratio**2))
            g1 += spacing_weight*w*log_ratio*u/h1
            g2 -= spacing_weight*w*log_ratio*v/h2
            self.scatter(ga,gb,owners[0],normal=g1)
            self.scatter(ga,gb,owners[1],normal=g2)
        for _,owner,w in self.walls:
            d,t = self.jets(a,b,owner)
            h,l = abs(d),abs(t)
            u,v = d/h,t/l
            dot = np.real(np.conj(u)*v)
            energy += .5*wall_weight*float(np.sum(w*dot**2))
            self.scatter(ga,gb,owner,normal=wall_weight*w*dot*(v-dot*u)/h,
                         tangent=wall_weight*w*dot*(u-dot*v)/l)
        return energy,ga,gb

    def diagnose(self,a,b):
        angle2,spacing2,wall2 = 0.,0.,0.
        per_edge = []
        q = nodes(a.shape[-1]);trim=abs(q)<=.8
        for key,owners,w in self.seams:
            d1,_=self.jets(a,b,owners[0]);d2,_=self.jets(a,b,owners[1])
            # Angle between d1 and -d2, invariant under chart parameter scale.
            angle=np.degrees(np.abs(np.angle(-d1*np.conj(d2))))
            ratio=np.log(abs(d1)/abs(d2))
            angle2+=float(np.sum(w*angle**2));spacing2+=float(np.sum(w*ratio**2))
            per_edge.append(dict(vertices=list(key),blocks=[o[0] for o in owners],
                                 angle_rms_deg=float(np.sqrt(np.sum(w*angle**2)/np.sum(w))),
                                 trimmed_angle_max_deg=float(np.max(angle[trim])),
                                 endpoint_angles_deg=[float(angle[0]),float(angle[-1])]))
        for _,owner,w in self.walls:
            d,t=self.jets(a,b,owner)
            cosine=np.real(np.conj(d)*t)/(abs(d)*abs(t))
            deviation=np.degrees(np.arcsin(np.clip(abs(cosine),0,1)))
            wall2+=float(np.sum(w*deviation**2))
        return dict(interface_direction_rms_deg=float(np.sqrt(angle2)),
                    interface_log_spacing_rms=float(np.sqrt(spacing2)),
                    boundary_normal_deviation_rms_deg=float(np.sqrt(wall2)),
                    endpoint_taper='(1-s^2)^2; endpoint defects are reported separately',
                    interfaces=per_edge,
                    vertex_block_incidence={str(k):len(v) for k,v in self.incidence.items()})
