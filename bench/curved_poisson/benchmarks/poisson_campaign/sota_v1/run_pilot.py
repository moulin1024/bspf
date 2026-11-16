"""Native bundle assembly shared with the scalar Poisson campaign."""
import numpy as np

def arrays(engine,forcing,boundary):
    g=np.zeros(engine.n**2);g[engine.boundary]=boundary
    data=dict(B=engine.basis,D=engine.derivative,ss=engine.hss[0],st=engine.hst[0],tt=engine.htt[0],
              V=engine._tensor_vectors,inverse=engine._tensor_inverse,weights=engine.weights[0],
              forcing=np.asarray(forcing).reshape(engine.weights[0].shape),boundary=g.reshape(engine.n,engine.n),terms=np.array([[0.]]))
    terms=engine._tensor_terms[0]
    if terms is not None:
        data['terms']=np.array([[len(terms[0])]])
        for i,(l,r) in enumerate(zip(*terms)):data[f'left{i}']=l;data[f'right{i}']=r
    return data
