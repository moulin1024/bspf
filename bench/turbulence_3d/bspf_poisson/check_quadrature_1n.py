import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
import scipy.linalg as la
sys.path.insert(0,'bench/curved_poisson/src')
from optimized_bvp import HybridAxis
rows=[]
for n in [65,97,129]:
 axis=HybridAxis((n-9)//2,degree=8,nodes='symmetric_qr')
 for q in [n,int(np.ceil(1.5*n))]:
  x,w=np.polynomial.legendre.leggauss(q);B,D=axis.evaluate((x+1)/2,1);w=w/2
  B=B[:,1:-1];D=D[:,1:-1]
  M=B.T@(w[:,None]*B);K=D.T@(w[:,None]*D)
  sv=la.svdvals(np.sqrt(w[:,None])*D)
  row=dict(n=n,q=q,mass_condition=float(np.linalg.cond(M)),stiffness_eigenvalue_min=float(la.eigvalsh(K)[0]),stiffness_eigenvalue_max=float(la.eigvalsh(K)[-1]),weighted_derivative_singular_ratio=float(sv[-1]/sv[0]))
  try:
   lam=la.eigvalsh(K,M);row.update(generalized_min=float(lam[0]),generalized_max=float(lam[-1]))
  except la.LinAlgError as e: row['error']=str(e)
  rows.append(row)
p=Path('bench/turbulence_3d/evidence/bspf_cuda3d_optimization/quadrature_1n_diagnostics.json');p.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
