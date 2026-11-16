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
 def gram(q):
  x,w=np.polynomial.legendre.leggauss(q); B,D=axis.evaluate((x+1)/2,1)
  return [A[:,1:-1].T@((w/2)[:,None]*A[:,1:-1]) for A in [B,D]]
 ref=gram(2*n)
 for ratio in [1,1.1,1.2,1.25,1.3,1.4,1.5]:
  q=int(np.ceil(n*ratio)); mats=gram(q)
  errs=[float(np.max(abs(la.eigvalsh(a,b)-1))) for a,b in zip(mats,ref)]
  rows.append(dict(n=n,q=q,ratio=q/n,mass_energy_defect=errs[0],stiffness_energy_defect=errs[1]))
Path('bench/turbulence_3d/evidence/bspf_cuda3d_optimization/axis_quadrature_scan.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows: print(r['n'],r['q'],f"{r['mass_energy_defect']:.2e}",f"{r['stiffness_energy_defect']:.2e}")
