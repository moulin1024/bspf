"""Stock-library Poisson comparison. One timed solve per configuration."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['OMPI_MCA_btl']='self';os.environ['OMPI_MCA_pml']='ob1'
import sys,json,time,subprocess,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'evidence/standard_poisson/v1';OUT.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'benchmarks/curved_quad_poisson'))
import run_turbulence_only as case
r=case.r
import numpy as np
from scipy.special import roots_legendre,eval_jacobi
from xml.etree import ElementTree as ET

MMS_QUADRATURE_BANDWIDTH=128

def dump(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def run(cmd,folder,name,env=None):
 result=subprocess.run(list(map(str,cmd)),cwd=folder,capture_output=True,text=True,timeout=240,env=env)
 (folder/name).write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError(f'exit {result.returncode}: {result.stdout[-1500:]} {result.stderr[-2500:]}')
def xml(folder,e,p,q):
 root=ET.Element('NEKTAR');geom=ET.SubElement(root,'GEOMETRY',DIM='2',SPACE='2');verts=ET.SubElement(geom,'VERTEX');edges=ET.SubElement(geom,'EDGE');els=ET.SubElement(geom,'ELEMENT');curves=ET.SubElement(geom,'CURVED');edge_ids={};boundary=[]
 for j in range(e+1):
  for i in range(e+1):
   v=r.PATCH.jet(i/e,j/e)[0];ET.SubElement(verts,'V',ID=str(j*(e+1)+i)).text=f'{v[0]:.17g} {v[1]:.17g} 0'
 def edge(a,b):
  key=tuple(sorted((a,b)))
  if key in edge_ids:return edge_ids[key]
  eid=len(edge_ids);edge_ids[key]=eid;ET.SubElement(edges,'E',ID=str(eid)).text=f'{a} {b}'
  ia,ja=a%(e+1),a//(e+1);ib,jb=b%(e+1),b//(e+1)
  if ia==ib and ia in (0,e) or ja==jb and ja in (0,e):boundary.append(eid)
  points=[]
  for z in np.linspace(0,1,4):
   xy=r.PATCH.jet(((1-z)*ia+z*ib)/e,((1-z)*ja+z*jb)/e)[0];points.append(f'{xy[0]:.17g} {xy[1]:.17g} 0')
  ET.SubElement(curves,'E',ID=str(eid),EDGEID=str(eid),NUMPOINTS='4',TYPE='PolyEvenlySpaced').text=' '.join(points)
  return eid
 for j in range(e):
  for i in range(e):
   a=j*(e+1)+i;b=a+1;d=a+e+1;c=d+1;ids=[edge(a,b),edge(b,c),edge(c,d),edge(d,a)];ET.SubElement(els,'Q',ID=str(j*e+i)).text=' '.join(map(str,ids))
 comps=ET.SubElement(geom,'COMPOSITE');ET.SubElement(comps,'C',ID='0').text=f'Q[0-{e*e-1}]';ET.SubElement(comps,'C',ID='1').text='E['+','.join(map(str,boundary))+']';ET.SubElement(geom,'DOMAIN').text='C[0]'
 ex=ET.SubElement(root,'EXPANSIONS');ET.SubElement(ex,'E',COMPOSITE='C[0]',BASISTYPE='Modified_A,Modified_A',NUMMODES=f'{p+1},{p+1}',POINTSTYPE='GaussLobattoLegendre,GaussLobattoLegendre',NUMPOINTS=f'{q},{q}',FIELDS='u')
 co=ET.SubElement(root,'CONDITIONS');si=ET.SubElement(co,'SOLVERINFO')
 for k,v in [('EQTYPE','Poisson'),('Projection','Continuous'),('GlobalSysSoln','DirectStaticCond')]:ET.SubElement(si,'I',PROPERTY=k,VALUE=v)
 ET.SubElement(ET.SubElement(co,'PARAMETERS'),'P').text='Lambda = 0';ET.SubElement(ET.SubElement(co,'VARIABLES'),'V',ID='0').text='u';ET.SubElement(ET.SubElement(co,'BOUNDARYREGIONS'),'B',ID='0').text='C[1]';ET.SubElement(ET.SubElement(ET.SubElement(co,'BOUNDARYCONDITIONS'),'REGION',REF='0'),'D',VAR='u',VALUE='0')
 for name in ('Forcing','ExactSolution'):ET.SubElement(ET.SubElement(co,'FUNCTION',NAME=name),'E',VAR='u',VALUE='0')
 ET.ElementTree(root).write(folder/'case.xml',encoding='utf-8',xml_declaration=True)
 # Documented collection implementation; upstream kernels, no custom assembly.
 (folder/'case.opt').write_text('<NEKTAR><COLLECTIONS DEFAULT="SumFac"/></NEKTAR>')
def vec(bundle,key,n):return bundle[key].ravel()[:n]
def packed(v):
 v=np.asarray(v).ravel();return v.reshape(1,-1) if len(v)<=100000 else np.pad(v,(0,(-len(v))%1024)).reshape(-1,1024)
def prepare(row,folder):
 start=time.perf_counter();e,p,q=row['elements_per_axis'],row['degree'],row['quadrature'];method=row['method'];guard=row['guard']
 if method=='nektar':
  xml(folder,e,p,q);cmd=[HERE/'build/nektar_reference','--no-exp-opt',folder/'case.xml'];run(cmd,folder,'export.log',dict(os.environ,BSPF_EXPORT_COORDINATES='1'));c=r.read_bundle(folder/'coordinates.bin');reference=np.loadtxt(folder/'reference_coordinates.txt');eid=reference[:,0].astype(int);s=(eid%e+reference[:,1])/e;t=(eid//e+reference[:,2])/e;xy=r.PATCH.jet(s,t)[0];actual=np.c_[c['x'].ravel(),c['y'].ravel()];ge=float(np.max(abs(xy-actual)));assert ge<2e-12,ge
  u,_,_,f=r.truth(actual,guard);samples=dict(c,exact=u.reshape(c['x'].shape),forcing=-f.reshape(c['x'].shape))
  for key in c:
   if key.startswith('bx'):
    k=key[2:];samples['boundary'+k]=r.truth(np.c_[c[key].ravel(),c['by'+k].ravel()],guard)[0].reshape(c[key].shape)
  r.write_bundle(folder/'nektar_samples.bin',samples)
 else:
  base=[HERE/'build/mfem_reference',e,p,q];run(base+['export'],folder,'export.log');c=r.read_bundle(folder/'coordinates.bin');n=int(c['nload'].item());nd=int(c['ndofs'].item());xy=r.PATCH.jet(vec(c,'s',n),vec(c,'t',n))[0];actual=np.c_[vec(c,'x',n),vec(c,'y',n)];ge=float(np.max(abs(xy-actual)));assert ge<2e-12,ge
  samples=dict(c,forcing=packed(r.truth(actual,guard)[3]),boundary=packed(r.truth(np.c_[vec(c,'bx',nd),vec(c,'by',nd)],guard)[0]));r.write_bundle(folder/'samples.bin',samples);cmd=base+['solve']
 row.update(preprocessing_seconds=time.perf_counter()-start,geometry_max_error=ge);return cmd

def basis(p,z):return np.column_stack([1-z,z]+[z*(1-z)*eval_jacobi(k-2,1,1,2*z-1) for k in range(2,p+1)])
def error(row,folder):
 values=[];e,p=row['elements_per_axis'],row['degree']
 if row['method']=='nektar':
  cs={}
  for line in (folder/'coefficients.txt').read_text().splitlines():
   parts=line.split();eid,n=map(int,parts[:2]);assert n==(p+1)**2;cs[eid]=np.array(parts[2:],float).reshape(p+1,p+1).T
  for qa in (max(p+6,int(np.ceil(MMS_QUADRATURE_BANDWIDTH/e))+10),max(p+6,int(np.ceil(MMS_QUADRATURE_BANDWIDTH/e))+10)+8):
   z,w=roots_legendre(qa);z=(z+1)/2;w=w/2;B=basis(p,z);num=den=0.
   cache=OUT/f'exact_e{e}_q{qa}_{"guard" if row["guard"] else "turbulence"}.npz'
   if cache.exists():c=np.load(cache);exact,weight=c['exact'],c['weight']
   else:
    exact=[];weight=[]
    for eid in range(e*e):
     xy,_,_,j,*_=r.PATCH.jet((eid%e+z[:,None])/e,(eid//e+z[None,:])/e);exact.append(r.truth(xy,row['guard'])[0].reshape(qa,qa));weight.append(np.outer(w,w)*j/e**2)
    exact=np.array(exact);weight=np.array(weight);np.savez(cache,exact=exact,weight=weight)
   for eid,c in cs.items():num+=np.sum(weight[eid]*(B@c@B.T-exact[eid])**2);den+=np.sum(weight[eid]*exact[eid]**2)
   values.append(float(np.sqrt(num/den)))
 else:
  for k in range(2):
   b=r.read_bundle(folder/f'audit{k}.bin');n=int(b['count'].item());exact=r.truth(np.c_[vec(b,'x',n),vec(b,'y',n)],row['guard'])[0];w=vec(b,'weights',n);u=vec(b,'u',n);values.append(float(np.sqrt(np.sum(w*(u-exact)**2)/np.sum(w*exact**2))))
 return dict(l2_relative=values[-1],error_quadrature_values=values,error_quadrature_stable=abs(values[0]-values[1])<.01*values[-1]+1e-11)

def main():
 path=OUT/'results.json';data=json.loads(path.read_text()) if path.exists() else dict(protocol='One native solve per configuration, no PDE warmup or repetitions. Sampled RHS to solution, setup/factorization and analytic data evaluation excluded. Stock solver libraries, no custom competitor kernels.',rows=[],versions=dict(nektar='5.10.0',mfem='4.9',hypre='3.1.0'),geometry='Same four-cubic-spline Coons patch; exact Q3 geometry',mms='Saved seed20261115 synthetic turbulence only; exact nonzero Dirichlet data',sweep=dict(elements_per_axis=[4,8,16],degrees=[4,8,12,16],additional=[[8,24]]))
 data['status']='running'
 jobs=[(m,2,6,True) for m in ('nektar','mfem')]+[('mfem',2,16,True),('mfem',4,8,True)]+[(m,e,p,False) for e,p in [(e,p) for e in (4,8,16) for p in (4,8,12,16)]+[(8,24)] for m in ('nektar','mfem')]+[('nektar',8,32,False),('nektar',16,24,False),('mfem',24,16,False),('mfem',32,16,False)]
 data['accuracy_extension']={'reason':'Initial ladder did not cover tight error targets; extend supported h/p configurations without changing upstream kernels','nektar':[[8,32],[16,24]],'mfem':[[24,16],[32,16]]}
 for method,e,p,guard in jobs:
  key=f'{method}_e{e}_p{p}_{"guard" if guard else "mms"}'
  old=next((v for v in data['rows'] if v['id']==key),None)
  if old:
   if old['status']=='failed' and method=='mfem' and p<=16:
    data.setdefault('qualification_adapter_failures',[]).append(old);data['rows'].remove(old)
    archive=OUT/(key+'_adapter_failure_'+str(len(data['qualification_adapter_failures'])));original=OUT/key
    if original.exists() and not archive.exists():original.rename(archive)
   else:continue
  if method=='mfem' and p>16:
   if not old:
    data['rows'].append(dict(id=key,method=method,degree=p,elements_per_axis=e,guard=guard,status='unsupported',reason='Installed CPU PA kernel D1D/Q1D limit; no upstream kernel modifications'));dump(path,data)
   continue
  if not guard:assert all(v['status']=='validated' for v in data['rows'] if v['guard']), 'Qualification failed; do not time the campaign'
  row=dict(id=key,method=method,elements_per_axis=e,degree=p,guard=guard,quadrature=max(p+4,int(np.ceil(MMS_QUADRATURE_BANDWIDTH/e))+8) if not guard else (40 if method=='mfem' and p==8 else 16),status='preparing');row['library_configuration']=('DirectStaticCond; SumFac collections' if method=='nektar' else ('PA+batched LOR/BoomerAMG' if p<=8 else 'PA+public LegacyAssembleSystem LOR/BoomerAMG'));data['rows'].append(row);dump(path,data);folder=OUT/key;folder.mkdir(exist_ok=True)
  try:
   cmd=prepare(row,folder);row['status']='attempted';dump(path,data);run(cmd,folder,'solve.log');timing=json.loads((folder/'timing.json').read_text());timing['solve_seconds']=timing.get('solve_seconds',timing.get('first_solve_seconds'));assert timing['solve_calls']==1;row['timing']=timing;row.update(error(row,folder));row['status']='validated' if row['error_quadrature_stable'] else 'quadrature_unresolved'
   if guard and row['l2_relative']>1e-9:row['status']='guard_failed'
  except Exception as ex:row.update(status='failed',failure=repr(ex));traceback.print_exc()
  dump(path,data);print(key,row['status'],row.get('timing',{}).get('solve_seconds'),row.get('l2_relative'),flush=True)
 data['status']='complete';dump(path,data)
if __name__=='__main__':main()
