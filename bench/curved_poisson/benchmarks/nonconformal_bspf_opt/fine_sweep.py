"""Finer scalar-Poisson resolution campaign; never retime an existing case."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,subprocess,importlib.util,traceback,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SRC=ROOT/'evidence/nonconformal_bspf_opt/hard_shell128';OUT=SRC/'fine_sweep_v1';OUT.mkdir(exist_ok=True)
NS=list(range(96,289,8))
NEK=sorted(set([(8,p) for p in range(12,41,2)]+[(16,p) for p in range(6,33,2)]+[(16,p) for p in range(21,32,2)]))
def read(path):return json.loads(path.read_text())
def dump(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def load_opt():
 s=importlib.util.spec_from_file_location('fine_bspf_driver',HERE/'run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.configure_hard();return m
def name(n,mode):return f'modal_n{n}_q{2*n}'+('_nodal_rhs' if mode=='nodal' else '')
def bspf_path(n,mode):
 old=SRC/name(n,mode)/'result.json';return old if old.exists() else OUT/'bspf'/name(n,mode)/'result.json'
def one_bspf(n,mode):
 opt=load_opt();opt.OUT=OUT/'bspf';opt.OUT.mkdir(exist_ok=True)
 # Reuse only geometry/basis and analytic audit caches; never solver outputs.
 for pattern in ('axis_*.npz','exact_global_q*.npz','audit_forcing_*.npy'):
  for src in SRC.glob(pattern):
   dst=opt.OUT/src.name
   if not dst.exists():dst.symlink_to(src)
 opt.configure_hard=lambda:None
 sys.argv=[str(HERE/'run.py'),'--hard','--n',str(n),'--forcing-source',mode];opt.main()
def snapshot():
 rows={mode:[] for mode in ('direct','nodal')}
 for mode in rows:
  for n in NS:
   path=bspf_path(n,mode)
   if path.exists():
    row=read(path);row['source_result']=str(path.relative_to(ROOT));row['reused']=path.parent.parent==SRC;rows[mode].append(row)
 old=read(SRC/'nektar/results.json')['rows'];new=read(OUT/'nektar/results.json')['rows'] if (OUT/'nektar/results.json').exists() else []
 rows['nektar']=[]
 for e,p in NEK:
  key=f'nektar_e{e}_p{p}_mms';row=next((r.copy() for r in old+new if r['id']==key),None)
  if row:
   row['reused']=any(r['id']==key for r in old);row['source_result']=str((SRC/'nektar/results.json' if row['reused'] else OUT/'nektar/results.json').relative_to(ROOT));rows['nektar'].append(row)
 dump(OUT/'results.json',rows)
def main():
 if len(sys.argv)>1 and sys.argv[1]=='--one-bspf':one_bspf(int(sys.argv[2]),sys.argv[3]);return
 if not (SRC/'field.npz').exists():load_opt()
 protocol=dict(bspf_n=NS,bspf_q='2*n',bspf_modes=['direct','nodal'],nektar_ep=NEK,nektar_quadrature='max(p+4,ceil(256/E)+8)',protocol='One native single-thread solve per configuration, no warmups or repeats; existing cases reused; setup and analytic sampling excluded. Fixed geometry, MMS and upstream solver kernels unchanged.',field_sha256=hashlib.sha256((SRC/'field.npz').read_bytes()).hexdigest(),status='running')
 dump(OUT/'protocol.json',protocol)
 # Alternate RHS policies at each resolution; all numerical work is sequential.
 for n in NS:
  for mode in ('direct','nodal'):
   path=bspf_path(n,mode)
   if path.exists():continue
   print(f'START BSPF n={n} {mode}',flush=True)
   proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--one-bspf',str(n),mode],capture_output=True,text=True)
   (OUT/f'bspf_n{n}_{mode}.log').write_text(proc.stdout+proc.stderr)
   if proc.returncode:
    path.parent.mkdir(parents=True,exist_ok=True);dump(path,dict(id=name(n,mode),n=n,q=2*n,forcing_source=mode,status='failed',failure=proc.stderr[-3000:]))
   row=read(path);print('DONE',row['id'],row['status'],row.get('solve_seconds'),row.get('l2_relative'),flush=True);snapshot()
 opt=load_opt();s=importlib.util.spec_from_file_location('fine_standard',ROOT/'benchmarks/standard_poisson/campaign.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c.OUT=OUT/'nektar';c.OUT.mkdir(exist_ok=True);c.MMS_QUADRATURE_BANDWIDTH=256
 for src in (SRC/'nektar').glob('exact_*.npz'):
  dst=c.OUT/src.name
  if not dst.exists():dst.symlink_to(src)
 path=c.OUT/'results.json';data=read(path) if path.exists() else dict(rows=[]);old=read(SRC/'nektar/results.json')['rows']
 for e,p in NEK:
  key=f'nektar_e{e}_p{p}_mms'
  if any(r['id']==key for r in old+data['rows']):continue
  row=dict(id=key,method='nektar',elements_per_axis=e,degree=p,guard=False,quadrature=max(p+4,int(c.np.ceil(256/e))+8),status='preparing');data['rows'].append(row);dump(path,data);folder=c.OUT/key;folder.mkdir(exist_ok=True);print('START',key,flush=True)
  try:
   cmd=c.prepare(row,folder);row['status']='attempted';dump(path,data);c.run(cmd,folder,'solve.log');row['timing']=read(folder/'timing.json');row['timing']['solve_seconds']=row['timing'].get('solve_seconds',row['timing'].get('first_solve_seconds'));assert row['timing']['solve_calls']==1;row.update(c.error(row,folder));row['status']='validated' if row['error_quadrature_stable'] else 'quadrature_unresolved'
  except Exception as ex:row.update(status='failed',failure=repr(ex));traceback.print_exc()
  dump(path,data);print('DONE',key,row['status'],row.get('timing',{}).get('solve_seconds'),row.get('l2_relative'),flush=True);snapshot()
 snapshot();protocol['status']='complete';dump(OUT/'protocol.json',protocol)
if __name__=='__main__':main()
