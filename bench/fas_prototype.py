import sys, time, numpy as np
sys.path.insert(0,'examples/pde')
import laval_nozzle_2d as m, bspf_cpp as b
nz=m.Nozzle(4.0,shape='sextic')
rms=lambda R: float(np.sqrt((R**2).mean()))

def weight(r, axis):
    r = np.moveaxis(r, axis, 0)
    out = r[::2].copy()
    out[1:-1] = 0.25 * r[1:-2:2] + 0.5 * r[2:-1:2] + 0.25 * r[3::2]
    out[0] = 0.5 * r[0] + 0.5 * r[1]
    out[-1] = 0.5 * r[-1] + 0.5 * r[-2]
    return np.moveaxis(out, 0, axis)
restrict = lambda r: weight(weight(r, 0), 1)
OMEGA = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
TRACK = len(sys.argv) > 8
MAXC = int(sys.argv[8]) if TRACK else 400

class Level:
    def __init__(self, n):
        self.cpu=m.make_solver(nz,*n); self.gpu=b.cuda.CompressibleSolver2D(self.cpu,refresh=10); self.n=n
    def smooth(self,U,steps):
        if steps<=0: return U
        U,_=self.gpu.solve(U,max_steps=steps,tolerance=0.0,anderson=0,check=10**9); return U
    def residual(self,U,f):
        self.gpu.set_source(f); return self.gpu.rhs(U)

def vcycle(levels,l,U,f,nu1,nu2,nuc):
    L=levels[l]; L.gpu.set_source(f)
    if l==len(levels)-1: return L.smooth(U,nuc)
    U=L.smooth(U,nu1)
    r=L.residual(U,f)
    C=levels[l+1]
    Uc0=np.ascontiguousarray(U[::2,::2]); C.gpu.set_source(None)
    fc=np.ascontiguousarray(restrict(r)-C.gpu.rhs(Uc0))
    Uc=vcycle(levels,l+1,Uc0.copy(),fc,nu1,nu2,nuc)
    U=U+OMEGA*L.cpu.prolong(C.cpu,np.ascontiguousarray(Uc-Uc0))
    L.gpu.set_source(f)
    return L.smooth(U,nu2)

fine=tuple(int(v) for v in sys.argv[1].split('x')); nlev=int(sys.argv[2]); nu1,nu2,nuc=(int(v) for v in sys.argv[3:6]); target=float(sys.argv[6])
shapes=[fine]
for _ in range(nlev-1): shapes.append(((shapes[-1][0]+1)//2,(shapes[-1][1]+1)//2))
levels=[Level(s) for s in shapes]
U=m.initial_state(nz,levels[0].cpu,False); f0=None
levels[0].gpu.set_source(None); U=levels[0].smooth(U,50)
t0=time.perf_counter(); hist=[]
work=50.0
cost=[ (s[0]*s[1])/(fine[0]*fine[1]) for s in shapes]
per_cycle=(nu1+nu2)*sum(cost[:-1])+nuc*cost[-1]
for cyc in range(1,400):
    U=vcycle(levels,0,U,None,nu1,nu2,nuc); work+=per_cycle
    r=rms(levels[0].residual(U,None)); hist.append(r)
    if TRACK and cyc%5==0:
        dH,ds=m.invariants(U); print('   cycle %d residual %.1e Cd %.6f entropy %.1e'%(cyc,r,levels[0].cpu.mass_flow(U).mean()/m.critical_mass_flow(),ds.max()),flush=True)
    if (r<target and not TRACK) or not np.isfinite(r) or cyc>=MAXC: break
dt=time.perf_counter()-t0
dH,ds=m.invariants(U)
print(fine,'levels',[f'{s[0]}x{s[1]}' for s in shapes],'nu',(nu1,nu2,nuc),': cycles',cyc,'residual %.1e'%r,'fine-step work %.0f'%work,'%.1f s'%dt,' Cd %.6f entropy %.1e'%(levels[0].cpu.mass_flow(U).mean()/m.critical_mass_flow(),ds.max()))
print('   residual per cycle:',' '.join('%.0e'%h for h in hist[::max(1,len(hist)//12)]),flush=True)
