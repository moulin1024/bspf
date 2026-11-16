"""Plot the measured incremental benefit of sharing tensor contractions."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/bspf-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[1]/'evidence/bspf_cuda3d_shared'
rows=json.loads((out/'optimization.json').read_text())
with (out/'optimization.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(10,4.8))
for key,label in [('baseline_cg_s','Independent gradients'),('warm_cg_s','Shared gradients + transpose')]:
    axes[0].loglog([r['n'] for r in rows],[r[key] for r in rows],'o-',label=label)
axes[0].set(xlabel='Nodes per axis',ylabel='Warm CG time (s)');axes[0].legend()
axes[1].plot([r['n'] for r in rows],[r['speedup'] for r in rows],'o-')
axes[1].set(xlabel='Nodes per axis',ylabel='Incremental speedup');axes[1].axhline(1,color='gray',linestyle='--')
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('BSPF degree 8: shared contractions on RTX 3070')
fig.text(.5,.02,'Both paths use FP64 reflection splitting and FP32 preconditioning. Median of six warm solves per mode.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.94])
for ext in ['png','pdf']:fig.savefig(out/f'optimization.{ext}',dpi=180)
print(out/'optimization.png')
