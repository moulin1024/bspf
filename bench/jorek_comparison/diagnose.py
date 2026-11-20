"""Diagnose the fixed-current replay by measuring weak defects and localizing the field discrepancy."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from campaign import Restart, gradient, rule, run


def analyze(ref,axis,ids,free,wall,A,rhs,native,solution,tables,order):
    defect = rhs-A@native
    interiors = ids[:,1:-1,1:-1].ravel()
    assert len(np.unique(interiors))==len(interiors)
    bubble = np.zeros_like(rhs)
    bubble[interiors] = splu(A[interiors][:,interiors].tocsc()).solve(defect[interiors])
    trace = solution-native-bubble
    bubble_energy = (bubble @ (A@bubble))/((solution-native) @ (A@(solution-native)))
    geoc,B,Ds,Dt,wc = tables(*np.polynomial.legendre.leggauss(20))
    agg = np.zeros(3)
    cell_stats = []
    bubble_norms = np.zeros(2)
    interpolation_max = 0.
    for p,g in enumerate(ids.reshape(ref.count,-1)):
        R,Z,psi,_ = ref.fields(p,geoc[0])
        a,c,ps,_ = ref.fields(p,geoc[1])
        b,d,pt,_ = ref.fields(p,geoc[2])
        det = a*d-b*c
        dr,dz = gradient(Ds,Dt,*[v[:,None] for v in (a,b,c,d,det)])
        wt = wc*abs(det)/R**2
        gr,gz = gradient(ps,pt,a,b,c,d,det)
        field = np.stack([native[g],solution[g]],axis=1)
        r,z = dr@field,dz@field
        solved_difference = wt@((r[:,1]-gr)**2+(z[:,1]-gz)**2)
        interpolation_max = max(interpolation_max,np.max(abs(B@native[g]-psi)))
        agg += [wt@(gr*gr+gz*gz),wt@((r[:,0]-gr)**2+(z[:,0]-gz)**2),
                solved_difference]
        for i,v in enumerate((bubble,trace)):
            bubble_norms[i] += wt@((dr@v[g])**2+(dz@v[g])**2)
        cell_stats.append([R.mean(),Z.mean(),solved_difference,max(abs(ref.fields(p,geoc[0])[3]))])
    cells = np.array(cell_stats)
    ranking = cells[np.argsort(-cells[:,2])]
    summary = dict(field_error_relative=float(np.sqrt(agg[2]/agg[0])),
        native_flux_interpolation_max_abs=float(interpolation_max),
        native_field_interpolation_relative=float(np.sqrt(agg[1]/agg[0])),
        bubble_fraction_energy_squared=float(bubble_energy),
        fixed_native_edges_field_effect_relative=float(np.sqrt(bubble_norms[0]/agg[0])),
        harmonic_trace_field_effect_relative=float(np.sqrt(bubble_norms[1]/agg[0])),
        fraction_squared_field_error_in_vacuum_cells=float(cells[cells[:,3]<1e-8*cells[:,3].max(),2].sum()/agg[2]),
        native_free_load_defect_relative=float(np.linalg.norm(defect[free])/np.linalg.norm(rhs[free])),
        top_one_percent_cells_fraction_squared_error=float(ranking[:int(np.ceil(.01*ref.count)),2].sum()/agg[2]),
        worst_element_mean_RZ=ranking[0,:2].tolist())
    return summary,cells


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restart',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    ref = Restart(args.restart)
    results = []
    for order in [16,24]:
        def diagnostics(*data):
            global cells
            summary,cells = analyze(*data)
            results.append(dict(assembly_order=order,**summary))
        run(ref,4,assembly_order=order,diagnose=diagnostics)
        print(results[-1],flush=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    edge = np.r_[np.linspace(0,1,8)]
    bf = rule(edge)[0]
    xyz = np.stack([ref.fields(p,bf)[:2].T.reshape(8,8,2) for p in range(ref.count)])
    polys = np.concatenate((xyz[:,:,0],xyz[:,-1,1:],xyz[:,-2::-1,-1],xyz[:,0,-2:0:-1]),axis=1)
    colors = np.log10(np.maximum(cells[:,2]/cells[:,2].sum(),1e-14))
    fig,ax = plt.subplots(figsize=(4.5,6),layout='constrained')
    art=PolyCollection(polys,array=colors,cmap='inferno',edgecolors='face',linewidths=.2)
    art.set_clim(-7,0)
    ax.add_collection(art);ax.autoscale_view();ax.set(aspect='equal',xlabel='R (m)',ylabel='Z (m)',
            title='MAST-U · field discrepancy by element')
    fig.colorbar(art,ax=ax,label='log₁₀ fraction of total squared field difference')
    fig.savefig(args.output.with_suffix('.png'),dpi=180)
    # Harmonic control on the original mesh, including its collapsed axis.
    ref.coeff[2] = ref.coeff[1].copy()
    ref.coeff[3] = 0
    control = run(ref,4)
    args.output.write_text(json.dumps(dict(tests=results,harmonic_Z_control=control),indent=2)+'\n')
