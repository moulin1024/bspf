"""Compare the reconstructed ITER field domain with an actual saved job mesh.

Reads only the first toroidal plane. Does not run or modify GENE-X.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import shapely
from shapely.geometry import shape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def audit(job, output):
    here=Path(__file__).resolve().parent
    domain=shape(json.loads((here/'meshes/iter/domain.geojson').read_text()))
    with h5py.File(job/'mesh.nc') as f:
        x,y=f['RZ_grid/R'][0],f['RZ_grid/Z'][0]
        g=f['multigrid/multigrid_plane001/mesh_lvl_001']
        inner=np.zeros(len(x),bool);inner[g['inner_indices'][:]-1]=True
        boundary=np.zeros(len(x),bool);boundary[g['boundary_indices'][:]-1]=True
        target=f['parcon/not_in_target'][0]>.5
        district=g['district'][:]
    active=inner&target
    inside=shapely.contains_xy(domain,x,y)
    extra=inside&~active
    distances=shapely.distance(shapely.points(x[extra],y[extra]),domain.boundary)
    spacing=float(np.median(np.diff(np.unique(x))))
    report=dict(job=str(job),mesh=str(job/'mesh.nc'),plane=0,
                equilibrium_sha256=hashlib.sha256((job/'ITER_q10.nc').read_bytes()).hexdigest(),
                params_sha256=hashlib.sha256((job/'params_in.txt').read_bytes()).hexdigest(),
                total_rows=len(x),interior_rows=int(inner.sum()),active_field_interior_rows=int(active.sum()),
                active_points_outside_partition_domain=int(np.sum(active&~inside)),
                additional_points_inside_smooth_domain=int(extra.sum()),
                additional_points_are_all_boundary_rows=bool(boundary[extra].all()),
                additional_point_district_counts={str(k):int(v) for k,v in zip(*np.unique(district[extra],return_counts=True))},
                maximum_additional_point_distance_to_boundary=float(distances.max(initial=0)),
                spacing_RZ=spacing,
                interpretation='Compare continuum domain with snapped Cartesian boundary rows, not identical discrete row sets.')
    assert report['active_points_outside_partition_domain']==0
    assert report['additional_points_are_all_boundary_rows']
    assert distances.max(initial=0)<spacing/2+1e-5
    output.mkdir(parents=True,exist_ok=True)
    (output/'iter_job_domain_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,8),constrained_layout=True)
    for ax in axes:
        ax.scatter(x[active],y[active],s=.5,c='#457b9d',rasterized=True,label='Active field interior')
        ax.scatter(x[inner&~target],y[inner&~target],s=1,c='#e9a03b',rasterized=True,label='Target / identity rows')
        ax.scatter(x[extra],y[extra],s=2,c='#d62828',rasterized=True,label='Boundary rows inside smooth domain')
        for ring in [domain.exterior,*domain.interiors]:
            a=np.asarray(ring.coords);ax.plot(a[:,0],a[:,1],c='black',lw=.8)
        ax.set_aspect('equal');ax.set_xlabel('Normalized R');ax.set_ylabel('Normalized Z (flipped)')
    axes[0].set_title('Saved ITER job vs reconstructed field boundary')
    axes[1].set_xlim(.62,1.0);axes[1].set_ylim(.48,.88)
    axes[1].set_title('Divertor detail');axes[0].legend(loc='lower left',fontsize=8)
    fig.savefig(output/'iter_job_domain_overlay.png',dpi=180)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--job',type=Path,default=Path.home()/'genex/runs/iter-q10-anchor')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'domain_audit')
    args=p.parse_args();audit(args.job,args.output)
