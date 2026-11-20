"""Uniform bicubic subdivision of the MAST-U grid for the native-element MMS study."""
from pathlib import Path
import h5py
import numpy as np
from campaign import Restart,rule

CORNERS = [(0,0),(1,0),(1,1),(0,1)]
SIGNS = np.array([[1,1,1,1],[1,-1,-1,1],[1,1,-1,-1],[1,-1,1,-1]])


def subdivide(path,output,n):
    ref = Restart(path)
    with h5py.File(path) as f:
        vx=f['vertex'][:]-1
        ix=f['index'][:]-1
        original=f['x'][:].reshape(2,4,-1)
        sizes=f['size'][:]
        tags=f['boundary'][:]
    q=np.linspace(0,1,n+1)
    coef=ref.coeff[:2].reshape(2,16,-1).transpose(2,0,1)
    pos,ds,dt,mix=[(coef@tab).reshape(-1,2,n+1,n+1) for tab in rule(q,mixed=True)]
    # Identify complete parent edges with shared flux and tangential DOFs.
    sides=[(0,3),(1,2),(0,1),(3,2)]
    edge_keys=[]
    from collections import Counter
    counts=Counter()
    for side,(a,b) in enumerate(sides):
        tangent=2 if side<2 else 1
        ends=np.stack([ix[[0,tangent]][:,vx[c]].T for c in (a,b)],axis=1)
        keys=[]
        for e in ends:
            A,B=map(tuple,e)
            key=tuple(sorted((A,B)))
            keys.append((key,A>B))
            counts[key]+=1
        edge_keys.append(keys)

    coords=[]; indices=[]; boundary=[]; connectivity=[]; scales=[]
    next_id=int(ix.max()+1)
    frames={}
    def allocate(count):
        nonlocal next_id
        ids=np.arange(next_id,next_id+count)
        next_id+=count
        return ids
    for p in range(ref.count):
        local=np.empty((n+1,n+1),int)
        factor=np.ones((4,n+1,n+1))
        for i in range(n+1):
            for j in range(n+1):
                local[i,j]=len(coords)
                if i in (0,n) and j in (0,n):
                    c=CORNERS.index((i//n,j//n))
                    v=vx[c,p]
                    coords.append(original[:,:,v]);indices.append(ix[:,v]);boundary.append(tags[v])
                    factor[:,i,j]=sizes[:,c,p]*SIGNS[:,c]/[1,n,n,n*n]
                    continue
                geom=np.column_stack([pos[p,:,i,j],ds[p,:,i,j]/(3*n),
                                      dt[p,:,i,j]/(3*n),mix[p,:,i,j]/(9*n*n)])
                tag=0
                if i not in (0,n) and j not in (0,n):
                    ids=allocate(4)
                else:
                    side=(int(i==n) if i in (0,n) else 2+int(j==n))
                    tangent=2 if side<2 else 1
                    transverse=3-tangent
                    (edge,reverse)=edge_keys[side][p]
                    k=j if side<2 else i
                    key=(edge,n-k if reverse else k)
                    a,b=sides[side]
                    collapsed=ix[0,vx[a,p]]==ix[0,vx[b,p]]
                    if collapsed:
                        ids=allocate(4)
                        ids[0]=ix[0,vx[a,p]]
                        factor[tangent,i,j]=0.
                    else:
                        if key not in frames:
                            frames[key]=(allocate(4),geom[:,[tangent,transverse]].copy())
                        shared,directions=frames[key]
                        ids=np.empty(4,int)
                        ids[[0,tangent,transverse,3]]=shared
                        # Use one Hermite tangent frame at each new interface node.
                        # Keep each incident element's directional scale and orientation.
                        for kdir,direction in zip([tangent,transverse],directions.T):
                            vector=geom[:,kdir]
                            ratio=np.linalg.norm(vector)/np.linalg.norm(direction)
                            factor[kdir,i,j]=np.copysign(ratio,vector@direction)
                            geom[:,kdir]=direction
                        factor[3,i,j]=factor[1,i,j]*factor[2,i,j]
                        geom[:,3]/=factor[3,i,j]
                        if counts[edge]==1:
                            tag=tangent
                coords.append(geom);indices.append(ids);boundary.append(tag)
        for i in range(n):
            for j in range(n):
                locations=[(i+a,j+b) for a,b in CORNERS]
                connectivity.append([local[k] for k in locations])
                scales.append(np.column_stack([factor[:,a,b] for a,b in locations])*SIGNS)
    with h5py.File(output,'w') as f:
        f['x']=np.stack(coords,axis=-1)[:,:,None,:]
        f['index']=np.array(indices).T+1
        f['boundary']=boundary
        f['vertex']=np.array(connectivity).T+1
        f['size']=np.stack(scales,axis=-1)
        f.create_dataset('values',shape=(3,4,1,len(coords)),dtype='f8',fillvalue=0.)
