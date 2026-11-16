"""Structured unit-cube Nek5000 v002 mesh, two identical boundary fields."""
import struct
from pathlib import Path
import numpy as np

def write_mesh(path, n):
    # Prenek counterclockwise vertex ordering, lower then upper z face.
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    boundaries=[]
    with Path(path).open('wb') as f:
        f.write(f'#v002{n**3:9d}{3:3d}{n**3:9d} hdr'.ljust(80).encode())
        f.write(struct.pack('<f',6.54321))
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    el=1+i+n*(j+n*k)
                    f.write(struct.pack('<d',0))
                    f.write(np.asarray(((corners+[i,j,k])/n).T,dtype='<f8').tobytes())
                    # Nek face order y-, x+, y+, x-, z-, z+.
                    for face,is_outer in enumerate([j==0,i==n-1,j==n-1,i==0,k==0,k==n-1],1):
                        if is_outer: boundaries.append((el,face))
        f.write(struct.pack('<d',0)) # curved-edge records: mapping applied in UDF
        for field in range(2):
            f.write(struct.pack('<d',len(boundaries)))
            for el,face in boundaries:
                f.write(struct.pack('<7d',el,face,0,0,0,0,1)) # bc(5)=boundary ID
                f.write(b'W       ')
