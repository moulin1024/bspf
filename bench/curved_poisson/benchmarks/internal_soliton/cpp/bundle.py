"""Small named-matrix binary format shared with the standalone C++ solver."""
import struct
from pathlib import Path
import numpy as np


def write_bundle(path, arrays):
    with Path(path).open('wb') as f:
        f.write(b'BSPFCPP1');f.write(struct.pack('<I',len(arrays)))
        for name,value in arrays.items():
            a=np.asarray(value,dtype='<f8')
            if a.ndim==0:a=a.reshape(1,1)
            if a.ndim==1:a=a.reshape(1,-1)
            if a.ndim!=2:raise ValueError(name)
            text=name.encode();f.write(struct.pack('<I',len(text)));f.write(text)
            f.write(struct.pack('<QQ',*a.shape));f.write(a.tobytes(order='C'))


def read_bundle(path):
    result={}
    with Path(path).open('rb') as f:
        if f.read(8)!=b'BSPFCPP1':raise ValueError('Invalid bundle')
        count,=struct.unpack('<I',f.read(4))
        for _ in range(count):
            length,=struct.unpack('<I',f.read(4));name=f.read(length).decode()
            rows,cols=struct.unpack('<QQ',f.read(16))
            result[name]=np.frombuffer(f.read(rows*cols*8),dtype='<f8').reshape(rows,cols).copy()
    return result
