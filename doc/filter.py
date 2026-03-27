
from scipy.fftpack import dct, idct, dst, idst
import numpy as np


def apply_filter_dct(u, alpha=32.0, m=32):
    nx = len(u)
    
    # Apply DCT (type II is the standard DCT)
    u_dct = dct(u, type=2)
    
    # Create filter in frequency domain
    k = np.arange(nx)
    filter_factor = np.exp(-alpha * (k/nx)**m)
    
    # Apply filter
    u_dct_filtered = u_dct * filter_factor
    
    # Transform back using inverse DCT
    u_filtered = idct(u_dct_filtered, type=2) / (2*nx)
    
    return u_filtered