# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport cos

def one_energy_cython(double[:, ::1] arr, int ix, int iy, int nmax):
    """Cythonized energy calculation for a single lattice site"""
    cdef:
        double en = 0.0
        int ixp = (ix + 1) % nmax
        int ixm = (ix - 1) % nmax
        int iyp = (iy + 1) % nmax
        int iym = (iy - 1) % nmax
        double ang
    
    # Calculate energy contributions from neighbors
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    return en