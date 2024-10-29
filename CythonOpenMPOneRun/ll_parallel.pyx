# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, exp
from libc.stdlib cimport rand as c_rand
from libc.stdlib cimport RAND_MAX
from cython.parallel cimport parallel, prange
from openmp cimport omp_get_thread_num, omp_get_max_threads, omp_set_num_threads

ctypedef np.float64_t DTYPE_t

cdef double random_uniform() nogil:
    return c_rand() / RAND_MAX

def initdat(int nmax):
    return np.random.random_sample((nmax,nmax))*2.0*np.pi

cdef double one_energy(double[:, ::1] arr, int ix, int iy, int nmax) nogil:
    cdef:
        double en = 0.0
        int ixp = (ix + 1) % nmax
        int ixm = (ix - 1) % nmax
        int iyp = (iy + 1) % nmax
        int iym = (iy - 1) % nmax
        double ang

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) * cos(ang))
    
    return en

cdef int mc_update(double[:, ::1] arr, double Ts, int ix, int iy, int nmax) nogil:
    cdef:
        double en0, en1, ang, boltz
        double scale = 0.1 + Ts
    
    ang = (random_uniform() - 0.5) * scale
    
    en0 = one_energy(arr, ix, iy, nmax)
    arr[ix,iy] += ang
    en1 = one_energy(arr, ix, iy, nmax)
    
    if en1 <= en0:
        return 1
    else:
        boltz = exp(-(en1 - en0) / Ts)
        if boltz >= random_uniform():
            return 1
        else:
            arr[ix,iy] -= ang
            return 0

def MC_step(double[:, ::1] arr, double Ts, int nmax):
    cdef:
        int i, j, tid
        int num_threads = omp_get_max_threads()
        int[::1] thread_accepted = np.zeros(num_threads, dtype=np.int32)
    
    with nogil, parallel():
        tid = omp_get_thread_num()
        for i in prange(nmax):
            for j in range(nmax):
                thread_accepted[tid] += mc_update(arr, Ts, i, j, nmax)
    
    cdef int total_accepted = 0
    for i in range(num_threads):
        total_accepted += thread_accepted[i]
    
    return total_accepted / (nmax * nmax)

def run_simulation(int n_threads):
    cdef:
        int nsteps = 100
        int nmax = 500
        double temp = 0.5
        int it
        double[:, ::1] lattice = initdat(nmax)
    
    omp_set_num_threads(n_threads)
    
    for it in range(1, nsteps+1):
        MC_step(lattice, temp, nmax)