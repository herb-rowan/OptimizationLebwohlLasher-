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

# Define C types for better performance
ctypedef np.float64_t DTYPE_t

cdef double random_uniform() nogil:
    return c_rand() / RAND_MAX

def initdat(int nmax):
    """Initialize the lattice with random orientations"""
    return np.random.random_sample((nmax,nmax))*2.0*np.pi

def one_energy(double[:, ::1] arr, int ix, int iy, int nmax):
    """Optimized energy calculation for a single lattice site"""
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

def all_energy(double[:, ::1] arr, int nmax):
    """Calculate total energy of the lattice"""
    cdef:
        double enall = 0.0
        int i, j
    
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall

def get_order(double[:, ::1] arr, int nmax):
    """Calculate order parameter"""
    cdef:
        int a, b, i, j
        double[:, ::1] Qab = np.zeros((3,3))
        double[:, ::1] delta = np.eye(3)
        
    # Create unit vectors
    cdef double[:, :, ::1] lab = np.zeros((3, nmax, nmax))
    for i in range(nmax):
        for j in range(nmax):
            lab[0,i,j] = cos(arr[i,j])
            lab[1,i,j] = sin(arr[i,j])
            
    # Calculate Q tensor
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3.0 * lab[a,i,j] * lab[b,i,j] - delta[a,b]
    
    # Normalize
    cdef double norm_factor = 2.0 * nmax * nmax
    for a in range(3):
        for b in range(3):
            Qab[a,b] = Qab[a,b] / norm_factor
            
    # Convert to numpy array for eigenvalue calculation
    eigenvalues = np.linalg.eigvals(np.asarray(Qab))
    return np.max(eigenvalues.real)

# ... [previous imports and other functions remain the same until MC_step] ...

def MC_step(double[:, ::1] arr, double Ts, int nmax):
    """Perform one Monte Carlo step"""
    cdef:
        int i, j, ix, iy, accept = 0
        double en0, en1, ang, boltz
        double scale = 0.1 + Ts
        
    # Pre-compute random numbers - note the conversion to double
    cdef double[:, ::1] xran = np.random.randint(0, high=nmax, size=(nmax, nmax)).astype(np.float64)
    cdef double[:, ::1] yran = np.random.randint(0, high=nmax, size=(nmax, nmax)).astype(np.float64)
    cdef double[:, ::1] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    
    for i in range(nmax):
        for j in range(nmax):
            ix = int(xran[i,j])
            iy = int(yran[i,j])
            ang = aran[i,j]
            
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            
            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= random_uniform():
                    accept += 1
                else:
                    arr[ix,iy] -= ang
                    
    return accept / (nmax * nmax)

def main(str program, int nsteps, int nmax, double temp, int pflag):
    """Main simulation function"""
    # Initialize arrays
    cdef:
        double[:, ::1] lattice = initdat(nmax)
        double[:] energy = np.zeros(nsteps+1)
        double[:] ratio = np.zeros(nsteps+1)
        double[:] order = np.zeros(nsteps+1)
        int it
        
    # Initial values
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)
    
    # Main loop
    for it in range(1, nsteps+1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
        
    return np.asarray(lattice), np.asarray(energy), np.asarray(ratio), np.asarray(order)