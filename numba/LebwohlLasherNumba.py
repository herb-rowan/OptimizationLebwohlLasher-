"""
Numba-optimized version of the Lebwohl-Lasher code.
Based on the original by SH 16-Oct-23
Optimized using Numba JIT compilation
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit, prange

#=======================================================================
def initdat(nmax):
    """Initialize lattice with random orientations"""
    return np.random.random_sample((nmax,nmax))*2.0*np.pi

#=======================================================================
@jit(nopython=True)
def one_energy(arr, ix, iy, nmax):
    """Compute energy of a single cell with periodic boundaries"""
    en = 0.0
    ixp = (ix+1)%nmax
    ixm = (ix-1)%nmax
    iyp = (iy+1)%nmax
    iym = (iy-1)%nmax

    # Calculate neighbor contributions
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en

#=======================================================================
@jit(nopython=True, parallel=True)
def all_energy(arr, nmax):
    """Compute total lattice energy in parallel"""
    enall = 0.0
    for i in prange(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

#=======================================================================
@jit(nopython=True)
def get_order_tensor(arr, nmax):
    """Calculate Q tensor components"""
    Qxx = 0.0
    Qxy = 0.0
    Qyy = 0.0
    
    for i in range(nmax):
        for j in range(nmax):
            cos_theta = np.cos(arr[i,j])
            sin_theta = np.sin(arr[i,j])
            Qxx += 3*cos_theta*cos_theta - 1
            Qxy += 3*cos_theta*sin_theta
            Qyy += 3*sin_theta*sin_theta - 1
            
    norm = 2.0*nmax*nmax
    Qxx /= norm
    Qxy /= norm
    Qyy /= norm
    
    # Calculate largest eigenvalue
    trace = Qxx + Qyy
    det = Qxx*Qyy - Qxy*Qxy
    # Solve characteristic equation: lambda^2 - trace*lambda + det = 0
    discriminant = np.sqrt(trace*trace - 4*det)
    lambda1 = (trace + discriminant)/2
    lambda2 = (trace - discriminant)/2
    
    return max(lambda1, lambda2)

#=======================================================================
@jit(nopython=True)
def MC_step(arr, Ts, nmax):
    """Optimized Monte Carlo step"""
    scale = 0.1 + Ts
    accept = 0
    
    # Pre-generate random numbers
    for i in range(nmax):
        for j in range(nmax):
            ix = np.random.randint(0, nmax)
            iy = np.random.randint(0, nmax)
            ang = np.random.normal(0, scale)
            
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            
            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= np.random.random():
                    accept += 1
                else:
                    arr[ix,iy] -= ang
                    
    return accept/(nmax*nmax)

#=======================================================================
def plotdat(arr,pflag,nmax):
    """Plot lattice configuration"""
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    
    if pflag==1:
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2:
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """Save simulation data to file"""
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Numba-Output-{:s}.txt".format(current_datetime)
    with open(filename,"w") as FileOut:
        print("#=====================================================",file=FileOut)
        print("# File created:        {:s}".format(current_datetime),file=FileOut)
        print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
        print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
        print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
        print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
        print("#=====================================================",file=FileOut)
        print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
        print("#=====================================================",file=FileOut)
        for i in range(nsteps+1):
            print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)

#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """Main simulation function"""
    # Create and initialise lattice
    lattice = initdat(nmax)
    plotdat(lattice,pflag,nmax)
    
    # Initialize arrays for measurements
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    
    # Set initial values
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5
    order[0] = get_order_tensor(lattice,nmax)

    # Perform MC steps with timing
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order_tensor(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(
        program, nmax, nsteps, temp, order[nsteps-1], runtime))
    
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)

#=======================================================================
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))