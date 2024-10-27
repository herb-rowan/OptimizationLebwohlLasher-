
"""
Fixed MPI version of the Lebwohl-Lasher code with proper random seed handling.
"""

from mpi4py import MPI
import sys
import time
import datetime
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def debug_print(msg):
    """Print debug message with rank information."""
    comm.Barrier()  # Synchronize to keep output clean
    for i in range(size):
        if rank == i:
            print(f"[Rank {rank}] {msg}")
            sys.stdout.flush()
        comm.Barrier()

def get_safe_random_seed():
    """Generate a safe random seed within numpy's limits."""
    # Use current time and rank to generate seed, but ensure it's within bounds
    t = int(time.time() * 1000)  # Get milliseconds
    seed = (t + rank) % (2**32 - 1)  # Ensure seed is within valid range
    return seed

def initdat(nmax):
    """Initialize the lattice with random orientations."""
    debug_print(f"Entering initdat with nmax={nmax}")
    
    if rank == 0:
        # Set a seed for reproducibility
        np.random.seed(get_safe_random_seed())
        arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    else:
        arr = None
        
    debug_print("Broadcasting initial lattice")
    arr = comm.bcast(arr, root=0)
    debug_print("Completed initdat")
    return arr

def one_energy(arr, ix, iy, nmax):
    """Compute energy of one cell."""
    en = 0.0
    ixp = (ix+1)%nmax
    ixm = (ix-1)%nmax
    iyp = (iy+1)%nmax
    iym = (iy-1)%nmax

    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en

def MC_step(arr, Ts, nmax):
    """Perform one Monte Carlo step in parallel."""
    debug_print(f"Starting MC_step with T={Ts}")
    
    # Calculate workload distribution
    base_chunk = nmax // size
    extra = nmax % size
    my_start = rank * base_chunk + min(rank, extra)
    my_size = base_chunk + (1 if rank < extra else 0)
    my_end = my_start + my_size

    debug_print(f"My chunk: {my_start} to {my_end}")

    # Parameters for trial moves
    scale = 0.1 + Ts
    local_accept = 0
    attempts = my_size * nmax  # Number of attempts for my chunk

    # Set up random number generation
    seed = get_safe_random_seed()
    np.random.seed(seed)
    debug_print(f"Using random seed: {seed}")

    # Pre-generate random numbers
    xran = np.random.randint(0, nmax, size=attempts)
    yran = np.random.randint(0, nmax, size=attempts)
    aran = np.random.normal(scale=scale, size=attempts)
    rands = np.random.uniform(0.0, 1.0, size=attempts)

    # Store original array for synchronization
    local_arr = arr.copy()
    attempt = 0

    debug_print(f"Starting {attempts} attempts")
    
    # Perform local Monte Carlo moves
    for i in range(my_start, my_end):
        for j in range(nmax):
            ix = xran[attempt]
            iy = yran[attempt]
            ang = aran[attempt]
            
            en0 = one_energy(local_arr, ix, iy, nmax)
            local_arr[ix,iy] += ang
            en1 = one_energy(local_arr, ix, iy, nmax)
            
            if en1 <= en0:
                local_accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= rands[attempt]:
                    local_accept += 1
                else:
                    local_arr[ix,iy] -= ang
            
            attempt += 1

    debug_print("Completed local moves")

    # Gather acceptance counts
    debug_print("Gathering acceptance counts")
    total_accept = comm.reduce(local_accept, op=MPI.SUM, root=0)
    
    # Synchronize lattice changes
    debug_print("Synchronizing lattice")
    if rank == 0:
        changes = np.zeros_like(arr)
    else:
        changes = None
        
    # Gather all changes to root
    local_changes = local_arr - arr
    comm.Reduce(local_changes, changes, op=MPI.SUM, root=0)
    
    # Broadcast updated lattice
    if rank == 0:
        arr += changes
        ratio = total_accept/(nmax*nmax)
    else:
        ratio = None
    
    debug_print("Broadcasting final results")
    arr = comm.bcast(arr, root=0)
    ratio = comm.bcast(ratio, root=0)
    
    debug_print("Completed MC_step")
    return ratio

def main(program, nsteps, nmax, temp, pflag):
    """Main simulation function."""
    debug_print(f"Starting main with nsteps={nsteps}, nmax={nmax}, temp={temp}")
    
    # Initialize lattice
    lattice = initdat(nmax)
    
    if rank == 0:
        ratio = np.zeros(nsteps+1)
        ratio[0] = 0.5
    else:
        ratio = None

    # Time the Monte Carlo steps
    initial = MPI.Wtime()
    
    for it in range(1, nsteps+1):
        debug_print(f"Starting step {it}")
        ratio_step = MC_step(lattice, temp, nmax)
        
        if rank == 0:
            ratio[it] = ratio_step
            if it % 5 == 0:  # Progress update every 5 steps
                print(f"Completed {it}/{nsteps} steps")
    
    final = MPI.Wtime()
    runtime = final - initial
    
    if rank == 0:
        print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}, "
              f"Time: {runtime:8.6f} s, Processes: {size}")

if __name__ == '__main__':
    if len(sys.argv) == 5:
        main(sys.argv[0], 
             int(sys.argv[1]),    # iterations
             int(sys.argv[2]),    # size
             float(sys.argv[3]),  # temperature
             int(sys.argv[4]))    # plot flag
    else:
        if rank == 0:
            print(f"Usage: mpiexec -n <processes> python {sys.argv[0]} "
                  "<ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
