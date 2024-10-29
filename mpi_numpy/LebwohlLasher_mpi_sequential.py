
"""
MPI-parallelized and NumPy-vectorized version of the Lebwohl-Lasher code.
Combines distributed memory parallelism with vectorized operations.
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
    comm.Barrier()
    for i in range(size):
        if rank == i:
            print(f"[Rank {rank}] {msg}")
            sys.stdout.flush()
        comm.Barrier()

def initdat(nmax):
    """Initialize the lattice with random orientations."""
    debug_print(f"Entering initdat with nmax={nmax}")
    if rank == 0:
        arr = np.linspace(0, 2.0*np.pi, nmax*nmax).reshape((nmax, nmax))
    else:
        arr = None
    arr = comm.bcast(arr, root=0)
    debug_print("Completed initdat")
    return arr

def compute_energy_vectorized(arr, ix_range, nmax):
    """Compute energy for a range of indices using vectorized operations."""
    # Create arrays of neighboring indices
    ixp = (ix_range + 1) % nmax
    ixm = (ix_range - 1) % nmax
    
    # Create views for all neighbors at once
    angles_right = arr[ix_range, :] - arr[ixp, :]
    angles_left = arr[ix_range, :] - arr[ixm, :]
    angles_up = arr[ix_range, :] - np.roll(arr[ix_range, :], -1, axis=1)
    angles_down = arr[ix_range, :] - np.roll(arr[ix_range, :], 1, axis=1)
    
    # Compute energy contributions vectorized
    energy = 0.5 * (1.0 - 3.0 * np.cos(angles_right)**2) + \
             0.5 * (1.0 - 3.0 * np.cos(angles_left)**2) + \
             0.5 * (1.0 - 3.0 * np.cos(angles_up)**2) + \
             0.5 * (1.0 - 3.0 * np.cos(angles_down)**2)
    
    return energy.sum()

def MC_step_vectorized(arr, Ts, nmax):
    """Perform one Monte Carlo step with vectorized operations."""
    debug_print(f"Starting vectorized MC_step with T={Ts}")
    
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
    
    # Generate all random numbers at once
    np.random.seed(int(time.time() * 1000 + rank) % (2**32 - 1))
    n_attempts = my_size * nmax
    xran = np.random.randint(my_start, my_end, size=n_attempts)
    yran = np.random.randint(0, nmax, size=n_attempts)
    ang_changes = np.random.normal(scale=scale, size=n_attempts)
    rand_uniform = np.random.uniform(0.0, 1.0, size=n_attempts)
    
    # Store original array for synchronization
    local_arr = arr.copy()
    
    # Perform vectorized Monte Carlo moves
    for i in range(n_attempts):
        ix, iy = xran[i], yran[i]
        ang = ang_changes[i]
        
        # Calculate initial energy for the cell and its neighbors
        en0 = compute_energy_vectorized(local_arr, np.array([ix]), nmax)
        
        # Make trial move
        local_arr[ix, iy] += ang
        
        # Calculate new energy
        en1 = compute_energy_vectorized(local_arr, np.array([ix]), nmax)
        
        # Accept or reject move
        if en1 <= en0:
            local_accept += 1
        else:
            boltz = np.exp(-(en1 - en0) / Ts)
            if boltz >= rand_uniform[i]:
                local_accept += 1
            else:
                local_arr[ix, iy] -= ang
    
    # Gather acceptance counts and sync lattice
    total_accept = comm.reduce(local_accept, op=MPI.SUM, root=0)
    
    # Synchronize lattice changes
    changes = np.zeros_like(arr) if rank == 0 else None
    local_changes = local_arr - arr
    comm.Reduce(local_changes, changes, op=MPI.SUM, root=0)
    
    if rank == 0:
        arr += changes
        ratio = total_accept/(nmax*nmax)
    else:
        ratio = None
    
    arr = comm.bcast(arr, root=0)
    ratio = comm.bcast(ratio, root=0)
    
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
        ratio_step = MC_step_vectorized(lattice, temp, nmax)
        
        if rank == 0:
            ratio[it] = ratio_step
            if it % 5 == 0:
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
