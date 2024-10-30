
"""
Simple test script for the MPI Lebwohl-Lasher implementation.
"""

import subprocess
import sys
import time

def run_test(processes, size=20, steps=10):
    """Run a single test with specified parameters."""
    cmd = [
        'mpiexec',
        '-n', str(processes),
        sys.executable,
        'LebwohlLasher_mpi.py',
        str(steps),
        str(size),
        '0.5',  # temperature
        '0'     # plot flag
    ]
    
    print(f"\nRunning test with {processes} processes...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        end_time = time.time()
        
        print("\nProcess output:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
            
        print(f"\nTotal wall time: {end_time - start_time:.2f} seconds")
        print("Test completed successfully")
        
    except subprocess.TimeoutExpired:
        print("Error: Test timed out after 60 seconds")
    except Exception as e:
        print(f"Error running test: {e}")

def main():
    """Run basic tests."""
    print("Starting basic MPI tests...")
    
    # Test with 1 process first
    print("\n=== Testing with 1 process ===")
    run_test(1)
    
    # Test with 2 processes
    print("\n=== Testing with 2 processes ===")
    run_test(2)

if __name__ == "__main__":
    main()