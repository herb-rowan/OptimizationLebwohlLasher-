import pytest
import numpy as np
from LebwohlLasher import initdat as initdat_serial, plotdat
from LebwohlLasher_mpi import initdat as initdat_mpi, one_energy, get_safe_random_seed, MC_step

def test_initdat_serial():
    nmax = 5
    arr = initdat_serial(nmax)
    assert arr.shape == (nmax, nmax), "Array shape is incorrect"
    assert np.all((arr >= 0) & (arr < 2 * np.pi)), "Array values should be within [0, 2π]"

def test_initdat_mpi():
    nmax = 5
    arr = initdat_mpi(nmax)
    assert arr.shape == (nmax, nmax), "Array shape is incorrect (MPI)"
    assert np.all((arr >= 0) & (arr < 2 * np.pi)), "Array values should be within [0, 2π] (MPI)"

def test_get_safe_random_seed():
    seed = get_safe_random_seed()
    assert isinstance(seed, int), "Seed should be an integer"
    assert 0 <= seed < 2**32, "Seed should be within 32-bit unsigned integer range"

@pytest.mark.parametrize("ix, iy, expected_energy", [
    (0, 0, 0.0),  # Adjust expected values based on known lattice configuration
])
def test_one_energy(ix, iy, expected_energy):
    nmax = 3
    arr = np.random.random((nmax, nmax)) * 2 * np.pi
    energy = one_energy(arr, ix, iy, nmax)
    assert isinstance(energy, float), "Energy should be a float"
    # Note: the expected energy value is dependent on the lattice; set realistic expectations

def test_MC_step():
    nmax = 3
    arr = np.random.random((nmax, nmax)) * 2 * np.pi
    Ts = 1.0  # Set a test temperature
    updated_arr = MC_step(arr, Ts, nmax)
    assert updated_arr.shape == (nmax, nmax), "Updated array shape is incorrect"
