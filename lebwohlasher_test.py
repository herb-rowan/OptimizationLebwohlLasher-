import pytest
import numpy as np
from LebwohlLasher import initdat, plotdat

def test_initdat():
    nmax = 5
    lattice = initdat(nmax)
    # Check if the lattice has the correct shape
    assert lattice.shape == (nmax, nmax)
    # Check if all values are within the range [0, 2pi]
    assert np.all(lattice >= 0) and np.all(lattice <= 2 * np.pi)

@pytest.mark.parametrize("pflag, expected_shape", [
    (0, ()),  # No plot, expecting no output
    (1, (5, 5)),  # Example lattice size for energy plot
    (2, (5, 5)),  # Example lattice size for angle plot
])
def test_plotdat(pflag, expected_shape):
    nmax = 5
    lattice = initdat(nmax)
    if pflag != 0:
        plotdat(lattice, pflag, nmax)
        # Here, we're not testing the plot output, but checking that plotdat runs without error
        assert True
    else:
        assert plotdat(lattice, pflag, nmax) is None  # No plot expected for pflag = 0
