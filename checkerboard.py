from numba import jit, prange
import numpy as np
import numba
import time 

cores = 28
numba.set_num_threads(cores)

# Parameters
L = 512  # Lattice size LxL
T = 1  # Temperature
J = 1.0  # Coupling constant (J > 0 for ferromagnetic interaction)
steps = 512  # Number of Monte Carlo steps
k_B = 1  # Boltzmann constant
H = 0  # External magnetic field
np.random.seed(42)  # Set seed for reproducibility

@jit(nopython=True)
def initialise_lattice(L):
    lattice = 2 * np.random.randint(0, 2, size=(L, L)) - 1
    return lattice

@jit(nopython=True, parallel=True)
def monte_carlo_step_checkerboard(lattice, T, J=1.0, k_B=1.0):
    L = lattice.shape[0]
    updated_lattice = lattice.copy()  # Temporary array to store updates

    # Update black spin
    for i in prange(L):
        for j in range(L):
            if (i + j) % 2 == 0:  # Black spins
                S = lattice[i, j]
                neighbours = (lattice[(i+1) % L, j] + lattice[(i-1) % L, j] +
                              lattice[i, (j+1) % L] + lattice[i, (j-1) % L])
                delta_E = 2 * J * S * neighbours
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / (k_B * T)):
                    updated_lattice[i, j] = -S
                else:
                    updated_lattice[i, j] = S

    lattice[:] = updated_lattice  # Apply updates for black spins

    # Update white spins
    for i in prange(L):
        for j in range(L):
            if (i + j) % 2 == 1:  # White spins
                S = lattice[i, j]
                neighbours = (lattice[(i+1) % L, j] + lattice[(i-1) % L, j] +
                              lattice[i, (j+1) % L] + lattice[i, (j-1) % L])
                delta_E = 2 * J * S * neighbours
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / (k_B * T)):
                    updated_lattice[i, j] = -S
                else:
                    updated_lattice[i, j] = S

    lattice[:] = updated_lattice  # Apply updates for white spins

start = time.time()

lattice = initialise_lattice(L)


for i in range(steps):
    monte_carlo_step_checkerboard(lattice, T)

print(cores, round((time.time() - start), 2))