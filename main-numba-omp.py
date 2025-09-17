import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit, prange
import os

# Edit number of threads

os.environ["OMP_NUM_THREADS"] = "4"

# Parameters
L = 128  # Lattice size LxL
T = 1  # Temperature
J = 1.0  # Coupling constant (J > 0 for ferromagnetic interaction)
steps = 256  # Number of Monte Carlo steps
k_B = 1  # Boltzmann constant
H = 0  # External magnetic field
np.random.seed(42)  # Set seed for reproducibility

# Initialize the lattice (random spins +1 or -1)
@jit(nopython=True)
def initialise_lattice(L):
    lattice = 2 * np.random.randint(0, 2, size=(L, L)) - 1
    return lattice

# Calculate the magnetisation for a spin configuration
@jit(nopython=True)
def calc_mag(lattice):
    return np.sum(lattice)

# Sum the neighbors in the 2D lattice
@jit(nopython=True)
def sum_neighbours(lattice, i, j, L):
    return lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]

# Calculate energy for a spin configuration (parallelized with prange)
@jit(nopython=True, parallel=True)
def calc_energy(lattice, L, J, H):
    energy = 0.0
    for i in prange(L):  # Parallelized the outer loop
        for j in range(L):
            S = lattice[i, j]
            neighbours = sum_neighbours(lattice, i, j, L)
            energy += -J * S * neighbours / 2  # Each pair counted twice
    energy -= H * np.sum(lattice)
    return energy

# Monte Carlo step using Metropolis algorithm (parallelized with prange)
@jit(nopython=True, parallel=True)
def monte_carlo_step(lattice, T, L, J, k_B):
    # Instead of working on the entire lattice at once, each thread will work on a chunk of the lattice
    for _ in prange(L**2):  # Parallelize across lattice
        i, j = np.random.randint(0, L, 2)  # Randomly choose a spin
        S = lattice[i, j]
        neighbours = sum_neighbours(lattice, i, j, L)
        delta_E = 2 * J * S * neighbours
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            lattice[i, j] *= -1  # Flip the spin

# Run the simulation
lattice = initialise_lattice(L)
before_lattice = lattice.copy()  # Save the initial lattice state for plotting

energies = []
magnetizations = []
lattice_histories = []

start_time = time.time()  # Track start time for performance analysis

for step in range(steps):
    monte_carlo_step(lattice, T, L, J, k_B)
    energy = calc_energy(lattice, L, J, H)
    magnetization = calc_mag(lattice)
    energies.append(energy)
    magnetizations.append(magnetization)
    lattice_histories.append(lattice.copy())  # Save history of the lattice

end_time = time.time()  # End time for performance analysis
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Plot the results
plt.plot(energies, label="Energy")
plt.plot(magnetizations, label="Magnetization")
plt.xlabel("Monte Carlo step")
plt.legend()
plt.show()
