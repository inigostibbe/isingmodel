import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit, prange
import numba

numba.set_num_threads(12)  # Set the number of threads

# Parameters
L = 128  # Lattice size LxL
T = 1  # Temperature
J = 1.0  # Coupling constant (J > 0 for ferromagnetic interaction)
steps = 2048  # Number of Monte Carlo steps
k_B = 1  # Boltzmann constant
H = 0  # External magnetic field
np.random.seed(42)  # Set seed for reproducibility

# Initialize the lattice (random spins +1 or -1)
@jit(nopython=True)
def initialise_lattice(L):
    lattice = 2 * np.random.randint(0, 2, size=(L, L)) - 1
    return lattice

# Sum the neighbors in the 2D lattice
@jit(nopython=True)
def sum_neighbours(lattice, i, j, L):
    return lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]

# Monte Carlo step using Metropolis algorithm (parallelized by dividing lattice)
@jit(nopython=True, parallel=True)
def monte_carlo_step(lattice, T, L, J, k_B):
    for _ in prange(L**2 // 1):  # Divide iterations by 4 since we work on 4 rectangles
        for rectangle in range(1):  # Process each rectangle in parallel
            # Calculate j_offset for each of the 4 regions (L x L/4 each)
            i = np.random.randint(0, L)  # Full height
            j_offset = rectangle * (L // 1)
            j = np.random.randint(0, L // 1) + j_offset  # Random within the L/4 width block
            
            S = lattice[i, j]
            neighbours = sum_neighbours(lattice, i, j, L)
            delta_E = 2 * J * S * neighbours
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
                lattice[i, j] *= -1  # Flip the spin

# @jit(nopython=True)
# def monte_carlo_step(lattice, T, L, J, k_B):
#     for _ in range(L**2 // 2):  # Divide iterations by 4 since we work on 4 rectangles
#         for rectangle in range(2):  # Process each rectangle in parallel
#             # Calculate j_offset for each of the 4 regions (L x L/4 each)
#             i = np.random.randint(0, L)  # Full height
#             j_offset = rectangle * (L // 2)
#             j = np.random.randint(0, L // 2) + j_offset  # Random within the L/4 width block
            
#             S = lattice[i, j]
#             neighbours = sum_neighbours(lattice, i, j, L)
#             delta_E = 2 * J * S * neighbours
#             if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
#                 lattice[i, j] *= -1  # Flip the spin

# Calculate energy for a spin configuration (parallelized)
@jit(nopython=True, parallel=True)
def calc_energy(lattice, L, J, H):
    energy = 0.0
    for i in prange(L):  # Parallelize the outer loop with prange
        for j in range(L):
            S = lattice[i, j]
            neighbours = sum_neighbours(lattice, i, j, L)
            energy += -J * S * neighbours / 2  # Each pair counted twice
    energy -= H * np.sum(lattice)
    return energy

# Run the simulation
lattice = initialise_lattice(L)
energies = []
magnetizations = []
lattice_histories = []

start_time = time.time()

for step in range(steps):
    monte_carlo_step(lattice, T, L, J, k_B)
    energy = calc_energy(lattice, L, J, H)
    magnetization = np.sum(lattice)/(L*L)
    energies.append(energy)
    magnetizations.append(magnetization)
    lattice_histories.append(lattice.copy())
    # print(step)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Lattice points: {L*L}, Steps: {steps}, Elapsed time: {elapsed_time:.2f} seconds")


def plot_lattice(lattice, title):
    plt.imshow(lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)  
    plt.title(title)
    # plt.colorbar(label='Spin')

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

# Plot the "before" state
plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)  # Create subplot for the "before" state
plot_lattice(lattice_histories[0], 'Before')

# Plot the "after" state
plt.subplot(1, 4, 2)  # Create subplot for the "after" state
plot_lattice(lattice_histories[128], '10% of steps')

# Plot the "after" state
plt.subplot(1, 4, 3)  # Create subplot for the "after" state
plot_lattice(lattice_histories[512], '50% of steps')

# Plot the "after" state
plt.subplot(1, 4, 4)  # Create subplot for the "after" state
plot_lattice(lattice, 'After')

plt.show()