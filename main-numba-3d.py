import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

start_time = time.time()

# Parameters
L = 64           # Lattice size LxLxL (reduce size for 3D to keep things manageable)
T = 1            # Temperature
J = 1.0          # Coupling constant (J > 0 for ferromagnetic interaction)
steps = 256      # Number of Monte Carlo steps
k_B = 1          # Boltzmann constant
H = 0            # External magnetic field
np.random.seed(42)  # Set seed for reproducibility

# Initialize the lattice (random spins +1 or -1) in 3D
@jit(nopython=True)
def initialise_lattice(L):
    # Generate a random 3D lattice of 0s and 1s and map them to -1 and 1
    lattice = 2 * np.random.randint(0, 2, size=(L, L, L)) - 1
    return lattice

# Calculate the magnetisation for a spin configuration in 3D
@jit(nopython=True)
def calc_mag(lattice):
    return np.sum(lattice)

# Sum the neighbors in the 3D lattice
@jit(nopython=True)
def sum_neighbours(lattice, i, j, k, L):
    return (lattice[(i+1)%L, j, k] + lattice[(i-1)%L, j, k] +  # neighbors in x-direction
            lattice[i, (j+1)%L, k] + lattice[i, (j-1)%L, k] +  # neighbors in y-direction
            lattice[i, j, (k+1)%L] + lattice[i, j, (k-1)%L])   # neighbors in z-direction

# Calculate energy for a spin configuration in 3D
@jit(nopython=True)
def calc_energy(lattice, L, J, H):
    energy = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                S = lattice[i, j, k]
                neighbours = sum_neighbours(lattice, i, j, k, L)
                energy += -J * S * neighbours / 2  # Each pair counted twice
    energy -= H * np.sum(lattice)
    return energy

# Monte Carlo step using Metropolis algorithm in 3D
@jit(nopython=True)
def monte_carlo_step(lattice, T, L, J, k_B):
    for _ in range(L**3):  # Iterate over all lattice points
        i, j, k = np.random.randint(0, L, 3)  # Choose random spin to flip
        S = lattice[i, j, k]
        # Calculate the energy difference if the spin is flipped
        neighbours = sum_neighbours(lattice, i, j, k, L)
        delta_E = 2 * J * S * neighbours
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            lattice[i, j, k] *= -1  # Flip the spin

# Run simulation
lattice = initialise_lattice(L)
before_lattice = lattice.copy()  # Save the initial lattice state for plotting

energies = []
magnetizations = []
lattice_histories = []

for step in range(steps):
    monte_carlo_step(lattice, T, L, J, k_B)
    
    energy = calc_energy(lattice, L, J, H)
    magnetization = np.sum(lattice)
    energies.append(energy)
    magnetizations.append(magnetization)

    # Create array of all lattice states
    lattice_histories.append(lattice.copy())

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# # Plot the results
# plt.plot(energies, label="Energy")
# plt.plot(magnetizations, label="Magnetization")
# plt.xlabel("Monte Carlo step")
# plt.legend()
# plt.show()

# Function to visualize a slice of the lattice in 2D (since 3D is hard to visualize directly)
def plot_lattice_slice(lattice, slice_index, title):
    plt.imshow(lattice[slice_index], cmap='gray', interpolation='none', vmin=-1, vmax=1)  # Show a 2D slice
    plt.title(title)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

# Function to plot the 3D lattice as voxels
def plot_3d_voxels(lattice):
    L = lattice.shape[0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D grid of colors based on the spins (+1 = red, -1 = blue)
    colors = np.empty(lattice.shape, dtype=object)
    colors[lattice == 1] = 'red'
    colors[lattice == -1] = 'blue'

    # Create a voxel plot with color mapping
    ax.voxels(lattice == 1, facecolors=colors, edgecolor='k', alpha = 0.8)

    # Set labels
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    ax.set_title('3D Lattice Plot')

    plt.show()

# Example: Plot the final state of the lattice in 3D voxels
plot_3d_voxels(lattice_histories[-1])
