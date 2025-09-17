import numpy as np
import matplotlib.pyplot as plt
import time 
from numba import jit, njit, prange
from numba import config

# COMMENT OR UNCOMMENT THE JIT STATEMENTS TO TEST THE TIME DIFFERENCE

start_time = time.time()

# Parameter
L = 128           # Lattice size LxL
T = 1        # Temperature - critical temperature is around 2.269
J = 1                 # Coupling constant (J > 0 for ferromagnetic interaction)
steps = int(512*2)            # Number of Monte Carlo steps
k_B = 1      # Boltzmann constant
H = 0              # External magnetic field
np.random.seed(42) # Set seed for reproducibility

# Initialize the lattice (random spins +1 or -1)
@jit(nopython=True)
def initialise_lattice(L):
    # Generate a random lattice of 0s and 1s and map them to -1 and 1
    lattice = 2 * np.random.randint(0, 2, size=(L, L)) - 1
    return lattice

# Calculate the magnetisation for a spin configuration
@jit(nopython=True)
def calc_mag(lattice):
    return np.sum(lattice)

@jit(nopython=True)
def sum_neighbours(lattice,i,j):
    return lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]

# Calculate energy for a spin configuration
@jit(nopython=True)
def calc_energy(lattice):
    energy = 0
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbours = sum_neighbours(lattice,i,j)
            energy += -J * S * neighbours / 2  # Each connection between neighbours is counted twice so / 2

    energy -= H * np.sum(lattice) # this is -h sum s_i
    return energy

# Monte Carlo step using Metropolis algorithm
@jit(nopython=True)
def monte_carlo_step(lattice, T):
    for _ in range(L**2):
        i, j = np.random.randint(0, L, 2)
        S = lattice[i, j]
        # Calculate the energy difference if the spin is flipped
        neighbours = sum_neighbours(lattice,i,j)
        delta_E = 2 * J * S * neighbours
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            lattice[i, j] *= -1

import matplotlib.pyplot as plt
import matplotlib.animation as animation

steps = 128
temperatures = [0.1, 2.269, 5]
L = 512

lattice_histories = {T: [] for T in temperatures}
lattices = {T: initialise_lattice(L) for T in temperatures}

for T in temperatures:
    for i in range(int(steps)):
        monte_carlo_step(lattices[T], T)
        lattice_histories[T].append(lattices[T].copy())

def plot_lattice(ax, lattice, title):
    ax.imshow(lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def update(frame):
    for ax, T in zip(axs, temperatures):
        ax.clear()
        plot_lattice(ax, lattice_histories[T][frame], f'T = {T}, Step {frame}')

ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, repeat=True)

ani.save('lattice_animation.mp4', writer='ffmpeg', fps=20)

plt.show()