import numpy as np
import matplotlib.pyplot as plt
import time 
from numba import jit
# test

start_time = time.time()

# Parameter
L = 128           # Lattice size LxL
T = 5         # Temperature - critical temperature is around 2.269
J = 1                 # Coupling constant (J > 0 for ferromagnetic interaction)
steps = 1024            # Number of Monte Carlo steps
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
            energy += -J * S * neighbours / 2  # Each pair counted twice

    energy -= H * np.sum(lattice)
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

# Run simulation
lattice = initialise_lattice(L)
before_lattice = lattice.copy()  # Save the initial lattice state for plotting

energies = []
magnetizations = []
lattice_histories = []

for step in range(steps):
    monte_carlo_step(lattice, T)
    
    energy = calc_energy(lattice)
    magnetization = np.sum(lattice)/(L*L)
    energies.append(energy)
    magnetizations.append(magnetization)

    # Create array of all lattice states
    lattice_histories.append(lattice.copy())

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Lattice points: {L*L}, Steps: {steps}, Elapsed time: {elapsed_time:.2f} seconds")

# Plot the results
# plt.plot(energies, label="Energy")
# plt.xlabel("Monte Carlo step")
# plt.legend()
# plt.show()

count=0
# Plot the magnetisations
# for i in range(16):
#     magnetizations = []
#     lattice = initialise_lattice(L)
#     for step in range(steps):
#         monte_carlo_step(lattice, T)
#         magnetization = np.sum(lattice)/(L*L)
#         magnetizations.append(magnetization)
#         magnetizations_current = magnetizations
#     if magnetization > 0:
#         count += 1
#     plt.plot(range(steps), magnetizations_current, label="Magnetization")

# # plot a line at y = 0
# print(count)
# plt.axhline(y=0, color='black', linestyle='--', label="y = 0")
# plt.xlim(0, steps)
# plt.ylabel("Magnetisation")
# plt.xlabel("Step")
# plt.show()

import matplotlib.pyplot as plt

temperatures = [1, 2.269, 5]
fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=100)

for idx, T in enumerate(temperatures):
    ax = axes[idx]
    all_magnetizations = []
    for i in range(16):
        magnetizations = []
        lattice = initialise_lattice(L)
        for step in range(steps):
            monte_carlo_step(lattice, T)
            magnetization = np.sum(lattice) / (L * L)
            magnetizations.append(magnetization)
        ax.plot(range(steps), magnetizations)
        all_magnetizations.extend(magnetizations)
        ax.set_xlim(0, steps)
        ax.axhline(y=0, color='black', linestyle='--')
    maxvalue = max(abs(m) for m in all_magnetizations)
    ax.set_ylim(-maxvalue-0.05, maxvalue+0.05)
    ax.set_xticks(range(0, steps + 1, 64))
    if idx == 0:
        ax.set_title("Magnetisation vs Stepcount for different T's", fontsize=16)
    if idx == 1:
        ax.set_ylabel("Magnetisation", fontsize=14)
    if idx == len(temperatures) - 1:
        ax.set_xlabel("Stepcount", fontsize=14)
    else:
        ax.set_xticklabels([])
    ax.legend([f"T = {T}"], loc='upper right', fontsize='large')

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()

exit()

# Function to plot the lattice

def plot_lattice(lattice, title):
    plt.imshow(lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)  # coolwarm for clear distinction
    plt.title(title)
    # plt.colorbar(label='Spin')

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

# Code to plot a gradient time series of lattice progression
def gradplot(total_columns, plots):

    fig, axes = plt.subplots(1, plots+1, figsize=(15, 7), gridspec_kw={'wspace': 0, 'hspace': 0})  # Use gridspec_kw to remove spacing

    indiv_columns = total_columns//plots # or total_columns//steps
    start = 0
    stop = indiv_columns - 1
    step = 0
    spp = int(steps/plots) # steps per plot

    for i in range(plots+1):
        if i == plots:
            step = step - 1

        ax = axes[i]
        plt.sca(ax)
        plot_lattice(lattice_histories[step][:, start:stop], step)

        # Update start and stop for the next step
        start = (start + indiv_columns) % L  # Wrap around after 128 columns
        stop = (start + indiv_columns - 1) % L  # Ensure stop stays within bounds
        step += spp


total_columns = 4*L
plots = 32

gradplot(total_columns, plots)
plt.tight_layout()
plt.show()

# Simulate for different temperatures
temperatures = np.linspace(2, 2.5, 30)
magnetizations = []

# for T in temperatures:
#     # Initialize the lattice
#     print(T)
#     lattice = initialise_lattice(L) 
    
#     # Run the simulation
#     for step in range(steps):
#         monte_carlo_step(lattice, T)
    
#     # After thermalization, calculate the average magnetization
#     mag = np.abs(calc_mag(lattice)) / (L * L)  # Normalize by number of spins
#     magnetizations.append(mag)

# # Plot results
# plt.plot(temperatures, magnetizations, '-o')
# plt.axvline(x=2.269, color='r', linestyle='--', label="Theoretical $T_c$")
# plt.xlabel("Temperature (T)")
# plt.ylabel("Magnetisation")
# plt.title("Magnetisation vs Temperature")
# plt.legend()
# plt.show()

def plot_lattice(lattice, title):
    plt.imshow(lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)  
    plt.title(title)
    # plt.colorbar(label='Spin')

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

# Plot the "before" state
plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)  # Create subplot for the "before" state
plot_lattice(lattice_histories[0], 'Initial')

# Plot the "after" state
plt.subplot(1, 4, 2)  # Create subplot for the "after" state
plot_lattice(lattice_histories[round(steps/10)], '10% of steps')

# Plot the "after" state
plt.subplot(1, 4, 3)  # Create subplot for the "after" state
plot_lattice(lattice_histories[round(steps/2)], '50% of steps')

# Plot the "after" state
plt.subplot(1, 4, 4)  # Create subplot for the "after" state
plot_lattice(lattice, 'Final')

plt.show()