import numpy as np
import matplotlib.pyplot as plt
import time 

start_time = time.time()

# Parameters
L = 128           # Lattice size LxL
T = 0.1             # Temperature
J = 1.0                 # Coupling constant (J > 0 for ferromagnetic interaction)
steps = 256            # Number of Monte Carlo steps
k_B = 1      # Boltzmann constant
H = 0              # External magnetic field

# Initialize the lattice (random spins +1 or -1)
def initialise_lattice(L):
    lattice = np.random.choice([1, -1], size=(L, L))
    return lattice

# Calculate the magnetisation for a spin configuration
def calc_mag(lattice):
    return np.sum(lattice)

def sum_neighbours(i,j):
    return lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]

# Calculate energy for a spin configuration
def calc_energy(lattice):
    energy = 0
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbours = sum_neighbours(i,j)
            energy += -J * S * neighbours / 2  # Each pair counted twice

    energy -= H * np.sum(lattice)
    return energy

# Monte Carlo step using Metropolis algorithm
def monte_carlo_step(lattice, T):
    for _ in range(L**2):
        i, j = np.random.randint(0, L, 2)
        S = lattice[i, j]
        # Calculate the energy difference if the spin is flipped
        neighbours = sum_neighbours(i,j)
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
    
    # if step % 100 == 0:  # Sample every 100 steps
    energy = calc_energy(lattice)
    magnetization = np.sum(lattice)
    energies.append(energy)
    magnetizations.append(magnetization)

    # Makes copies of the lattice at 25% and 50% of the steps
    if step == steps//10:
        lattice_10 = lattice.copy()

    if step == steps//2:
        lattice_50 = lattice.copy()

    lattice_histories.append(lattice.copy())

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Plot the results
plt.plot(energies, label="Energy")
plt.plot(magnetizations, label="Magnetization")
plt.xlabel("Monte Carlo step")
plt.legend()
plt.show()

# Function to plot the lattice

def plot_lattice(lattice, title):
    plt.imshow(lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)  # coolwarm for clear distinction
    plt.title(title)
    # plt.colorbar(label='Spin')

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

print(len(lattice_histories))

# Code to plot a gradient time series of lattice progression
def gradplot(total_columns, plots):

    fig, axes = plt.subplots(1, plots+1, figsize=(10, 5), gridspec_kw={'wspace': 0, 'hspace': 0})  # Use gridspec_kw to remove spacing

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
        start = (start + indiv_columns) % 128  # Wrap around after 128 columns
        stop = (start + indiv_columns - 1) % 128  # Ensure stop stays within bounds
        step += spp


total_columns = 512
plots = 32

gradplot(total_columns, plots)
plt.tight_layout()
plt.show()

# # Code to plot a gradient time series of lattice progression
# def gradplot(total_columns=512, steps):

#     # plt.figure(figsize=(10, 5))

#     fig, axes = plt.subplots(1, steps, figsize=(15, 10), gridspec_kw={'wspace': 0, 'hspace': 0})  # Use gridspec_kw to remove spacing

#     indiv_columns = total_columns//steps

#     start = 0
#     stop = indiv_columns - 1

#     for i in range(steps):
#         ax = axes[i]
#         plt.sca(ax)
#         # plt.subplot(1, steps, i+1)  # Create subplot for the "after" state
#         plot_lattice(lattice_histories[i][:, start:stop], i+1)

#         # Update start and stop for the next step
#         start = (start + indiv_columns) % 128  # Wrap around after 128 columns
#         stop = (start + indiv_columns - 1) % 128  # Ensure stop stays within bounds


# total_columns = 512
# steps = 100

# gradplot(total_columns, steps)
# plt.tight_layout()
# plt.show()