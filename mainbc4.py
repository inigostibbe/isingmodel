from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start timer
comm.Barrier()  # Synchronize all processes before starting the timer
start_time = time.time()

# Lattice and model parameters
L = 32               # Size of the lattice along one dimension
T = 1                 # Temperature
J = 1.0               # Coupling constant
steps = 10            # Number of Monte Carlo steps
k_B = 1               # Boltzmann constant
np.random.seed(42 + rank)  # Seed per rank for randomness (different for each process)

# Divide lattice among processes
local_L = L // size  # Each process gets a section of the lattice

# Initialize local lattice segment on each process
local_lattice = np.random.choice([-1, 1], (local_L, L))

# Function to exchange borders with neighboring processes
def exchange_borders(local_lattice):
    """Exchange borders with neighboring processes (periodic boundary)."""
    top_border = local_lattice[0, :].copy()
    bottom_border = local_lattice[-1, :].copy()
    recv_top = np.empty(L, dtype=int)
    recv_bottom = np.empty(L, dtype=int)

    # Compute neighbor ranks
    rank_above = (rank - 1 + size) % size
    rank_below = (rank + 1) % size

    # Send bottom row to the next process, receive top row from the previous
    comm.Sendrecv(bottom_border, dest=rank_below, sendtag=0,
                  recvbuf=recv_top, source=rank_above, recvtag=1)

    # Send top row to the previous process, receive bottom row from the next
    comm.Sendrecv(top_border, dest=rank_above, sendtag=1,
                  recvbuf=recv_bottom, source=rank_below, recvtag=0)

    return recv_bottom, recv_top

# Function to perform a Monte Carlo update step
def monte_carlo_step(local_lattice, recv_bottom, recv_top, T, J, k_B):
    """Perform a Monte Carlo update step on the lattice portion."""
    local_L, n_cols = local_lattice.shape

    # Create a new lattice with boundary rows included
    extended_lattice = np.vstack([recv_bottom, local_lattice, recv_top])

    for i in range(1, local_L + 1):  # Iterate over the local rows (excluding ghost rows)
        for j in range(n_cols):
            S = extended_lattice[i, j]
            # Calculate energy change using neighbors
            neighbors_sum = (
                extended_lattice[i, (j + 1) % n_cols] + extended_lattice[i, (j - 1) % n_cols] +
                extended_lattice[i + 1, j] + extended_lattice[i - 1, j]
            )
            delta_E = 2 * J * S * neighbors_sum
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
                local_lattice[i - 1, j] *= -1  # Flip spin

# Monte Carlo simulation
for step in range(steps):
    # Exchange borders before Monte Carlo step
    recv_bottom, recv_top = exchange_borders(local_lattice)
    # Perform Monte Carlo update step with the received borders
    monte_carlo_step(local_lattice, recv_bottom, recv_top, T, J, k_B)

# Gather results back to the master process
sendcounts = [((L // size + (1 if i < L % size else 0)) * L) for i in range(size)]
displacements = [sum(sendcounts[:i]) for i in range(size)]
gathered_lattice = None
if rank == 0:
    gathered_lattice = np.empty((L, L), dtype=int)

# Flatten local_lattice to send data correctly
comm.Gatherv(local_lattice.flatten(), [gathered_lattice, sendcounts, displacements, MPI.INT], root=0)

# End timer
comm.Barrier()  # Synchronize all processes before stopping the timer
end_time = time.time()

# Print the time taken by the master process
if rank == 0:
    print(f"Time taken for simulation: {end_time - start_time:.2f} seconds")
