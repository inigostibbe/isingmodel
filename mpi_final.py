from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Lattice and model parameters
L = 64
T = 2.269
J = 1.0
steps = 256
k_B = 1

# Divide rows among processes
rows_per_proc = L // size
extra_rows = L % size

if rank < extra_rows:
    local_L = rows_per_proc + 1  # Extra row for processes with rank < extra_rows
else:
    local_L = rows_per_proc

# Initialize each local lattice independently
np.random.seed(42 + rank)  # Use different seeds for different processes
local_lattice = np.random.choice([-1, 1], (local_L, L))

def monte_carlo_step(local_lattice, T, J, k_B, ghost_top, ghost_bottom):
    """Perform a Monte Carlo update step."""
    for i in range(local_L):
        for j in range(L):
            S = local_lattice[i, j]
            neighbors_sum = (
                local_lattice[i, (j + 1) % L] + local_lattice[i, (j - 1) % L] +
                (ghost_top[j] if i == 0 else local_lattice[i - 1, j]) +
                (ghost_bottom[j] if i == local_L - 1 else local_lattice[i + 1, j])
            )
            delta_E = 2 * J * S * neighbors_sum
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
                local_lattice[i, j] *= -1

def exchange_borders(local_lattice):
    """Exchange borders with neighboring processes."""
    top_border = local_lattice[0, :].copy()
    bottom_border = local_lattice[-1, :].copy()
    ghost_top = np.zeros_like(top_border)
    ghost_bottom = np.zeros_like(bottom_border)

    if rank < size - 1:
        comm.Sendrecv(bottom_border, dest=rank + 1, recvbuf=ghost_bottom, source=rank + 1)
    if rank > 0:
        comm.Sendrecv(top_border, dest=rank - 1, recvbuf=ghost_top, source=rank - 1)

    return ghost_top, ghost_bottom

# Synchronize and start timer
comm.Barrier()
start_time = time.time()

# Monte Carlo simulation
for step in range(steps):
    ghost_top, ghost_bottom = exchange_borders(local_lattice)
    monte_carlo_step(local_lattice, T, J, k_B, ghost_top, ghost_bottom)

# Synchronize and end timer
comm.Barrier()
end_time = time.time()

# # Gather the final lattice for visualization
# row_counts = comm.gather(local_L, root=0)  # Gather number of rows per process

# if rank == 0:
#     recvcounts = [row_count * L for row_count in row_counts]  # Elements contributed by each process
#     displacements = [sum(recvcounts[:i]) for i in range(size)]  # Starting indices for each process
#     gathered_lattice = np.zeros((L * L), dtype=int)  # Flattened buffer for gathering
# else:
#     recvcounts = None
#     displacements = None
#     gathered_lattice = None


# Commenting out the gathering part as it does not work on BC4

# comm.Gatherv(local_lattice.flatten(),
#              [gathered_lattice, recvcounts, displacements, MPI.INT], root=0)

# Reshape the gathered lattice on the root process
if rank == 0:
    # gathered_lattice = gathered_lattice.reshape((L, L))
    # plt.imshow(gathered_lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)
    # plt.title('Final Lattice Configuration')
    # plt.show()
    print(f"L = {L}, steps = {steps} Time taken for simulation: {end_time - start_time:.2f} seconds")
