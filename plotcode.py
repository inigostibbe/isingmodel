
# def plot_lattice(lattice, title):
#     plt.imshow(lattice, cmap='gray', interpolation='none', vmin=-1, vmax=1)  # coolwarm for clear distinction
#     plt.title(title)
#     # plt.colorbar(label='Spin')

#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])

# # Plot the "before" state
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 4, 1)  # Create subplot for the "before" state
# plot_lattice(before_lattice, 'Before')

# # Plot the "after" state
# plt.subplot(1, 4, 2)  # Create subplot for the "after" state
# plot_lattice(lattice_10, '10% of steps')

# # Plot the "after" state
# plt.subplot(1, 4, 3)  # Create subplot for the "after" state
# plot_lattice(lattice_50, '50% of steps')

# # Plot the "after" state
# plt.subplot(1, 4, 4)  # Create subplot for the "after" state
# plot_lattice(lattice, 'After')

# Number of columns in the plot 512 (this is 4 full plots 4x128)

def gradplot(width=512, plots):

    # plt.figure(figsize=(10, 5))

    fig, axes = plt.subplots(1, steps, figsize=(15, 10), gridspec_kw={'wspace': 0, 'hspace': 0})  # Use gridspec_kw to remove spacing

    indiv_columns = total_columns//steps

    start = 0
    stop = indiv_columns - 1
    step = 0
    spp = steps/plots # steps per plot

    for i in range(steps):
        ax = axes[i]
        plt.sca(ax)
        # plt.subplot(1, steps, i+1)  # Create subplot for the "after" state
        plot_lattice(lattice_histories[step][:, start:stop], i+1)

        # Update start and stop for the next step
        start = (start + indiv_columns) % 128  # Wrap around after 128 columns
        stop = (start + indiv_columns - 1) % 128  # Ensure stop stays within bounds

        step += spp


width = 512 # total number of columns
plots = 16
steps = 128

steps/plots = # number of steps per plot

#lattice_histories[i]