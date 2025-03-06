import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import Spirit module 
from spirit import state, configuration, simulation, geometry, system, hamiltonian, io

def custom_color_map(spins, threshold=0.999):
    """Generate a custom color based on the z-component of spins."""
    colors = np.zeros(spins.shape[0])  # Initialize color array

    # Assign colors based on the z-component of spins
    for i in range(spins.shape[0]):
        if spins[i, 2] >= threshold:
            colors[i] = 1  # Assign red for spins very close to 1
        elif spins[i, 2] <= -threshold:
            colors[i] = -1  # Assign blue for spins very close to -1
        else:
            colors[i] = spins[i, 2]  # Use actual z-component values for intermediate colors

    # Normalize colors for use with colormap
    return (colors + 1) / 2  # Scale colors from [0, 2] to [0, 1]

def plot_spin_configuration(positions, spins, output_file, title, message=None):
    """Plots spin configuration with a white background and no box around the plot."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")  # Set background to white
    
    # Generate custom color map for spins
    colors = custom_color_map(spins)
    
    # Plot spin vectors
    ax.quiver(
        positions[:, 0], positions[:, 1], spins[:, 0], spins[:, 1], 
        color=plt.cm.jet(colors), pivot="middle", scale=50, scale_units="xy", width=0.005
    )
    
    # Remove the box (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove ticks, labels, and the grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Turn off axis completely
    plt.axis('off')

    # Add title in black for visibility
    plt.title(title, fontsize=12, fontweight="bold", color='black')

    # Add optional message (if provided)
    if message:
        plt.text(0.5, 1.05, message, fontsize=12, ha='center', transform=plt.gca().transAxes, color='black')

    # Save with white background
    plt.savefig(output_file, bbox_inches='tight', facecolor="white", dpi=300)
    plt.close(fig)
    
    print(f"Plot saved: {output_file}")

def normalize_spins(spins, threshold=0.999):
    """Normalize spins but keep original values for visualization and threshold values."""
    norms = np.linalg.norm(spins, axis=1, keepdims=True)
    max_norm = np.max(norms)
    
    scaled_spins = spins / max_norm  # Normalize spins
    scaled_spins[np.abs(scaled_spins) > threshold] = np.sign(scaled_spins[np.abs(scaled_spins) > threshold])  

    return scaled_spins

def get_profile(positions_new, spins_new, positions, spins, output_file, cutoff_center=0, cutoff_ring=0.33, center_up=None):
    """Analyzes skyrmion profile using new positions/spins but plots using original ones."""
    spins_new = normalize_spins(spins_new)

    if center_up is None:
        center_up = np.mean(spins_new[:, 2]) < 0

    mask = spins_new[:, 2] > cutoff_center if center_up else spins_new[:, 2] < cutoff_center
    positions_center = positions_new[mask]

    if len(positions_center) == 0:
        print(f"No skyrmion detected for {output_file}. Plotting anyway.")
        plot_spin_configuration(positions, spins, f"plot_{output_file}.png",
                                f"Spin Configuration (B = {output_file})", message="No Skyrmion Detected")
        return None, 0, None

    center = np.mean(positions_center, axis=0)
    rho = np.linalg.norm(positions_new - center, axis=1)

    profile = np.zeros((len(spins_new), 5))
    profile[:, 0] = rho
    profile[:, 1:4] = spins_new

    direction_vectors = positions_new - center
    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    normalized_directions = direction_vectors / norms

    profile[:, 4] = np.sum(spins_new * normalized_directions, axis=1)

    profile = profile[np.argsort(profile[:, 0])]
    ring_mask = np.abs(profile[:, 3]) < cutoff_ring

    if np.sum(ring_mask) == 0:
        print(f"Not enough valid points for fitting in {output_file}. Skipping curve fit.")
        radius = 0
        popt = None
    else:
        def mz_fun(rho, r, a):
            return a * (rho - r)
        radius = np.mean(profile[:, 0][ring_mask])
        popt, _ = curve_fit(mz_fun, profile[:, 0][ring_mask], profile[:, 3][ring_mask], p0=[radius, 1])
        radius = popt[0]

    plot_spin_configuration(positions, spins, f"AFM_skyrmion_{output_file}.png",
                            f"Skyrmion Profile ({output_file} meV)\nRadius = {radius:.1f} $\AA$")
    return profile, radius, popt

# Simulation setup
cfgfile = "input_AFM_skyrmion.cfg"
quiet = True

B_values = np.arange(0.00, 30, 5)  # Changing applied B from 0 to 30 at step 5
radii = []

for B in B_values:
    with state.State(cfgfile, quiet) as p_state:
        io.image_read(p_state, "AFM_skyrmion.ovf")
        hamiltonian.set_field(p_state, B, [0, 0, 1], idx_image=-1, idx_chain=-1)
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_LBFGS_OSO)
        system.update_data(p_state)

        # Get original spins and positions
        spins = system.get_spin_directions(p_state)
        positions = geometry.get_positions(p_state)

        # Initialize new lists
        spins_new = []
        positions_new = []

        # Get lattice structure
        n_cells = geometry.get_n_cells(p_state)     # number of cells in Bravais lattice
        n_cell_atoms = geometry.get_n_cell_atoms(p_state)  # number of basis atoms per unit cell

        # Filter spins and positions
        for c in range(n_cells[2]):
            for b in range(n_cells[1]):
                for a in range(n_cells[0]):
                    for i in range(n_cell_atoms):
                        sublattice = (a % 2)
                        sublattice_2 = (b % 2)
                        if sublattice_2 == sublattice:
                            idx = i + n_cell_atoms * (a + n_cells[0] * (b + n_cells[1] * c))
                            spins_new.append(spins[idx])
                            positions_new.append(positions[idx])

        # Convert to numpy arrays
        spins_new = np.array(spins_new)
        positions_new = np.array(positions_new)

        # Save to files (optional, for debugging)
        np.savetxt(f"spins_B{B:.2f}.txt", spins_new)
        # np.savetxt(f"positions_B{B:.2f}.txt", positions_new)

        # Compute profile and radius using filtered spins/positions
        profile, radius, popt = get_profile(positions_new, spins_new, positions, spins, f"B{B:.2f}")

        # Store radius value
        radii.append(radius)
