import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

# Import Spirit modules
from spirit import state, configuration, simulation, geometry, system, io, parameters

def sublattice(p_state, spins_file, positions_file):
    spins = system.get_spin_directions(p_state)
    positions = geometry.get_positions(p_state)

    spins_new = []
    positions_new = []

    n_cells = geometry.get_n_cells(p_state)  # Number of cells in Bravais lattice
    n_cell_atoms = geometry.get_n_cell_atoms(p_state)  # Basis atoms per unit cell

    for c in range(n_cells[2]):
        for b in range(n_cells[1]):
            for a in range(n_cells[0]):
                for i in range(n_cell_atoms):
                    idx = i + n_cell_atoms * (a + n_cells[0] * (b + n_cells[1] * c))
                    spins_new.append(spins[idx])
                    positions_new.append(positions[idx])

    np.savetxt(spins_file, spins_new)
    np.savetxt(positions_file, positions_new)

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
    """Plots spin configuration and ensures output is always generated."""
    plt.figure(figsize=(8, 8))

    colors = custom_color_map(spins)  # Get custom colors based on spins

    plt.quiver(positions[:, 0], positions[:, 1], spins[:, 0], spins[:, 1], 
               color=plt.cm.seismic(colors), pivot="middle", scale=50, scale_units="xy", width=0.005)

    plt.title(title)
    if message:
        plt.text(0.5, 1.05, message, fontsize=12, ha='center', transform=plt.gca().transAxes)

    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved: {output_file}")

def get_center(positions, spins, center_up=None, cutoff_center=0):
    if center_up is None:
        center_up = np.mean(spins[:, 2]) > 0

    mask = spins[:, 2] < cutoff_center if center_up else spins[:, 2] > cutoff_center
    positions_center = positions[mask]
    return np.mean(positions_center, axis=0)