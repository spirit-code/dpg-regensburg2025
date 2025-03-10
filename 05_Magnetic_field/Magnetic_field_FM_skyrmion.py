import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from spirit import state, configuration, simulation, geometry, system, hamiltonian, io

def custom_color_map(spins, threshold=0.999):
    """Generate a custom color based on the z-component of spins."""
    colors = np.zeros(spins.shape[0])

    for i in range(spins.shape[0]):
        if spins[i, 2] >= threshold:
            colors[i] = 1  # Assign red for spins very close to 1
        elif spins[i, 2] <= -threshold:
            colors[i] = -1  # Assign blue for spins very close to -1
        else:
            colors[i] = spins[i, 2]

    return (colors + 1) / 2  # Scale colors from [0, 2] to [0, 1]

def plot_spin_configuration(positions, spins, output_file, title, message=None):
    """Plots spin configuration and ensures output is always generated."""
    plt.figure(figsize=(8, 8))
    colors = custom_color_map(spins)
    plt.quiver(positions[:, 0], positions[:, 1], spins[:, 0], spins[:, 1], 
               color=plt.cm.seismic(colors), pivot="middle", scale=50, scale_units="xy", width=0.005)
    plt.title(title)
    if message:
        plt.text(0.5, 1.05, message, fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved: {output_file}")

def normalize_spins(spins, threshold=0.999):
    """Normalize spins but keep original values for visualization and threshold values."""
    norms = np.linalg.norm(spins, axis=1, keepdims=True)
    max_norm = np.max(norms)
    scaled_spins = spins / max_norm
    scaled_spins[np.abs(scaled_spins) > threshold] = np.sign(scaled_spins[np.abs(scaled_spins) > threshold])
    return scaled_spins

def get_profile(positions, spins, output_file, cutoff_center=0, cutoff_ring=0.33, center_up=None):
    """Analyzes the skyrmion profile and extracts key properties."""
    spins = normalize_spins(spins)
    if center_up is None:
        center_up = np.mean(spins[:, 2]) < 0
    mask = spins[:, 2] > cutoff_center if center_up else spins[:, 2] < cutoff_center
    positions_center = positions[mask]

    if len(positions_center) == 0:
        print(f"No skyrmion detected for {output_file}. Plotting anyway.")
        plot_spin_configuration(positions, spins, f"plot_{output_file}.png", f"Spin Configuration (B = {output_file})", "No Skyrmion Detected")
        return None, 0, None

    center = np.mean(positions_center, axis=0)
    rho = np.linalg.norm(positions - center, axis=1)
    profile = np.zeros((len(spins), 5))
    profile[:, 0] = rho
    profile[:, 1:4] = spins
    direction_vectors = positions - center
    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    normalized_directions = direction_vectors / norms
    profile[:, 4] = np.sum(spins * normalized_directions, axis=1)
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

    plot_spin_configuration(positions, spins, f"FM_skyrmion_{output_file}.png", f"Skyrmion Profile ( {output_file})\nRadius = {radius:.1f} Å")
    return profile, radius, popt