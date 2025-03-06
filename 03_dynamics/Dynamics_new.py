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

# Configuration file
cfgfile = "./input1.cfg"
n_iterations = 5000  # Total number of simulation steps
n_itertions_step = 1000  # Steps per iteration

total = []
quiet = True

with state.State(cfgfile, quiet) as p_state:
    io.image_read(p_state, "Bloch.ovf")
    system.update_data(p_state)
    alpha=0.2 #alpha value
    u=3 #current density related parameter
    parameters.llg.set_damping(p_state, alpha)#alpha value
    parameters.llg.set_stt(p_state, True, u, [1, 0, 0])
    
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_DEPONDT, single_shot=True, n_iterations=n_iterations)
    
    for x in range(int(n_iterations / n_itertions_step)):
        simulation.n_shot(p_state, n_itertions_step)
        
        if x == 0:
            io.image_write(p_state, "spins_0.ovf")  # Save the initial spin file
        
        if x == int(n_iterations / n_itertions_step) - 1:
            io.image_write(p_state, "spins_final.ovf")  # Save only the final spin file
        
        spins = np.array(system.get_spin_directions(p_state))
        sublattice(p_state, "spins.ovf", "positions.ovf")
        center = get_center(np.loadtxt("positions.ovf"), np.loadtxt("spins.ovf"))
        
        total.append([center[0], center[1]])
    
    simulation.stop_all(p_state)

# Save dynamics data
np.savetxt('dynamics.txt', np.array(total), fmt='%1.6f %1.6f', header='Center1_x Center1_y')
# Compute velocity and Hall angle
data = np.loadtxt("dynamics.txt")

Vx = np.mean(np.diff(data[:, 0]))
Vy = np.mean(np.diff(data[:, 1]))
V = math.sqrt(Vx**2 + Vy**2)
Theta = math.degrees(math.atan(Vy / Vx))

# Save results
velocity_results = [[Vx, Vy, V, Theta]]
np.savetxt('velocity_results.txt', velocity_results, fmt='%1.6f', header='Vx Vy V Theta(degrees)')

# Print results
print(f"Velocity components: Vx = {Vx:.5f}, Vy = {Vy:.5f}")
print(f"Total velocity: V = {V:.5f}")
print(f"Hall angle: Theta = {Theta:.5f} degrees")

# Plot dynamics data with annotations
data = np.loadtxt('dynamics.txt')
plt.figure()
plt.plot(data[:, 0], data[:, 1], marker='o', linestyle='-')
plt.xlabel('Position_x')
plt.ylabel('Position_y')
plt.title('Skyrmion Dynamics')

# Add velocity and Hall angle annotations
plt.text(0.5, 0.7, f'V = {V:.2f}, Hall angle = {Theta:.2f}°',
         fontsize=10, ha='center', transform=plt.gca().transAxes)
plt.grid()
plt.savefig('dynamics_plot.png')
plt.show()

# # Load and plot initial and final spin configurations
# positions = geometry.get_positions(p_state)
# spins_initial = np.loadtxt("spins_0.ovf")
# spins_final = np.loadtxt("spins_final.ovf")
# centers = np.array(total)

# # Plotting spin configurations with custom color mapping
# plot_spin_configuration(positions, spins_initial, "spins_0_plot.png", "Initial Spin Configuration")
# plot_spin_configuration(positions, spins_final, "spins_final_plot.png", "Final Spin Configuration")

