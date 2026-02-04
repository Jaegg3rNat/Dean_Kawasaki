"""
Bugs1_1d.py

Author: Nathan Silvano
Date: [2025-May-DD]
Description:
    .
"""
# import sns
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import h5py
import os
import sys


# Functions

@njit(parallel=True)
def compute_forces(xp, fx, N, xL, epsilon, R):
    fx[:] = 0.0  # Reset the force array each time step

    # Temporary array to avoid race conditions
    fx_temp = np.zeros(N)

    for i in prange(N - 1):  # Particle i loop
        for j in range(i + 1, N):  # Particle j loop (always after i)
            # Compute shortest periodic distance with sign
            dx = xp[i] - xp[j]
            rx = dx - xL * np.round(dx / xL)  # This keeps rx in [-xL/2, xL/2]

            # Compute force magnitude
            # r2 = rx ** 2

            r = abs(rx)
            # if r <2.5*R:
            potential_term = epsilon * np.exp(- (r / R) ** 3)

            force_magnitude = (3 / R) * (r / R) ** 2 * potential_term

            fijx = force_magnitude * (rx / r)  # Keep direction of rx

            fx_temp[i] += fijx  # Force on particle i
            fx_temp[j] -= fijx  # Equal and opposite force on particle j

    # Copy results back to fx
    for i in prange(N):
        fx[i] = fx_temp[i]

    return fx


def diffusion(x, D0, D1):
    return D0 + D1 * x


"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
def bugs1(number):
    """
    _________________________________________________________________________________________________________________________________________
    _______________________________________________________Initialization of System__________________________________________________________
    """
    # System parameters
    N = int(sys.argv[1])  # Number of particles

    # Domain definition
    bounds = np.array([0, 1])
    L = bounds[1] - bounds[0]

    # Time array
    T = 0.5
    dt = 1e-6
    t = np.arange(0, T + dt, dt)
    nt = len(t)

    # Interacting potential if needed
    # New interaction potential parameters
    epsilon = 0.0333
    R = 0.1
    # distumbral = 2.5 * R  # cutoff region


    """
    _________________________________________________________________________________________________________________________________________
    _______________________________________________________Directory Setup______________________________________________________________
    """
    # Define a directory to store results
    main_directory = "Bugs_1d"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

    # Automatically create subdirectory based on initial configuration
    lattice_size_dir = f"{main_directory}/D(x)"
    if not os.path.exists(lattice_size_dir):
        os.makedirs(lattice_size_dir)

    # Create path for saving results based on nx value and system parameters
    path = f"{lattice_size_dir}/_N{N}"
    if not os.path.exists(path):
        os.makedirs(path)

    # Create file for saving results
    h5file = h5py.File(f"{path}/dat.h5", "w")

    """
    _________________________________________________________________________________________________________________________________________
    _______________________________________________________Initial Conditions______________________________________________________________
    """
    # Initialize random positions for N bugs within the bounds
    # x = np.random.uniform(bounds[0], bounds[1], N)  # x positions

    # Initialize positions with Gaussian distribution around the center
    center = (bounds[1] + bounds[0]) / 2  # Center point (0.0 in this case)
    mu = 0.05  # Standard deviation of the initial distribution
    # Generate positions using normal distribution
    x = np.random.normal(center, mu, N)

    # Apply periodic boundary conditions to initial positions to ensure they're within bounds
    x = bounds[0] + (x - bounds[0]) % L

    # Store initial positions
    h5file.create_dataset(f"t{round(t[0], 7)}", data=x)

    # Initialize force arrays
    # fx = np.zeros(N)

    # # Create arrays for histogram plotting
    hist_range = (bounds[0], bounds[1])  # Range of the histogram matching the domain bounds
    bins = 256
    # # # Plot initial positions with histogram
    # plt.figure(figsize=(10, 6))
    #
    # # # Create two subplots
    # plt.subplot(2, 1, 1)
    # plt.scatter(x, np.zeros_like(x), alpha=0.5, s=10)  # Scatter plot on x-axis
    # plt.xlabel('x')
    # plt.title('Initial Bug Positions (Scatter)')
    # plt.grid(True)
    # plt.xlim(bounds[0], bounds[1])
    # #
    # plt.subplot(2, 1, 2)
    # plt.hist(x, bins=bins, range=hist_range, density=True, alpha=0.7)
    # # plt.plot(x_array, np.exp(-(x_array - center)**2 / (2 * mu**2)) / (mu * np.sqrt(2 * np.pi)), 'r-', linewidth=2)
    # plt.xlabel('x')
    # plt.ylabel('Density')
    # plt.title('Initial Bug Distribution (Histogram)')
    # plt.grid(True)
    # plt.xlim(bounds[0], bounds[1])

    # plt.tight_layout()
    # plt.show()
    # plt.close()
    ################################################################
    # Create arrays to store results
    vec_time = [t[0]]
    x_min, x_max = bounds[0], bounds[1]

    # Time evolution
    for it in tqdm(range(1, nt)):
        #     # Compute interactions using optimized function
        # fx = compute_forces(x, fx, N, L, epsilon, R)
        #
        # Update positions (Langevin-like dynamics with noise)
        jump = np.sqrt(dt * 2 * diffusion(x, 0.1, 1))  # Jump size for Langevin-like dynamics
        noise_x = jump * np.random.normal(size=x.shape)
        #
        # Upadate position X of particles
        x += noise_x
    # --------------------------------------
        # Apply periodic boundary conditions
        # x = bounds[0] + (x - bounds[0]) % L
        # Reflective boundaries
        # Reflective boundaries (vectorized)
        x = np.where(x < x_min, x_min + (x_min - x), x)
        x = np.where(x > x_max, x_max - (x - x_max), x)
    # ----------------------------------------
    #
    #     if it % 1000 == 0:
    #         h5file.create_dataset(f"t{round(t[it], 7)}", data=x)
    #         vec_time.append(t[it].round(7))
    #
    #     if it % 1000 == 0:
    #         # Plot initial positions with histogram
    #         plt.figure(figsize=(10, 6))
    #
    #         # Create two subplots
    #         plt.subplot(2, 1, 1)
    #         plt.scatter(x, np.zeros_like(x), alpha=0.5, s=10)  # Scatter plot on x-axis
    #         plt.xlabel('x')
    #         plt.title('Bug Positions at t = {}'.format(round(it * dt, 3)))
    #         plt.grid(True)
    #         plt.xlim(bounds[0], bounds[1])
    #
    #         plt.subplot(2, 1, 2)
    #         plt.hist(x, bins=bins, range=hist_range, density=True, alpha=0.7)
    #         plt.xlabel('x')
    #         plt.ylabel('Density')
    #
    #         plt.grid(True)
    #         plt.xlim(bounds[0], bounds[1])
    #
    #         plt.tight_layout()
    #         # plt.show()
    #         plt.savefig(f'{path}/fig{it}')
    #         plt.close()
    np.savetxt(f'../Data/Model1/Model1Bugs_N{N}_{number}.txt', x, fmt='%.6e', header='x y')
    h5file.create_dataset("time", data=vec_time)
    # Explicitly close the file
    h5file.close()
bugs1(sys.argv[2])