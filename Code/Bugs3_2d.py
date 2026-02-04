"""
Bugs3_2d.py

Author: Nathan Silvano
Date: [2025-May-DD]
Description:
    .
"""
# Import necessary libraries
import math
import os  # For directory and file operations
import sys  # For system-specific parameters

# import cupy as cp  # For GPU-accelerated computations
import h5py  # For saving data in HDF5 format
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations
from numba import cuda, jit, njit, prange, float32
from tqdm import tqdm  # For progress bars

import re
# ==================================================================================
### HDF5 File Inspection
def print_name(name, obj):
    """Function to print the name of groups and datasets in an HDF5 file."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")


def print_hdf5_contents(file_path):
    """Function to open an HDF5 file and print its contents."""
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_name)


# =================================================================================
# this function is specfic to the DK project data
def time_hdf5(file_path):
    """Function to extract time values from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        t_values = np.array([
            float(re.search(r"t(\d+(\.\d+)?)", name).group(1))
            for name in f.keys() if re.match(r"t\d+(\.\d+)?$", name)
        ])
        t_values.sort()
    return t_values


# =================================================================================




# ==================================================================================
# =================================================================================
# Fourier Transform of Fields
# =================================================================================

def compute_power_spectrum_2d(field, N, bins):
    """Compute the 2D power spectrum of the field and return the frequencies and power spectrum."""
    L = 1
    nx = bins
    dx = L / nx
    fft_vals = np.fft.fft2(field)
    power_spectrum = np.abs(fft_vals) ** 2 / (nx * nx * N)  # Normalize by total number of points

    # # Real-space power (mean squared amplitude)
    # power_real = np.sum(np.abs(field) ** 2)
    #
    # # Fourier-space power (sum of power spectrum)
    # power_fourier = np.sum(np.abs(fft_vals) ** 2) /(nx*nx)  # Your normalization
    # print(power_fourier, power_real)
    # These should be approximately equal (up to floating-point errors)
    # assert np.allclose(power_real, power_fourier)

    kx = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi  # Convert to wavenumbers
    ky = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_magnitude = np.sqrt(kx_grid ** 2 + ky_grid ** 2)  # Compute radial wavenumber

    # Define radial bins
    k_bins = np.linspace(0, np.max(k_magnitude), nx // 2)
    radial_spectrum = np.zeros_like(k_bins)

    # Compute the radial power spectrum by averaging in annular bins
    for i in range(len(k_bins) - 1):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
        if np.any(mask):
            radial_spectrum[i] = np.mean(power_spectrum[mask])
    # plt.plot(radial_spectrum)

    return k_bins[:-1], radial_spectrum[:-1]  # Remove last bin to match lengths


def find_max_characteristic_frequency_2d(k_bins, radial_spectrum):
    """Find the characteristic radial wavenumber with the highest power after k=0."""

    mask = k_bins > 0  # Ignore k=0
    k_filtered = k_bins[mask]
    power_filtered = radial_spectrum[mask]

    max_index = np.argmax(power_filtered)
    return k_filtered[max_index], power_filtered[max_index]


def compute_smax(base_folder, N, p, snaps, nx):
    file_path = f'{base_folder}/_N{N}_D{p:.4f}/dat.h5'
    power_mean = np.zeros(nx // 2 - 1)
    # print(len(power_mean))
    with h5py.File(file_path, 'r') as f:
        t = time_hdf5(file_path)
        for i in range(snaps):
            rho = f[f't{t[-(i + 1)]}'][:]
            # rho = clean_high_frequencies(rho, 1, 128, plot=True)
            k, power = compute_power_spectrum_2d(rho, N, nx)
            # print(len(power))
            power_mean += power
        power_mean /= snaps
        f.close()
        _, smax = find_max_characteristic_frequency_2d(k, power_mean)
    return smax


def compute_S(snaps, N, D, bins,x,y):
    powerList = np.zeros(int(bins / 2 - 1))


    # Compute 2D histogram
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], density=True)
    # plt.imshow(counts, cmap='hot')
    # plt.show()
    # plt.close()
    # Compute power spectrum
    k_values, power_spectrum = compute_power_spectrum_2d(counts, N, bins)
    # print(k_values[0:10])
    powerList += power_spectrum
    # powerList /= snaps
    return powerList, k_values

if not cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your GPU drivers or environment.")

# ================================================================
def initial_random(N, bounds):
    # Initialize random positions for N particles within the bounds
    x = np.random.uniform(bounds[0, 0], bounds[0, 1], N)  # x positions
    y = np.random.uniform(bounds[1, 0], bounds[1, 1], N)  # y positions

    # Apply periodic boundary conditions (PBC) for a toroidal system
    x = bounds[0, 0] + (x - bounds[0, 0]) % (bounds[0, 1] - bounds[0, 0])
    y = bounds[1, 0] + (y - bounds[1, 0]) % (bounds[1, 1] - bounds[1, 0])
    return x, y


def initial_continue(N,p):
    # Open Bugs File
    # base_folder = '../Data/Model3/Bugs'
    base_folder = 'Bugs_2d/Model3/'
    # Open the HDF5 file and inspect contents
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    t0 = 0
    file_path = f'{base_folder}/_N{N}_p{p:.2f}/dat1.h5'
    with h5py.File(file_path, 'r') as f:
        time_ = f['time'][:]  # Export time array from simulation
        group_name = f"t{int(time_[-1])}"  # Example group
        if group_name in f:
            group = f[group_name]  # Access the group
            x0 = group["x"][:]  # Read the dataset for x
            y0 = group["y"][:]  # Read the dataset for y
            t0 = time_[-1]
        f.close()

    return x0, y0, t0


# -------------------------------------------------------------------------------------
# Initialization of System
# -------------------------------------------------------------------------------------

# System parameters

N = int(sys.argv[2])  # Number of particles
bins = 512  # Number of bins for histogram plotting

# Domain definition (2D system)
bounds = np.array([[0, 1], [0, 1]])  # Square domain
L = bounds[0, 1] - bounds[0, 0]  # System size (same for x and y)
print('L:', L)

# Time array
T = 200000  # Simulation duration
dt = 100 # Time step
t = np.arange(0, T + dt, dt)  # Array of time steps
nt = len(t)  # Number of time steps

D = 1000000
p = float(sys.argv[1])
R = 0.1

# -------------------------------------------------------------------------------------
# Directory Setup
# -------------------------------------------------------------------------------------

# Define a directory to store results
main_directory = "Bugs_2d"  # Main directory for results
if not os.path.exists(main_directory):
    os.makedirs(main_directory)  # Create directory if it doesn't exist

# Automatically create subdirectory based on initial configuration
lattice_size_dir = f"{main_directory}/Model3"  # Subdirectory for Gaussian initialization
if not os.path.exists(lattice_size_dir):
    os.makedirs(lattice_size_dir)

# Create path for saving results based on N and D values
path = f"{lattice_size_dir}/_N{N}_p{p:.2f}"  # Path for specific simulation
if not os.path.exists(path):
    os.makedirs(path)

# -------------------------------------------------------------------------------------
# Initial Conditions
# -------------------------------------------------------------------------------------

# x, y = initial_random(N, bounds)
x,y ,t0 = initial_continue(N,p)

# -------------------------------------------------------------------------------------
# Time Evolution
# -------------------------------------------------------------------------------------

# Create arrays to store results
vec_time = []  # Array to store time steps
# Ensure s_peak and time_series are defined before loop starts
s_peak = []
time_series = []

@njit
def compute_Nr_numba(x, y, R, L):
    N = x.shape[0]
    Nr = np.zeros(N, dtype=np.int32)
    R2 = R * R
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            # Periodic boundary conditions (minimum image)
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dist2 = dx * dx + dy * dy
            if dist2 < R2:
                Nr[i] += 1
    # Prevent zero (for division safety)
    for i in range(N):
        if Nr[i] == 0:
            Nr[i] = 1
    return Nr


def compute_Nr(x, y, R, L):
    """Compute number of neighbors within radius R using periodic BCs."""
    Nr = np.zeros(N, dtype=int)
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        # Apply periodic boundary conditions (minimum image)
        dx = dx - L * np.round(dx / L)
        dy = dy - L * np.round(dy / L)
        dist2 = dx ** 2 + dy ** 2
        Nr[i] = np.sum(dist2 < R ** 2) - 1  # exclude self
    return np.maximum(Nr, 1)  # prevent division by zero

def check_time_step_safety( noisex, noisey, dt, R, D, safety_factor=1):
    """
    Checks if dt is safe for Langevin particle simulations.

    Parameters:
    - fx, fy: Force components (shape [N], NOT multiplied by dt).
    - noisex, noisey: Noise components (ALREADY scaled by sqrt(2D dt)).
    - dt: Time step.
    - R: Interaction range.
    - D: Diffusion coefficient (default=1).
    - safety_factor: Max allowed displacement as a fraction of R (default=0.1).

    Returns:
    - Dictionary with diagnostics and recommendation.
    """


    # Noise displacement (already scaled by sqrt(2D dt))
    noise_disp = np.sqrt(noisex ** 2 + noisey ** 2)
    max_noise = np.max(noise_disp)
    mean_noise = np.mean(noise_disp)
    rms_noise = np.sqrt(np.mean(noise_disp ** 2))  # RMS noise displacement

    # Stability conditions

    safe_noise = (rms_noise < safety_factor * R)  # Prevent noise from dominating
    safe = safe_noise

    return {
        # 'rms_noise_ratio': rms_noise ,
        'Mean delta:': mean_noise,
        # 'energy_condition_met': safe_energy,
        'recommendation': 'Reduce dt' if not safe else 'dt is OK',
    }
def normal_integration(nt, x, y, L):
    h5file = h5py.File(f"{path}/dat1.h5", "w")  # HDF5 file for storing data
    # Time evolution loop
    for it in tqdm(range(1, nt)):  # Progress bar for time steps
        Nr = compute_Nr_numba(x, y, R, L)  # Compute number of neighbors

        jump = np.sqrt(2 * D * (Nr / N) ** p * dt)
        # Add noise to simulate diffusion
        noise_x = jump * np.random.normal(size=x.shape)
        noise_y = jump * np.random.normal(size=y.shape)


        x += noise_x
        y += noise_y

        # Apply periodic boundary conditions (PBC)
        x = bounds[0, 0] + (x - bounds[0, 0]) % (bounds[0, 1] - bounds[0, 0])
        y = bounds[1, 0] + (y - bounds[1, 0]) % (bounds[1, 1] - bounds[1, 0])
        if it % 10 == 0:
            print(check_time_step_safety( noise_x, noise_y, dt, R, D, safety_factor=1))

        # # Save data and generate plots at regular intervals
        if it % 1000 == 0:
            # Save particle positions to HDF5 file
            group = h5file.create_group(f"t{t[it]}")  # Create a group for each time
            group.create_dataset("x", data=x)
            group.create_dataset("y", data=y)
            vec_time.append(t[it].round(7))
        if it % 100 == 0:
            # Compute S and append to list
            power_spectrum, k_values = compute_S(1, N, D, 128, x, y)
            k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
            s_peak.append(s_max)
            time_series.append(it * dt)

            # Create a figure with 2 subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Left: scatter plot of particle positions
            axs[0].scatter(x, y, alpha=0.5, s=5)
            axs[0].scatter(x[0], y[0], color='red', s=20, label='Bug 0')  # Highlight the first bug
            axs[0].set_title(f"Bug Positions at t = {round(it * dt, 3)}")
            axs[0].grid(True)
            axs[0].set_aspect('equal')  # Optional: keeps aspect ratio square

            # Right: S vs time plot
            axs[1].plot(time_series, s_peak, '-o', markersize=4)
            axs[1].set_title("S vs Time")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("S (peak power spectrum)")

            # plt.figure(figsize=(6, 6))
            # plt.scatter(x, y, alpha=0.5, s=10)  # Scatter plot
            # plt.title(f"Bug Positions at t = {round(it * dt, 3)}")
            # plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{path}//fig")  # Save plot as image
            plt.close()

    # Save time array to HDF5 file
    h5file.create_dataset("time", data=vec_time)

    # Explicitly close the HDF5 file
    h5file.close()


def u0_continue(N, p):
    # Open Bugs File
    base_folder = 'Bugs_2d/Model3'
    # Open the HDF5 file and inspect contents
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    file_path = f'{base_folder}/_N{N}_p{p:.2f}/dat1.h5'
    with h5py.File(file_path, 'r') as f:
        time_ = f['time'][:]  # Export time array from simulation
        group_name = f"t{time_[-1]}"  # Example group
        if group_name in f:
            group = f[group_name]  # Access the group
            x0 = group["x"][:]  # Read the dataset for x
            y0 = group["y"][:]  # Read the dataset for y
        f.close()
    return x0, y0


def storage_optimazation(L, snaps, N, p):
    # Call File dat1
    x, y = u0_continue(N, p)
    # Create file dat2 for saving results
    h5file = h5py.File(f"{path}/dat1.h5", "w")  # HDF5 file for storing data
    # Time evolution loop
    for it in tqdm(range(1, snaps + 1)):  # Progress bar for time steps
        Nr = compute_Nr_numba(x, y, R, L)  # Compute number of neighbors

        jump = np.sqrt(2 * D * (Nr / N) ** p * dt)
        # Add noise to simulate diffusion
        noise_x = jump * np.random.normal(size=x.shape)
        noise_y = jump * np.random.normal(size=y.shape)

        # print(noise_x)
        # break
        # Update p1ositions
        x += noise_x
        y += noise_y

        # Apply periodic boundary conditions (PBC)
        x = bounds[0, 0] + (x - bounds[0, 0]) % (bounds[0, 1] - bounds[0, 0])
        y = bounds[1, 0] + (y - bounds[1, 0]) % (bounds[1, 1] - bounds[1, 0])

        # # Save data and generate plots at regular intervals

        # Save particle positions to HDF5 file
        group = h5file.create_group(f"t{round(t[it], 7)}")  # Create a group for each time
        group.create_dataset("x", data=x)
        group.create_dataset("y", data=y)
        vec_time.append(t[it].round(7))
        if it % 100 == 0:
            # Compute S and append to list
            power_spectrum, k_values = compute_S(1, N, D, 128, x, y)
            k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
            s_peak.append(s_max)
            time_series.append(it * dt)

            # Create a figure with 2 subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Left: scatter plot of particle positions
            axs[0].scatter(x, y, alpha=0.5, s=5)
            axs[0].scatter(x[0], y[0], color='red', s=20, label='Bug 0')  # Highlight the first bug
            axs[0].set_title(f"Bug Positions at t = {round(it * dt, 3)}")
            axs[0].grid(True)
            axs[0].set_aspect('equal')  # Optional: keeps aspect ratio square

            # Right: S vs time plot
            axs[1].plot(time_series, s_peak, '-o', markersize=4)
            axs[1].set_title("S vs Time")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("S (peak power spectrum)")

            # plt.figure(figsize=(6, 6))
            # plt.scatter(x, y, alpha=0.5, s=10)  # Scatter plot
            # plt.title(f"Bug Positions at t = {round(it * dt, 3)}")
            # plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{path}//fig")  # Save plot as image
            plt.close()

    # Save time array to HDF5 file
    h5file.create_dataset("time", data=vec_time)

    # Explicitly close the HDF5 file
    h5file.close()




# -------------------------------------------------------------------------------------
# Run the simulation
normal_integration(nt, x, y, L)  # Call the time evolution function
s_peak = []
time_series = []
storage_optimazation(L, 1000, N, p)
