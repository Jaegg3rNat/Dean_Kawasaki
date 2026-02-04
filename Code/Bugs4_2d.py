"""
Bugs4_2d.py

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

# -------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------


# # ==========================================================
# @njit(parallel=True)
# def compute_forces(xp, yp, fx, fy, N, L, epsilon, R):
#     """
#     Compute interaction forces between all pairs of particles in a 2D periodic system.
#
#     Parameters:
#         xp (np.array): Array of x-coordinates of particles.
#         yp (np.array): Array of y-coordinates of particles.
#         fx (np.array): Array to store x-components of forces.
#         fy (np.array): Array to store y-components of forces.
#         N (int): Number of particles.
#         L (float): System size (length of the periodic box).
#         epsilon (float): Interaction strength parameter.
#         R (float): Interaction range parameter.
#
#     Returns:
#         fx (np.array): Updated x-components of forces.
#         fy (np.array): Updated y-components of forces.
#     """
#     fx[:] = 0.0  # Reset force arrays
#     fy[:] = 0.0
#
#     # Temporary arrays to avoid race conditions in parallel computation
#     fx_temp = np.zeros(N)
#     fy_temp = np.zeros(N)
#
#     # Loop over all pairs of particles
#     for i in prange(N - 1):  # Parallel loop over particle i
#         for j in range(i + 1, N):  # Loop over particle j (always after i)
#             # Compute shortest periodic distance in x and y
#             dx = xp[i] - xp[j]
#             dy = yp[i] - yp[j]
#             rx = dx - L * np.round(dx / L)  # PBC in x
#             ry = dy - L * np.round(dy / L)  # PBC in y
#
#             # Compute actual distance r
#             r = np.sqrt(rx**2 + ry**2)
#
#             if r > 1e-12:  # Avoid division by zero
#                 # Compute force magnitude using an exponential potential
#                 potential_term = epsilon * np.exp(-(r / R) ** 3)
#                 force_magnitude = (3 / R) * (r / R) ** 2 * potential_term
#
#                 # Compute force components
#                 fijx = force_magnitude * (rx / r)  # x-component of force
#                 fijy = force_magnitude * (ry / r)  # y-component of force
#
#                 # Update forces (equal and opposite)
#                 fx_temp[i] += fijx
#                 fy_temp[i] += fijy
#                 fx_temp[j] -= fijx
#                 fy_temp[j] -= fijy
#
#     # Copy results back to fx and fy
#     for i in prange(N):
#         fx[i] = fx_temp[i]
#         fy[i] = fy_temp[i]
#
#     return fx, fy
# # # =========================================================
# @cuda.jit
# def compute_forces_cuda(xp, yp, fx, fy, N, L, epsilon, R,threads_per_block):
#     """
#     Compute interaction forces using CUDA parallelization with shared memory.
#     """
#     i = cuda.grid(1)  # Get thread index
#     if i >= N:
#         return  # Exit if out of bounds
#
#     # Shared memory for particle positions (x and y)
#     shared_xp = cuda.shared.array(32, dtype=float32)
#     shared_yp = cuda.shared.array(32, dtype=float32)
#
#     # Load particle positions into shared memory
#     shared_xp[cuda.threadIdx.x] = xp[i]
#     shared_yp[cuda.threadIdx.x] = yp[i]
#
#     # Synchronize threads within the block
#     cuda.syncthreads()
#
#     fx_i, fy_i = 0.0, 0.0  # Local force accumulation
#
#     # Loop over all particles in the block (optimized with shared memory)
#     for j in range(N):
#         if i != j:  # Avoid self-interaction
#             # Calculate the shortest distance between particles using PBC
#             dx = shared_xp[cuda.threadIdx.x] - xp[j]
#             dy = shared_yp[cuda.threadIdx.x] - yp[j]
#             rx = dx - L * round(dx / L)  # PBC in x
#             ry = dy - L * round(dy / L)  # PBC in y
#
#             r = math.sqrt(rx ** 2 + ry ** 2)
#
#             if r > 1e-12:  # Avoid division by zero
#                 potential_term = epsilon * math.exp(-(r / R) ** 3)
#                 force_magnitude = (3 / R) * (r / R) ** 2 * potential_term
#                 fijx = force_magnitude * (rx / r)
#                 fijy = force_magnitude * (ry / r)
#
#                 fx_i += fijx
#                 fy_i += fijy
#
#     # Store results back to global memory
#     fx[i] = fx_i
#     fy[i] = fy_i
#
# # Function to launch CUDA kernel
# def compute_forces_gpu(xp, yp, N, L, epsilon, R):
#     # Allocate device memory
#     d_xp = cuda.to_device(xp)
#     d_yp = cuda.to_device(yp)
#     d_fx = cuda.device_array(N, dtype=np.float32)
#     d_fy = cuda.device_array(N, dtype=np.float32)
#
#     # Optimized grid and block configuration
#     device = cuda.get_current_device()
#     max_blocks = device.MULTIPROCESSOR_COUNT * 8  # 8-10x multiprocessors
#     threads_per_block = 32  # Optimal for modern GPUs
#     blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
#     # blocks_per_grid = min(max_blocks, (N + threads_per_block - 1) // threads_per_block)
#
#     # Launch kernel
#     compute_forces_cuda[blocks_per_grid, threads_per_block](d_xp, d_yp, d_fx, d_fy, N, L, epsilon, R,threads_per_block)
#
#     # Copy results back to host
#     fx = d_fx.copy_to_host()
#     fy = d_fy.copy_to_host()
#
#     return fx, fy



@cuda.jit
def compute_forces_cuda(xp, yp, fx, fy, N, L, epsilon, R):
    i = cuda.grid(1)
    if i >= N:
        return

    fx_i, fy_i = 0.0, 0.0  # Local force accumulation

    xi = xp[i]
    yi = yp[i]

    for j in range(N):
        if i != j:
            dx = xi - xp[j]
            dy = yi - yp[j]
            rx = dx - L * math.floor((dx / L) + 0.5)
            ry = dy - L * math.floor((dy / L) + 0.5)

            r = math.sqrt(rx ** 2 + ry ** 2)

            if r > 1e-12:
                potential_term = epsilon * math.exp(-((r / R) ** 3))
                force_mag = (3 / R) * ((r / R) ** 2) * potential_term
                fijx = force_mag * (rx / r)
                fijy = force_mag * (ry / r)

                fx_i += fijx
                fy_i += fijy

    fx[i] = fx_i
    fy[i] = fy_i

def compute_forces_gpu(xp, yp, N, L, epsilon, R):
    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    d_xp = cuda.to_device(xp)
    d_yp = cuda.to_device(yp)
    d_fx = cuda.device_array(N, dtype=np.float32)
    d_fy = cuda.device_array(N, dtype=np.float32)

    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    compute_forces_cuda[blocks_per_grid, threads_per_block](d_xp, d_yp, d_fx, d_fy, N, L, epsilon, R)

    fx = d_fx.copy_to_host()
    fy = d_fy.copy_to_host()

    return fx, fy








# ================================================================
def initial_random(N,bounds):
    # Initialize random positions for N particles within the bounds
    x = np.random.uniform(bounds[0, 0], bounds[0, 1], N)  # x positions
    y = np.random.uniform(bounds[1, 0], bounds[1, 1], N)  # y positions

    # Apply periodic boundary conditions (PBC) for a toroidal system
    x = bounds[0, 0] + (x - bounds[0, 0]) % (bounds[0, 1] - bounds[0, 0])
    y = bounds[1, 0] + (y - bounds[1, 0]) % (bounds[1, 1] - bounds[1, 0])
    return x, y
def initial_continue(N):
    # Open Bugs File
    base_folder = '../Data/Model0/Bugs_2d/Model4'
    # base_folder = 'Bugs_2d/Model4'  # Adjusted path for the new directory structure
    # Open the HDF5 file and inspect contents
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    t0  = 0
    file_path = f'{base_folder}/_N{N}_D{0.82:.4f}/dat1.h5'
    with h5py.File(file_path, 'r') as f:
        time_ = f['time'][:]  # Export time array from simulation
        group_name = f"t{time_[-1]}"  # Example group
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

# Time array
T = 0.1  # Simulation duration
dt = 0.00001
t = np.arange(0, T + dt, dt)  # Array of time steps
nt = len(t)  # Number of time steps
t0 = 1

D = float(sys.argv[1])/1000



# Interaction potential parameters
epsilon = 0.0333  # Interaction strength
R = 0.1  # Interaction range


REAL_DIFFUSION = D*(epsilon * N)
jump = np.sqrt(2 * REAL_DIFFUSION * dt)  # Diffusion step size

# print(R/np.sqrt(3*REAL_DIFFUSION/2))
# -------------------------------------------------------------------------------------
# Directory Setup
# -------------------------------------------------------------------------------------

# Define a directory to store results
main_directory = "Bugs_2d"  # Main directory for results
if not os.path.exists(main_directory):
    os.makedirs(main_directory)  # Create directory if it doesn't exist

# Automatically create subdirectory based on initial configuration
lattice_size_dir = f"{main_directory}/Model4"  # Subdirectory for Gaussian initialization
if not os.path.exists(lattice_size_dir):
    os.makedirs(lattice_size_dir)

# Create path for saving results based on N and D values
path = f"{lattice_size_dir}/_N{N}_D{D*1000:.4f}"  # Path for specific simulation
if not os.path.exists(path):
    os.makedirs(path)



# -------------------------------------------------------------------------------------
# Initial Conditions
# -------------------------------------------------------------------------------------

# x, y = initial_random(N,bounds)

x,y ,t0 = initial_continue(N)
# plt.scatter(x, y, alpha=0.5, s=10)  # Scatter plot
# plt.show()
# plt.close()

# fx = np.zeros(N, dtype=cp.float16)
# fy = np.zeros(N, dtype=cp.float16)

fx = np.zeros(N, dtype=np.float32)
fy = np.zeros(N, dtype=np.float32)



# -----------------------
# Display Simulation Parameters
# -----------------------
print("Simulation Parameters:")
print("\nBiological Parameters:")
print(f"Real Diffusion (D): {REAL_DIFFUSION}")
print(f"Number of Particles (N): {N}")
print(f"Interaction Radius (R): {R}")
print(f"Kernel Amplitude (ε): {epsilon}")
print("\nDomain Parameters:")
print(f"Domain Length (L): {L}")
print(f'Time Step (dt): {dt}')
print(f'Stochastic Jump: {jump}')
print(f'Model Type: Bugs')
print(f'Simulation Time: {T}')
# -------------------------------------------------------------------------------------
# Time Evolution
# -------------------------------------------------------------------------------------

# Create arrays to store results
vec_time = []  # Array to store time steps
# Ensure s_peak and time_series are defined before loop starts
s_peak = []
time_series = []


def check_time_step_safety(fx, fy, noisex, noisey, dt, R, D, safety_factor=1):
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
    # Compute drift displacement (fx * dt)
    drift_disp = np.sqrt((fx * dt) ** 2 + (fy * dt) ** 2)
    max_drift = np.max(drift_disp)
    mean_drift = np.mean(drift_disp)

    # Noise displacement (already scaled by sqrt(2D dt))
    noise_disp = np.sqrt(noisex ** 2 + noisey ** 2)
    max_noise = np.max(noise_disp)
    mean_noise = np.mean(noise_disp)
    rms_noise = np.sqrt(np.mean(noise_disp ** 2))  # RMS noise displacement

    # Stability conditions
    safe_drift = (max_drift < safety_factor * R)  # Avoid force-driven jumps
    safe_noise = (rms_noise < safety_factor * R)  # Prevent noise from dominating
    safe_energy = (dt * np.max(np.sqrt(fx ** 2 + fy ** 2))) < R  # Prevent instability
    dV = (mean_drift+ mean_noise)/dt
    safe = safe_drift and safe_noise and safe_energy

    return {
        'max_drift_ratio': max_drift,
        # 'rms_noise_ratio': rms_noise ,
        'Mean delta:': mean_drift+ mean_noise,
        # 'energy_condition_met': safe_energy,
        'Ratio"': R/dV,
        'recommendation': 'Reduce dt' if not safe else 'dt is OK',
    }
def normal_integration(nt,x,y,fx,fy,L):
    h5file = h5py.File(f"{path}/dat1.h5", "w")  # HDF5 file for storing data
    # Time evolution loop
    for it in tqdm(range(1, nt)):  # Progress bar for time steps
        # Compute forces between particles (turn off the force if ideal gas model)
        # fx, fy = compute_forces(x, y, fx, fy, N, L, epsilon, R)

        fx, fy = compute_forces_gpu(x, y, N, L, epsilon, R)

        # Add noise to simulate diffusion
        noise_x = jump * np.random.normal(size=x.shape)
        noise_y = jump * np.random.normal(size=y.shape)

        # print('Average Noise:',np.mean(noise_x), 'Average force:', np.mean(fx))
        # if it == 500:
        #     break

        # Update positions
        x += fx * dt + noise_x
        y += fy * dt + noise_y

        # Apply periodic boundary conditions (PBC)
        x = bounds[0, 0] + (x - bounds[0, 0]) % (bounds[0, 1] - bounds[0, 0])
        y = bounds[1, 0] + (y - bounds[1, 0]) % (bounds[1, 1] - bounds[1, 0])


        # if it % 10 == 0:
        #     print(check_time_step_safety(fx, fy, noise_x, noise_y, dt, R, D, safety_factor=0.1))
        # # Save data and generate plots at regular intervals
        if it % 1000 == 0:
            # Save particle positions to HDF5 file
            group = h5file.create_group(f"t{t[it]}")  # Create a group for each time
            group.create_dataset("x", data=x)
            group.create_dataset("y", data=y)
            vec_time.append(t[it].round(7))

        # # Generate and save scatter plot of particle positions (optional)

        if it % 100 == 0:
            # Compute S and append to list
            power_spectrum, k_values = compute_S(1, N, D, 100,x,y)
            k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
            s_peak.append(s_max)
            time_series.append(it * dt)

            # Create a figure with 2 subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Left: scatter plot of particle positions
            axs[0].scatter(x, y, alpha=0.5, s=5)
            axs[0].scatter(x[0],y[0], color='red', s=20, label='Bug 0')  # Highlight the first bug
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
def u0_continue(N,D):
    # Open Bugs File
    base_folder = 'Bugs_2d/Model4'
    # Open the HDF5 file and inspect contents
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    file_path = f'{base_folder}/_N{N}_D{D*1000:.4f}/dat1.h5'
    with h5py.File(file_path, 'r') as f:
        time_ = f['time'][:]  # Export time array from simulation
        group_name = f"t{time_[-1]}"  # Example group
        if group_name in f:
            group = f[group_name]  # Access the group
            x0 = group["x"][:]  # Read the dataset for x
            y0 = group["y"][:]  # Read the dataset for y
        f.close()
    return x0, y0
def storage_optimazation(fx,fy,L,snaps,N,D):
    #Call File dat1
    x,y = u0_continue(N,D)
    # Create file dat2 for saving results
    h5file = h5py.File(f"{path}/dat1.h5", "w")  # HDF5 file for storing data
    # Time evolution loop
    for it in tqdm(range(1,snaps+1)):  # Progress bar for time steps
        # Compute forces between particles (turn off the force if ideal gas model)
        # fx, fy = compute_forces(x, y, fx, fy, N, L, epsilon, R)

        fx, fy = compute_forces_gpu(x, y, N, L, epsilon, R)

        # Add noise to simulate diffusion
        noise_x = jump * np.random.normal(size=x.shape)
        noise_y = jump * np.random.normal(size=y.shape)

        # Update positions
        x += fx * dt + noise_x
        y += fy * dt + noise_y

        # Apply periodic boundary conditions (PBC)
        x = bounds[0, 0] + (x - bounds[0, 0]) % (bounds[0, 1] - bounds[0, 0])
        y = bounds[1, 0] + (y - bounds[1, 0]) % (bounds[1, 1] - bounds[1, 0])

        # # Save data and generate plots at regular intervals

        # Save particle positions to HDF5 file
        group = h5file.create_group(f"t{t[it]}")  # Create a group for each time
        group.create_dataset("x", data=x)
        group.create_dataset("y", data=y)
        vec_time.append(t[it].round(7))


    # Save time array to HDF5 file
    h5file.create_dataset("time", data=vec_time)

    # Explicitly close the HDF5 file
    h5file.close()

'''
normal_integral:
    - Performs the normal Particle temporal evolution
storage_optimization:
    - Trick to maintain the storage usage low;
    - Starts from the end of particle integration and save it only the number of snaps configurations 
    - dt spacing. 
'''
normal_integration(nt,x,y,fx,fy,L)
storage_optimazation(fx,fy,L,1000,N,D)