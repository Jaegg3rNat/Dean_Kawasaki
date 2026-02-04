"""
SPDE3.py

Author: Nathan Silvano
Date: [2025-March-DD]
Description:
    This script handles the numerical integration of the Dean-Kawasaki equation.
    Both the Deterministic Equation as well the complete Stochastic equation are handle.
    It uses a PseudoSpectral approach with one time step to perform the integration.
    The noise is handle using the Cornalba algorithm.
"""

# === Import Necessary Libraries ===
import os
import re
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.fftpack import fft2, ifft2, ifftshift, fftfreq  # Fast Fourier Transform operations
from numba import jit, njit, prange  # JIT compilation for optimization


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


def time_hdf5(file_path):
    """Function to extract time values from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        t_values = np.array([
            float(re.search(r"t(\d+(\.\d+)?)", name).group(1))
            for name in f.keys() if re.match(r"t\d+(\.\d+)?$", name)
        ])
        t_values.sort()
    return t_values




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

def u0_continue(N, p, MODEL_TYPE):
    '''
    Initial Condition from the end of a deterministic density
    Args:
        N:

    Returns:

    '''
    # Open DK File
    base_folder = f'../Data/Model3/{MODEL_TYPE}/p{nx}'
    # base_folder = f'Model3/{MODEL_TYPE}/p{nx}'
    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/_N{N}_p{p:.1f}/dat.h5'
    with h5py.File(file_path, 'r') as f:
        t = time_hdf5(file_path)

        rho = f[f't{t[-1]}'][:]
        f.close()
    return rho


def tophat_kernel(dx, radius):
    # Calculate the radius in grid points corresponding to the competition radius
    r_int = int(radius / dx)  # Number of grid points within the competition radius

    # Initialize the kernel matrix
    num_points_inside = 0
    m = np.zeros((1 + 2 * r_int, 1 + 2 * r_int))  # Kernel matrix of appropriate size
    # m_norm = np.pi * radius ** 2  # Normalization constant for the 2D kernel

    # Populate the kernel matrix
    for i in range(-r_int, r_int + 1):
        for j in range(-r_int, r_int + 1):
            if i ** 2 + j ** 2 <= r_int ** 2:  # Inside the competition radius
                m[i + r_int, j + r_int] = 1.0  # / m_norm
                num_points_inside += 1

    # Calculate the area of the discretized kernel
    area_discretized_kernel = num_points_inside * dx ** 2
    m /= area_discretized_kernel
    # Initialize the domain-wide kernel matrix
    m2 = np.zeros((nx, ny))  # Kernel matrix in domain space

    # Place the kernel matrix at the center of the domain
    m2[nx // 2 - r_int:nx // 2 + r_int + 1, ny // 2 - r_int:ny // 2 + r_int + 1] = m

    # Normalize the kernel with grid spacing
    m2 *= dx * dy  # Ensure the kernel accounts for the area of each grid cell
    return m2


# =============================================
# =============================================
def stochastic_step_2d(u, dt, kx_, ky_, nstps, rho0, p):
    '''
    # ====================================================
    # FUNCTION: stochastic_step_2d
    # Performs a pseudospectral time-step operation in 2D
    # This is for the Stochastic DK
    # ====================================================
    Args:
        u:
        dt:
        D:
        kx_:
        ky_:
        nstps:

    Returns:

    '''
    # Compute the 2D Fourier transform of u
    u_hat = fft2(u)
    for _ in range(nstps):
        # ===========
        # DIFFUSION TERM
        # Convolve V with u in Fourier space
        conv_hat = V_hat * u_hat

        # Transform back to real space
        conv = ifftshift(ifft2(conv_hat))
        drho = conv ** p

        # # Multiply by u in real space
        full = u * drho

        diffusion_term = diff_coef * fft2(full)
        # print(diffusion_term)

        # ===========
        # NOISE TERM
        # up = np.maximum(0, u)  # Apply filter in square root
        # conv_hatp = V_hat * fft2(up)  # Convolve V with u in Fourier space
        # convp = ifftshift(ifft2(conv_hatp))
        convp =np.maximum(full, 0)  # Apply filter in square root
        noise_rho = np.sqrt(convp)
        # Generate Gaussian noise
        B = np.random.normal(0, np.sqrt(dt), size=(nx, ny))  # Create 2D gaussian noise
        # Compute Fourier derivatives separately in x and y directions
        Fx = fft2(noise_rho * B/dx)  # Fourier transform
        dFx_dx = -1j * kx_ * Fx  # Fourier derivative in x
        dFx_dy = -1j * ky_ * Fx  # Fourier derivative in y

        # Sum the components to get divergence
        noise_term = (dFx_dx + dFx_dy) * jump
        # ===========
        # Euler-Maruyama in Fourier space
        u_hat += diffusion_term + noise_term
        u = ifft2(u_hat).real  # Transform back to real space

    return u


# =============================================
# =============================================
def _step_2d(u, dt, kx_, ky_, nstps, rho0, p):
    '''
    # ====================================================
    # FUNCTION: stochastic_step_2d
    # Performs a pseudospectral time-step operation in 2D
    # This is for the Stochastic DK
    # ====================================================
    Args:
        u:
        dt:
        D:
        kx_:
        ky_:
        nstps:

    Returns:

    '''
    # Compute the 2D Fourier transform of u
    u_hat = fft2(u)
    for _ in range(nstps):
        # ===========
        # DIFFUSION TERM
        # Convolve V with u in Fourier space
        conv_hat = V_hat * u_hat

        # Transform back to real space
        conv = ifftshift(ifft2(conv_hat))
        drho = conv ** p

        # # Multiply by u in real space
        full = u * drho

        diffusion_term = diff_coef * fft2(full)
        # print(diffusion_term)

        # ===========
        # Euler-Maruyama in Fourier space
        u_hat += diffusion_term
        u = ifft2(u_hat).real  # Transform back to real space

    return u


def strat_step_2d(u, dt, kx_, ky_, nstps, rho0, p):
    '''
    # ====================================================
    # FUNCTION: stochastic_step_2d
    # Performs a pseudospectral time-step operation in 2D
    # This is for the Stochastic DK
    # ====================================================
    Args:
        u:
        dt:
        D:
        kx_:
        ky_:
        nstps:

    Returns:

    '''
    # Compute the 2D Fourier transform of u
    u_hat = fft2(u)
    for _ in range(nstps):
        # ===========
        # DIFFUSION TERM

        # 1. Compute V * u via convolution theorem
        conv_hat = V_hat * u_hat
        conv = ifftshift(ifft2(conv_hat))  # Get real part

        # 2. Compute (V*u)^p in real space
        conv_p = conv ** p

        # 3. Compute gradient of rho (proper 2D gradient)
        grad_x_rho = ifft2(1j * kx_ * u_hat).real
        grad_y_rho = ifft2(1j * ky_ * u_hat).real

        # 4. Compute full nonlinear term in real space
        full_x = conv_p * grad_x_rho
        full_y = conv_p * grad_y_rho

        # 5. Transform back to Fourier space and apply derivative
        diffusion_term_x = 1j * kx_ * fft2(full_x)
        diffusion_term_y = 1j * ky_ * fft2(full_y)

        # 6. Combine components and multiply by D0 and dt
        u_hat += D0 * (diffusion_term_x + diffusion_term_y) * dt

        # Transform back to real space for visualization/storage if needed
        u = ifft2(u_hat).real

    return u


# =============================================
# =============================================
"""
===========================================
    Simulation Parameters Configuration
===========================================
"""

# Set seed for reproducibility
seed = 10
np.random.seed(seed)
# -----------------------
# Biological Parameters
# -----------------------
R = 0.1  # Interaction radius
# Number of particles
N_PARTICLES = int(sys.argv[3])

# -----------------------
# Domain Parameters
# -----------------------
nx, ny = 128, 128  # Grid resolution
bounds = np.array([0, 1])
L = bounds[1] - bounds[0]  # System size

# Define spatial grid
x = np.linspace(*bounds, nx + 1)[:-1]
y = np.linspace(*bounds, ny + 1)[:-1]
X, Y = np.meshgrid(x, y, indexing="ij")

# Grid spacing
dx = L / nx
dy = L / ny
# Compute wave numbers
kx = fftfreq(nx, L / nx) * 2 * np.pi
ky = fftfreq(ny, L / ny) * 2 * np.pi
kx_, ky_ = np.meshgrid(kx, ky, indexing="ij")
# Compute magnitude of k-vector
k_magnitude = np.sqrt(kx_ ** 2 + ky_ ** 2)

sigma_k =100
# Gaussian filter: G(k) = exp(-k² / (2 σ_k²))
gaussian_filter = np.exp(-k_magnitude ** 2 / (2 * sigma_k ** 2))

rho0 = N_PARTICLES / L ** 2  # Initial density
D0 = 1e-4  # Diffusion coefficient
p = float(sys.argv[1])  # Exponent for nonlinearity
# -----------------------
# Interaction Kernel (Fourier Space)
M2 = tophat_kernel(dx, R)
V_hat = fft2(M2)  # Fourier transform of kernel

# Model type (deterministic by default)
MODEL_TYPE = sys.argv[2]
# -----------------------
# Time Parameters
# -----------------------
T = 15  # simulation duration
dt = 0.0001
t = np.arange(0, T + dt, dt)
nt = len(t)

# Coefficients
diff_coef = -D0 * (kx_ ** 2 + ky_ ** 2) * dt
jump = np.sqrt(2 * D0 / N_PARTICLES)


"""
===========================================
        Simulation Setup & Storage
===========================================
"""

# Create main results directory
MAIN_DIRECTORY = "Model3"
os.makedirs(MAIN_DIRECTORY, exist_ok=True)

# Automatically create subdirectory based on NX value
LATTICE_DIR = f"{MAIN_DIRECTORY}/{MODEL_TYPE}/p{nx}"
os.makedirs(LATTICE_DIR, exist_ok=True)

# Define path for saving results
RESULT_PATH = f"{LATTICE_DIR}/_N{N_PARTICLES}_p{p:.1f}"
os.makedirs(RESULT_PATH, exist_ok=True)

"""
===========================================
    Initial Conditions for Simulation
===========================================
"""
type = 'Homo'
if type == 'Homo':
    # Homogeneous + Fluctuations
    u0 = np.zeros((nx, ny))

    # np.random.seed(0)
    u0[:] = 1 + 1e-3 * np.random.rand(nx, ny)
else:
    u0 = u0_continue(N_PARTICLES,p, MODEL_TYPE)


# Create HDF5 file for storing results
h5file = h5py.File(f"{RESULT_PATH}/dat.h5", "w")


#
def fgaussian(kx, ky):
    """
    Generate a Gaussian function in Fourier space.

    Parameters:
    - kx, ky: Wave numbers.

    Returns:
    - Gaussian-modulated function.
    """
    width = 4 * np.pi ** 2 * 5  # Spread of the Gaussian
    return np.exp(-(kx ** 2 + ky ** 2) / width)


# Initialize the field
# Central Gaussian
# u0 = ifftshift(fft2(fgaussian(kx_, ky_ ))).real

u = np.copy(u0)
index = np.where(u == np.max(u))[0]

# Create arrays to store results
time_ = [t[0]]  # Time History
density_ = [np.mean(abs(u))]  # Density History
h5file.create_dataset(f"t{t[0]}", data=u)  # Save initial configuration in file

# -----------------------
# Display Simulation Parameters
# -----------------------
print("Simulation Parameters:")
print("\nBiological Parameters:")
print(f"Real Diffusion (D): {D0}")
print('Nonlinearity (p):', p)
print(f"Number of Particles (N): {N_PARTICLES}")
print(f"Interaction Radius (R): {R}")

print("\nDomain Parameters:")
print(f"Bounds: {bounds}")
print(f"Domain Length (L): {L}")
print(f"Grid Resolution: {nx} x {ny}")
print(f"Grid Spacing: dx={dx}, dy={dy}")
print(f'Time Step (dt): {dt}')
print(f"Initial Density: {rho0}")
print(f'Stochastic Jump: {jump}')
print(f'Model Tipy: {MODEL_TYPE}')
print(f'Nh**2: {N_PARTICLES *(dx * dx)}')

s_peak = []
time_series = []
def normal_integration(nt, u):
    count = 0
    for n in tqdm(range(1, nt)):
        # compute the next time step
        if MODEL_TYPE == 'det':
            u = _step_2d(u, dt, kx_, ky_, 50, rho0, p)  # Deterministic DK
            # u = strat_step_2d(u, dt, kx_, ky_, 50, rho0, p)
        else:
            u = stochastic_step_2d(u, dt, kx_, ky_, 50, rho0, p)

        time_.append(t[n])
        density_.append(np.mean(u))

        # # # Store data for density plot
        if n % 10000 == 0:
            # time_.append(round(t[n], 7))
            # density_.append(np.mean(u))
            h5file.create_dataset(f"t{t[n]}", data=u)
        # if n % 10000 == 0:
        #     # Create file for saving results
        #     h5file = h5py.File(f"{path}/dat2.h5", "w")
        #     time_.append(round(t[n], 7))
        #     density_.append(np.mean(u))
        #     h5file.create_dataset(f"t{round(t[n], 7)}", data=u)

        if n % 100 == 0:
            # # Generate and save scatter plot of particle positions (optional)

            # Compute S and append to list
            k_values,power_spectrum = compute_power_spectrum_2d(u,N_PARTICLES,nx)
            k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
            s_peak.append(s_max)
            time_series.append(n * dt)

            # Create a figure with 2 subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            im = axs[0].imshow(u.T, cmap="gnuplot", origin="lower", extent=np.concatenate((bounds, bounds)))
            fig.colorbar(im,ax = axs[0],ticks=np.linspace(np.min(u), np.min(u) + 0.9 * (np.max(u) - np.min(u)), 7))
            axs[0].set_title(f"t = {10 * t[n]:0.3f};")
            # plt.xlim([bounds[0], bounds[1]])
            axs[0].set_aspect('equal')  # Optional: keeps aspect ratio square

            # Right: S vs time plot
            axs[1].plot(time_series, s_peak, '-o', markersize=4)
            axs[1].set_title("S vs Time")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("S (peak power spectrum)")




            #     # #     # Optional: Save the figure
            plt.savefig(f"{RESULT_PATH}/density_{count + 1}.png")
            plt.close()
            # count += 1
    # Before Finish simulation Write the data
    # h5file.create_dataset("time", data=time_)
    # h5file.create_dataset("tot_density", data=density_)
    # Explicitly close the file
    h5file.close()


def storage_optimation(snaps):
    # Open DK File
    base_folder = f'Model3/{MODEL_TYPE}/p{nx}'
    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/_N{N_PARTICLES}_p{p:.1f}/dat.h5'
    with h5py.File(file_path, 'r') as f:
        t = time_hdf5(file_path)
        rho = f[f't{t[-1]}'][:]
        f.close()
    u = np.copy(rho)
    h5file = h5py.File(f"{RESULT_PATH}/dat.h5", "w")  #
    for n in tqdm(range(1, snaps)):
        # compute the next time step
        u = stochastic_step_2d(u, dt, kx_, ky_, 10, rho0, p)
        # Save each time step
        h5file.create_dataset(f"t{round(n * dt, 4)}", data=u)


normal_integration(nt, u)
if MODEL_TYPE == 'noise':
    storage_optimation(1000)
else:
    pass
