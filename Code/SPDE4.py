"""
SPDE4.py

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
from scipy import fftpack  # Fast Fourier Transform operations
from numba import jit, njit, prange  # JIT compilation for optimization


# ====== Functions ======
def print_name(name, obj):
    """Function to print the name of groups and datasets in an HDF5 file."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")


def time_hdf5(file_path):
    """Function to extract time values from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        t_values = np.array([
            float(re.search(r"t(\d+(\.\d+)?)", name).group(1))
            for name in f.keys() if re.match(r"t\d+(\.\d+)?$", name)
        ])
        t_values.sort()
    return t_values


# ====================================================
# FUNCTION: initial_continue
# Initializes or continues a simulation
# ====================================================


def u0_continue(N, D, MODEL_TYPE):
    '''
    Initial Condition from the end of a deterministic density
    Args:
        N:

    Returns:

    '''
    # Open DK File
    base_folder = f'../Data/Model0/{MODEL_TYPE}/p{nx}'
    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/_N{N}_D{D:.4f}/dat.h5'
    with h5py.File(file_path, 'r') as f:
        time_ = time_hdf5(file_path)  # Export time array from simulation
        # print("Highest t-value:", max_t)
        rho = f[f't{time_[-1]}'][:]
        f.close()
    return rho


# ====================================================
# FUNCTION: kernel_2d
# Creates a 2D kernel function
# ====================================================
def kernel_2d(epsilon, R, X, Y, bounds_x, bounds_y):
    """
    Creates a 2D exponential kernel function.

    Parameters:
    - epsilon: Small parameter.
    - R: Scaling factor.
    - X, Y: Coordinate grids.
    - bounds_x, bounds_y: Boundary conditions.

    Returns:
    - Computed kernel values.
    """
    # Center of the grid
    x0 = (bounds_x[0] + bounds_x[1]) / 2
    y0 = (bounds_y[0] + bounds_y[1]) / 2

    # Compute the distance from the center
    distance = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

    # Compute the 2D cubic exponential kernel
    fij_2d = np.exp(-((distance / R) ** 3))

    # Normalize the kernel (optional)
    # fij_2d /= np.trapz(np.trapz(fij_2d, x, axis=0), y, axis=0)

    # Plot the kernel (optional)
    # plt.imshow(fij_2d, extent=[bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]], origin='lower', cmap='viridis')
    # plt.colorbar(label='Kernel Value')
    # plt.title(f'2D Cubic Exponential Kernel (R = {R})')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    return fij_2d * dx * dy


# ====================================================
# FUNCTION: pseudo_step_2d
# Performs a pseudospectral time-step operation in 2D
# This is for the Deterministic DK
# ====================================================
def pseudo_step_2d(u, dt, D, kx_, ky_, m2, n, nstps):
    '''
    Numerical integration of Deterministic Dean-Kawasaki
    Args:
        u:
        dt:
        D:
        kx_:
        ky_:
        m2:
        n:
        nstps:

    Returns:

    '''
    # Compute the 2D Fourier transform of u
    u_hat = fftpack.fft2(u)
    for _ in range(nstps):
        # ===========
        # NON LOCAL TERM COMPUTATION
        # Convolve ∇V with u in Fourier space
        conv_x_hat = grad_V_x_hat * u_hat  # (∂V/∂x) * u
        conv_y_hat = grad_V_y_hat * u_hat  # (∂V/∂y) * u

        # Transform back to real space
        conv_x = fftpack.ifftshift(fftpack.ifft2(conv_x_hat))
        conv_y = fftpack.ifftshift(fftpack.ifft2(conv_y_hat))

        # # Multiply by u in real space
        u_conv_x = u * conv_x  # u * (∂V/∂x * u)
        u_conv_y = u * conv_y  # u * (∂V/∂y * u)

        # Compute the divergence in Fourier space
        u_conv_x_hat = fftpack.fft2(u_conv_x)
        u_conv_y_hat = fftpack.fft2(u_conv_y)

        div_x_hat = -1j * kx_ * u_conv_x_hat  # ∂/∂x (u * (∂V/∂x * u))
        div_y_hat = -1j * ky_ * u_conv_y_hat  # ∂/∂y (u * (∂V/∂y * u))

        # Sum the components of the divergence
        non_local_hat = div_x_hat * dt + div_y_hat * dt
        # ===========
        # DIFFUSION TERM
        diffusion_term = diff_coef * u_hat

        # ===========
        # Euler-Maruyama in Fourier space
        u_hat += diffusion_term + non_local_hat
        u = fftpack.ifft2(u_hat).real  # Transform back to real space

    return u


# ====================================================
# FUNCTION: stochastic_step_2d
# Performs a pseudospectral time-step operation in 2D
# This is for the Stochastic DK
# ====================================================
def stochastic_step_2d(u, dt, D, kx_, ky_, nstps=10):
    # Compute the 2D Fourier transform of u
    u_hat = fftpack.fft2(u)
    for _ in range(nstps):
        # ===========
        # NON LOCAL TERM COMPUTATION
        # Convolve ∇V with u in Fourier space
        conv_x_hat = grad_V_x_hat * u_hat  # (∂V/∂x) * u
        conv_y_hat = grad_V_y_hat * u_hat  # (∂V/∂y) * u

        # Transform back to real space
        conv_x = fftpack.ifftshift(fftpack.ifft2(conv_x_hat))
        conv_y = fftpack.ifftshift(fftpack.ifft2(conv_y_hat))

        # # Multiply by u in real space
        u_conv_x = u * conv_x  # u * (∂V/∂x * u)
        u_conv_y = u * conv_y  # u * (∂V/∂y * u)

        # Compute the divergence in Fourier space
        u_conv_x_hat = fftpack.fft2(u_conv_x)
        u_conv_y_hat = fftpack.fft2(u_conv_y)

        div_x_hat = -1j * kx_ * u_conv_x_hat  # ∂/∂x (u * (∂V/∂x * u))
        div_y_hat = -1j * ky_ * u_conv_y_hat  # ∂/∂y (u * (∂V/∂y * u))

        # Sum the components of the divergence
        non_local_hat = div_x_hat * dt + div_y_hat * dt
        # ===========
        # DIFFUSION TERM
        diffusion_term = diff_coef * u_hat
        # ===========
        # NOISE TERM
        # noise_x = jump * compute_divergence(u, dx, dy)/dx
        # noise_term = fftpack.fft2(noise_x)
        up = np.maximum(0, u)  # Apply filter in square root
        B = np.random.normal(0, 1, size=(nx, ny))  # Create 2D gaussian noise
        # Take FFT

        # # Compute Fourier derivatives separately in x and y directions
        Fx = np.sqrt(dt / (dx * dy)) * fftpack.fft2(np.sqrt(up) * B)  # Fourier transform
        dFx_dx = -1j * kx_ * Fx  # Fourier derivative in x
        dFx_dy = -1j * ky_ * Fx  # Fourier derivative in y

        # Sum the components to get divergence
        noise_term = (dFx_dx + dFx_dy) * jump
        # ===========

        # Euler-Maruyama in Fourier space
        u_hat += diffusion_term + non_local_hat + noise_term  # * mask
        u = fftpack.ifft2(u_hat).real  # Transform back to real space

    return u


# ====================================================
# FUNCTION: Interpretation
# Check The Ito x Stratonovich Interpretation
# ====================================================
def interpretatation(u, dt, diff_coef, jump, kx_, ky_):
    # Compute the 2D Fourier transform of u
    u_hat = fftpack.fft2(u)

    for _ in range(10):  # ===========
        # NON LOCAL TERM COMPUTATION
        # Convolve ∇V with u in Fourier space
        conv_x_hat = grad_V_x_hat * u_hat  # (∂V/∂x) * u
        conv_y_hat = grad_V_y_hat * u_hat  # (∂V/∂y) * u

        # Transform back to real space
        conv_x = fftpack.ifftshift(fftpack.ifft2(conv_x_hat))
        conv_y = fftpack.ifftshift(fftpack.ifft2(conv_y_hat))

        # # Multiply by u in real space
        u_conv_x = u * conv_x  # u * (∂V/∂x * u)
        u_conv_y = u * conv_y  # u * (∂V/∂y * u)

        # Compute the divergence in Fourier space
        u_conv_x_hat = fftpack.fft2(u_conv_x)
        u_conv_y_hat = fftpack.fft2(u_conv_y)

        div_x_hat = -1j * kx_ * u_conv_x_hat  # ∂/∂x (u * (∂V/∂x * u))
        div_y_hat = -1j * ky_ * u_conv_y_hat  # ∂/∂y (u * (∂V/∂y * u))

        # Sum the components of the divergence
        non_local_hat = div_x_hat * dt + div_y_hat * dt
        # ===========
        # DIFFUSION TERM
        diffusion_term = diff_coef * u_hat
        # ===========
        # NOISE TERM

        up = np.maximum(0, u)  # Apply filter in square root
        B = np.random.normal(0, 1, size=(nx, ny))  # Create 2D gaussian noise
        # Take FFT
        noise = -1j * (kx_ + ky_) * fftpack.fft2(np.sqrt(up) * B)
        gnDw_real = fftpack.ifft2(noise).real
        gnpDw_real = fftpack.ifft2(-1j * (kx_ + ky_) * fftpack.fft2(0.5 * B / np.sqrt(up + 1e-10))).real

        correction_realI = 0.5 * gnDw_real * gnpDw_real * (1 - dt / B ** 2) * jump ** 2 * dt / dx ** 2
        correction_realS = 0.5 * gnDw_real * gnpDw_real * (1) * jump ** 2 * dt / dx ** 2
        correction_f = fftpack.fft2(correction_realI)
        correction_f1 = fftpack.fft2(correction_realS)

        # ===========

        # print('Correction:', correction_realI- correction_realS)
        print('Difussion:', abs(np.mean(diffusion_term)), 'Ito:', abs(np.mean(correction_f)))
        print('Ito-Str:', abs(np.mean(correction_realI) - np.mean(correction_realS)))
        print('Ito-Str:', abs(np.mean(correction_f) - np.mean(correction_f1)))
        print('Diff - Ito:', abs(np.mean(diffusion_term) - np.mean(correction_f)))
        print('Nonlocal:', abs(np.mean(non_local_hat)))
        u_hat += diffusion_term + non_local_hat  + noise * jump * np.sqrt(dt / dx ** 2)  # * mask
        u = fftpack.ifft2(u_hat).real  # Transform back to real space


# ====================================================
# ====================================================

"""
===========================================
    Simulation Parameters Configuration
===========================================
"""

# Set seed for reproducibility
seed = 3
np.random.seed(seed)

# -----------------------
# Biological Parameters
# -----------------------
eps = 0.0333  # Kernel amplitude
R = 0.1  # Interaction radius

# Number of particles
N_PARTICLES = int(sys.argv[3])

# -----------------------
# Domain Parameters
# -----------------------
nx, ny = 100, 100  # Grid resolution
bounds = np.array([0, 1])
L = bounds[1] - bounds[0]  # System size

# Define spatial grid
x = np.linspace(*bounds, nx + 1)[:-1]
y = np.linspace(*bounds, ny + 1)[:-1]
X, Y = np.meshgrid(x, y, indexing="ij")

# Grid spacing
dx = L / nx
dy = L / ny

rho0 = N_PARTICLES / L ** 2  # Initial density
nr = rho0 * R  # Noise Normalization
# -----------------------
# Diffusion Parameter
# -----------------------

D = float(sys.argv[1]) / 1000
REAL_DIFFUSION = float(sys.argv[1]) * (eps * rho0)
# Compute wave numbers
kx = fftpack.fftfreq(nx, L / nx) * 2 * np.pi
ky = fftpack.fftfreq(ny, L / ny) * 2 * np.pi
kx_, ky_ = np.meshgrid(kx, ky, indexing="ij")

# Interaction Kernel (Fourier Space)
M2 = kernel_2d(eps, R, X, Y, bounds, bounds)
v_hat = fftpack.fft2(M2)  # Fourier transform of kernel

# Compute gradient of V in Fourier space
grad_V_x_hat = -1j * kx_ * v_hat  # ∂V/∂x
grad_V_y_hat = -1j * ky_ * v_hat  # ∂V/∂y

# Model type (deterministic by default)
MODEL_TYPE = sys.argv[2]
"""
===========================================
        Simulation Setup & Storage
===========================================
"""

# Create main results directory
MAIN_DIRECTORY = "dean_results2D"
os.makedirs(MAIN_DIRECTORY, exist_ok=True)

# Automatically create subdirectory based on NX value
LATTICE_DIR = f"{MAIN_DIRECTORY}/{MODEL_TYPE}/p{nx}"
os.makedirs(LATTICE_DIR, exist_ok=True)

# Define path for saving results
RESULT_PATH = f"{LATTICE_DIR}/_N{N_PARTICLES}_D{D * 1000:.4f}"
os.makedirs(RESULT_PATH, exist_ok=True)

# Create HDF5 file for storing results
h5file = h5py.File(f"{RESULT_PATH}/dat.h5", "w")

"""
===========================================
    Initial Conditions & Gaussian Profile
===========================================
"""


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
# u0 = fftpack.ifftshift(fftpack.fft2(fgaussian(kx_, ky_ ))).real

# Create a Gaussian initial condition centered at x=0
# width = 5  # Width/spread of the Gaussian
# amplitude = 1.0 / (width * np.sqrt(2 * np.pi))  # Height of the Gaussian
# u0 = amplitude * np.exp(-((X- L/2) ** 2+ (Y- L/2) ** 2) / (2 * width ** 2))

# Homogeneous + Fluctuations
# u0 = np.zeros((nx, ny))
# np.random.seed(0)
# u0[:] = 1e0 + 1e-3 * np.random.rand(nx, ny)
#
# Continuation from previous simulation
u0 = u0_continue(N_PARTICLES, 0.81, MODEL_TYPE)
plt.imshow(u0)
plt.show()
plt.close()

u = np.copy(u0)
umean = np.copy(u0)
index = np.where(u == np.max(u))[0]

# -----------------------
# Time Parameters
# -----------------------
T = 10  # simulation duration
dt = 0.001
t = np.arange(0, T + dt, dt)
nt = len(t)

# Create arrays to store results
time_ = [t[0]]  # Time History
density_ = [np.mean(abs(u))]  # Density History
h5file.create_dataset(f"t{t[0]}", data=u)  # Save initial configuration in file
# h5file.close()


# Coefficients
diff_coef = -D * (kx_ ** 2 + ky_ ** 2) * dt
jump = np.sqrt(2 * D / rho0)

# -----------------------
# Display Simulation Parameters
# -----------------------
print("Simulation Parameters:")
print("\nBiological Parameters:")
print(f"Real Diffusion (D): {REAL_DIFFUSION}, Normalized Diffusion: {D}")
print(f"Number of Particles (N): {N_PARTICLES}")
print(f"Interaction Radius (R): {R}")
print(f"Kernel Amplitude (ε): {eps}")

print("\nDomain Parameters:")
print(f"Bounds: {bounds}")
print(f"Domain Length (L): {L}")
print(f"Grid Resolution: {nx} x {ny}")
print(f"Grid Spacing: dx={dx}, dy={dy}")
print(f'Time Step (dt): {dt}')
print(f"Initial Density: {rho0}")
print(f'Stochastic Jump: {jump}')
print(f'Model Tipy: {MODEL_TYPE}')

print(f'Density Error: Nh*h ={N_PARTICLES * dx * dy}')


# ====================================================
# FUNCTION: normal_integration
# integrate the system
# ====================================================
def normal_integration(nt, u):
    count = 0
    for n in tqdm(range(1, nt)):
        # compute the next time step
        if MODEL_TYPE == 'det':
            u = pseudo_step_2d(u, dt, D, kx_, ky_, M2, n, 30)  # Deterministic DK
        else:
            u = stochastic_step_2d(u, dt, D, kx_, ky_, 30)  # Pseudospectral Cornalba

        # break
        # Wx = reader.get_noise(n)
        # u = stochastic_step_2d2(u, dt, D, kx_, ky_, 10, Wx)

        # # # Store data for density plot
        if n % 5000 == 0:
            # time_.append(round(t[n], 7))
            # density_.append(np.mean(u))
            h5file.create_dataset(f"t{t[n]}", data=u)
        # if n % 10000 == 0:
        #     # Create file for saving results
        #     h5file = h5py.File(f"{path}/dat2.h5", "w")
        #     time_.append(round(t[n], 7))
        #     density_.append(np.mean(u))
        #     h5file.create_dataset(f"t{round(t[n], 7)}", data=u)

        if n % 1000 == 0:
            ### Plot Figures (Optional)
            plt.subplots(1, 2, figsize=(25, 10))
            plt.subplots_adjust(wspace=0.05)
            plt.subplot(1, 2, 1)
            plt.imshow(u.T, cmap="gnuplot", origin="lower", extent=np.concatenate((bounds, bounds)))
            plt.colorbar(ticks=np.linspace(np.min(u), np.min(u) + 0.9 * (np.max(u) - np.min(u)), 7))

            plt.xlim([bounds[0], bounds[1]])
            plt.title(f"t = {10 * t[n]:0.3f};")
            plt.subplot(1, 2, 2)

            plt.plot(x, u[:, int(ny / 2)], '.-', c="k", label="u")
            # plt.plot(time_, density_, c="k")

            #     # #     # Optional: Save the figure
            plt.savefig(f"{RESULT_PATH}/density_{count + 1}.png")
            plt.close()
            # count += 1
    # Before Finish simulation Write the data
    # h5file.create_dataset("time", data=time_)
    # h5file.create_dataset("tot_density", data=density_)
    # Explicitly close the file
    h5file.close()


# ====================================================
def storage_optimation(snaps):
    # Open DK File
    base_folder = f'dean_results2D/{MODEL_TYPE}/p{nx}'
    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/_N{N_PARTICLES}_D{D * 1000:.4f}/dat.h5'
    with h5py.File(file_path, 'r') as f:
        t = time_hdf5(file_path)
        rho = f[f't{t[-1]}'][:]
        f.close()
    u = np.copy(rho)
    h5file = h5py.File(f"{RESULT_PATH}/dat.h5", "w")  #
    for n in tqdm(range(1, snaps)):
        # compute the next time step

        u = stochastic_step_2d(u, dt, D, kx_, ky_, 1)  # Pseudospectral Cornalba

        # Save each time step
        h5file.create_dataset(f"t{round(n * dt, 4)}", data=u)


# ====================================================
# ====================================================
# ====================================================
# ====================================================
'''
THIS SECTION PEFORMS THE EULER-MARUYAMA INTEGRATION WHERE THERE IS NO DIFFERENCE BETWEEN 
THE MULTIPLICATIVE STOCHASTIC PRESCRIPTION
'''
normal_integration(nt, u)
if MODEL_TYPE == 'noise':
    storage_optimation(1000)
else:
    pass


# interpretatation(u, dt, D, jump, kx_, ky_)
