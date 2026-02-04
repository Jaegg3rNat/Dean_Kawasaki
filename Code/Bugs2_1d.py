"""
Bugs2_1d.py

Author: Nathan Silvano
Date: [2025-May-DD]
Description:
    This script simulates a 1D system of particles that diffuse according to the local number of particles density
    D(rho).
    .
"""
import h5py  # For saving data in HDF5 format
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations
import sys, os, re
from numba import cuda, jit, njit, prange, float32
from tqdm import tqdm  # For progress bars


#This is for computing the number of particles inside a radius R
# @njit
# def compute_Nr_numba(x, R, L, N):
#     Nr = np.zeros(N, dtype=np.int32)
#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 continue
#             dx = x[i] - x[j]
#             # Periodic boundary conditions (minimum image)
#             dx -= L * np.round(dx / L)
#             if abs(dx) < R:
#                 Nr[i] += 1
#     # Prevent zero (for division safety)
#     for i in range(N):
#         if Nr[i] == 0:
#             Nr[i] = 1
#     return Nr



def compute_Nr_with_xr(x, nx, bounds):
    """
    Compute the number of particles in each bin of a fixed mesh grid and associate
    each particle with the Nr value of its bin.

    Parameters:
        x (np.ndarray): Array of particle positions (size N).
        nx (int): Number of bins in the mesh grid.
        bounds (tuple): Tuple specifying the domain bounds (min, max).

    Returns:
        tuple:
            Nr (np.ndarray): Array of size nx with the number of particles in each bin.
            xr (np.ndarray): Array of size N with the Nr value for each particle.
    """
    # Define the bin edges for the mesh grid
    bin_edges = np.linspace(bounds[0], bounds[1], nx + 1)

    # Use np.histogram to count particles in each bin
    Nr, _ = np.histogram(x, bins=bin_edges)

    # Determine the bin index for each particle
    bin_indices = np.digitize(x, bin_edges) - 1

    # Ensure bin indices are within valid range
    bin_indices = np.clip(bin_indices, 0, nx - 1)

    # Map each particle to its corresponding Nr value
    xr = Nr[bin_indices]

    return Nr, xr


#This is the wave profile function of the initial condition. Following the analitycal solution
def Wave_profile(x, t0, A, D0):
    xmax = A * t0 ** (1 / 3)
    if np.abs(x) < xmax:
        return (1 / (6 * D0 * t0)) * (A ** 2 * t0 ** (2 / 3) - x ** 2)
    else:
        return 0.0

#This function creates the samples of initial condtion inside the wave profile
def sample_particles(n_particles, t0, A, D0):
    xmax = A * t0 ** (1 / 3)
    umax = (A ** 2) / (6 * D0 * t0 ** (1 / 3))  # Maximum of u(x)

    particles = []
    while len(particles) < n_particles:
        x_candidate = np.random.uniform(-xmax, xmax)
        u_candidate = Wave_profile(x_candidate, t0, A, D0)
        # Accept with probability u_candidate / umax
        if np.random.uniform(0, umax) < u_candidate:
            particles.append(x_candidate)
    return np.array(particles)


# -------------------------------------------------------------------------------------
# Initialization of System
# -------------------------------------------------------------------------------------

# Domain definition (1D system)
bounds = np.array([-5, 5])  # Square domain
L = bounds[1] - bounds[0]  # System size
dt = 1e-5  # Time step

T = 0.02  # Simulation duration


run = int(sys.argv[1])  #File identifier for many simulations

D = 1  # Diffusion coefficient ( D0 in the paper)
N = 10000  # Number of particles

# phantom grid
nx = 1024  #number of points in the grid
dx = L / nx

# R = 2.5*dx  #Radius R (only use for the numba version of the Nr computation)
# -------------------------------------------------------------------------------------
# Directory Setup to save the data
# -------------------------------------------------------------------------------------

# Define a directory to store results
main_directory = "Bugs_1d"  # Main directory for results
if not os.path.exists(main_directory):
    os.makedirs(main_directory)  # Create directory if it doesn't exist

# Automatically create subdirectory based on initial configuration
lattice_size_dir = f"{main_directory}/WaveFront"  # Subdirectory for Gaussian initialization
if not os.path.exists(lattice_size_dir):
    os.makedirs(lattice_size_dir)

# Create path for saving results based on N and D values
path = f"{lattice_size_dir}/_N{N}"  # Path for specific simulation
if not os.path.exists(path):
    os.makedirs(path)
# Create file for saving results
h5file = h5py.File(f"{path}/dat{run}.h5", "w")
# -------------------------------------------------------------------------------------
# Initial Conditions
# -------------------------------------------------------------------------------------

#Initial time
t0 = 1e-5
t = np.arange(t0, T + dt, dt)  # Array of time steps
nt = len(t)  # Number of time steps


A = (9/2) ** (1 / 3)  # Amplitude parameter from analytical solution

# plt.plot(t, A*t**(1/3),'.-', label='Wave Profile')
# plt.plot(t0,A*t0**(1/3),'ro', label='Initial Position')
# plt.show()
# plt.close()

# Sample initial particle positions
x = sample_particles(N, t0, A, D)
# print("Sampled particles:", x)


# Nr, xr = compute_Nr_with_xr(x, nx, bounds)
# print(Nr)
# print(xr)


# Store initial positions (optional)
h5file.create_dataset(f"t{t0}", data=x)
# --.Plotting.--
# plt.hist(x, bins=int(nx/2), density=True)
# plt.xlabel("x")
# plt.xlim(-L/2, L/2)
# plt.ylabel("Particle density")
# plt.show()
# plt.close()


# Define histogram range and number of bins
hist_range = (-5, 5)
num_bins = nx
# Compute histogram
counts, bin_edges = np.histogram(x, bins=num_bins, range=hist_range, density=True)
# Compute bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# print(counts)
# Stack bin centers (x) and counts (y) into two columns
hist_data = np.column_stack((bin_centers, counts))

# # Save to ASCII file
# # np.savetxt(f"histogram_t{t0}.txt", hist_data, fmt="%.6e", header="x y")
# plt.plot(bin_centers, counts, label=f't={t0:.2f}')
# xx = np.linspace(-5, 5, 1000)
# # Evaluate the wave profile at each x (vectorize for speed)
# Wave_profile_vec = np.vectorize(Wave_profile)
# y_vals = Wave_profile_vec(xx, t0, A, D)
# plt.plot(xx,y_vals,'.-', label=f'Wave Profile at t={t0:.2f}')
# plt.show()


# -----------------------
# Display Simulation Parameters
# -----------------------
print("Simulation Parameters:")
print(f"Domain Length (L): {L}")
print(f'Simulation Time: {T}')
print(f'Time Step (dt): {dt}')
# print(f'Number of Time Steps (nt): {nt}')
print(f'Diffusion Coefficient (D): {D}')
print(f'Number of Particles (N): {N}')
print(f'size grid: {dx}')
# print(f'Local Radius (R): {R}')
print(f'Amplitude: {A}')

# -------------------------------------------------------------------------------------
# Run Simulation
# -------------------------------------------------------------------------------------
# Create arrays to store results
vec_time = [t0]  # Array to store time steps
# Simulation Loop
for i in tqdm(range(2,nt)):
    # Nr = compute_Nr_numba(x, R, L, N) (this is the version using radius R) replace xr for Nr
    _, xr = compute_Nr_with_xr(x, nx, bounds)
    # print(xr)
    # Update particle positions
    jump = np.sqrt(2 * D * dt * xr)
    x += jump * np.random.normal(size=x.shape) #(gaussian noise)
    # Apply periodic boundary conditions (PBC)
    x = bounds[0] + (x - bounds[0]) % (bounds[1] - bounds[0])
    vec_time.append(t0+dt*i)
    h5file.create_dataset(f"t{t0+dt*i}", data=x)
# Save time array to HDF5 file
h5file.create_dataset("time", data=vec_time)
# Explicitly close the HDF5 file
h5file.close()