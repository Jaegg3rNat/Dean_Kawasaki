"""
SPDE2.py

Author: Nathan Silvano
Date: [2025-May-DD]
Description:
    .
"""

import h5py  # For saving data in HDF5 format
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations
import sys, os, re
import h5py
from numba import cuda, jit, njit, prange, float32
from tqdm import tqdm  # For progress bars
from scipy.fftpack import fft, ifft, ifftshift, fftfreq  # Fast Fourier Transform operations

##########################
def f_det_dk(ud):
    '''
    Fourier update of PDE
    Args:
        ud: deterministic field in real space

    Returns:

    '''
    uhat = fft(ud)                  # Fourier transform
    u2hat = fft(ud ** 2)            # Fourier transform of u^2
    diff = -D0 * kx ** 2 * u2hat    # Fourier space second derivative
    uhat += diff * dt               # Update in Fourier space
    return ifft(uhat).real          # Inverse Fourier transform to get back to real space

##########################
def f_st_dk(us):
    '''
    Fourier update of SPDE
    Args:
        us: stochastic field in real space

    Returns:

    '''
    uhat = fft(us)                  # Fourier transform
    u2hat = fft(us * us**p)            # Fourier transform of u^2
    diff = -D0 * kx ** 2 * u2hat    # Fourier space second derivative

    #create gaussian noise in Real Space
    eta = np.random.normal(0, 1, size=nx)
    prod_hat = fft(np.sqrt(us**2) * eta)
    noise = -1j * kx * prod_hat     # Stochastic term in Fourier space
    uhat += diff * dt + np.sqrt(2 * D0 * dt / (rho0 * dx)) * noise  # Update in Fourier space
    return ifft(uhat).real  # Inverse Fourier transform to get back to real space

##########################
def det_dk(u):
    '''
    Deterministic diffusion update using finite differences.
    Args:
        u: deterministic field

    Returns:

    '''
    # Diffusion
    uu = u*(u)**p
    u2 = second_derivative(uu, nx)

    u += D0 * u2 * dt / (dx ** 2)
    return u

##########################
def stoc_dk(u):
    '''
    Stochastic diffusion update using finite differences.
    Args:
        u:

    Returns:

    '''
    # Diffusion
    uu =u*  (u) ** p
    u2 = second_derivative(uu, nx)

    eta = np.random.normal(0, 1, size=nx)

    up = np.maximum(0, uu)
    product = np.sqrt(up) * eta
    noise = first_derivate(product,nx)

    u += D0 * u2 * dt / (dx ** 2) + noise * np.sqrt(2 * D0 * dt / (rho0 * dx))
    return u
#########################
##########################
@njit
def first_derivate(u, nx):
    du = np.empty_like(u)
    for i in range(nx):
        du[i] = (u[(i + 1) % nx] - u[(i - 1) % nx]) / (2 * dx)
    return du
@njit
def second_derivative(u, nx):
    """Compute the second derivative of u using finite differences."""
    d2u = np.empty_like(u)

    for i in range(nx):
        d2u[i] = u[(i - 1) % nx] - 2 * u[i] + u[(i + 1) % nx]
    return d2u
#############################
#############################




'''
Begin Parameters Definition
'''


t0 = 1e-5                       # Initial time
dt = 1* 1e-10                       # Time step size
T = 0.01                        # Final time
t = np.arange(t0, T + dt, dt)   # Array of time steps
nt = len(t)                     # Number of time steps
L = 2*np.pi# System size
nx = 1000                       #Number of grid points
dx = L / nx                     # Spatial step size
print(f"dx = {dx}, dt = {dt}, L = {L}, nx = {nx}, nt = {nt}")
p = 1

D0 = 1                          # Diffusion coefficient
N = 5000                        # Number of particles

A = (9* (2*D0)/ 2) ** (1 / 3)   # Amplitude parameter from analytical solution

x = np.linspace(-L / 2, L / 2, nx)  # Spatial grid points
kx = 2 * np.pi * fftfreq(nx, d=dx)  # Wave numbers for Fourier transform

#############
def analytical_solution(t0, A, x_array):
    """Compute the analytical solution for the wave profile."""
    results = []
    xmax = A * t0 ** (1 / 3)
    for x in x_array:
        if np.abs(x) < xmax:
            results.append((1 / (12 * D0 * t0)) * (A ** 2 * t0 ** (2 / 3) - x ** 2))
        else:
            results.append(0.0)
    return np.array(results)
#############

u0 = analytical_solution(t0, A, x)  # Initial condition at t0
rho0 = N / L                        # density of particles

ud = np.copy(u0)
us = np.copy(u0)

'''' Particles Functions Initialization'''
##################
# def compute_Nr_with_xr(x, nx, bounds):
#     """
#     Compute the number of particles in each bin of a fixed mesh grid and associate
#     each particle with the Nr value of its bin.
#
#     Parameters:
#         x (np.ndarray): Array of particle positions (size N).
#         nx (int): Number of bins in the mesh grid.
#         bounds (tuple): Tuple specifying the domain bounds (min, max).
#
#     Returns:
#         tuple:
#             Nr (np.ndarray): Array of size nx with the number of particles in each bin.
#             xr (np.ndarray): Array of size N with the Nr value for each particle.
#     """
#     # Define the bin edges for the mesh grid
#     bin_edges = np.linspace(bounds[0], bounds[1], nx + 1)
#
#     # Use np.histogram to count particles in each bin
#     Nr, _ = np.histogram(x, bins=bin_edges)
#
#     # Determine the bin index for each particle
#     bin_indices = np.digitize(x, bin_edges) - 1
#
#     # Ensure bin indices are within valid range
#     bin_indices = np.clip(bin_indices, 0, nx - 1)
#
#     # Map each particle to its corresponding Nr value
#     xr = Nr[bin_indices]
#
#     return Nr, xr
# ##############
# #This is the wave profile function of the initial condition. Following the analitycal solution
# def Wave_profile(x, t0, A, D0):
#     xmax = A * t0 ** (1 / 3)
#     if np.abs(x) < xmax:
#         return (1 / (12 * D0 * t0)) * (A ** 2 * t0 ** (2 / 3) - x ** 2)
#     else:
#         return 0.0
# ################
# #This function creates the samples of initial condtion inside the wave profile
# def sample_particles(n_particles, t0, A, D0):
#     xmax = A * t0 ** (1 / 3)
#     umax = (A ** 2) / (12 * D0 * t0 ** (1 / 3))  # Maximum of u(x)
#
#     particles = []
#     while len(particles) < n_particles:
#         x_candidate = np.random.uniform(-xmax, xmax)
#         u_candidate = Wave_profile(x_candidate, t0, A, D0)
#         # Accept with probability u_candidate / umax
#         if np.random.uniform(0, umax) < u_candidate:
#             particles.append(x_candidate)
#     return np.array(particles)
# #####################


# Sample initial particle positions
# xp0 =sample_particles(N, t0, A, D0)
# print("Sampled particles:", x)
# ############
# plt.plot(x, ud, '.-')
# plt.hist(xp0, bins=int(nx), density=True)
# plt.show()
# plt.close()
# ############

steps = 18000000
counts = 1000
snaps = int(steps / (counts-1))  # Number of snapshots to take
# xfrontd = np.zeros(counts)
# xfrontp = np.zeros(counts)
# xfronts = np.zeros(counts)
time_array = [t0]
# xfrontp[0],xfronts[0],xfrontd[0] = A * t0 ** (1 / 3),A * t0 ** (1 / 3),A * t0 ** (1 / 3)




# # print(np.shape(udmean), np.shape(usmean))
# hist_range = (-L/2, L/2)  # or specify your desired range
# bounds = np.array([-L/2, L/2])  # Square domain


run =20
job = 2
# Define a directory to store results
main_directory = "data"  # Main directory for results
if not os.path.exists(main_directory):
    os.makedirs(main_directory)  # Create directory if it doesn't exist

path = f"{main_directory}"  # Path for the current job
for j in range(run):
    jj = 0

    usmean = np.zeros((len(u0), counts))
    udmean = np.zeros((len(u0), counts))

    # xmean = np.zeros((len(xp0), counts))


    ud = np.copy(u0)
    us = np.copy(u0)
    udmean[:, 0] = ud
    usmean[:, 0] = us
    # xp =  xp0
    # xmean[:, 0] = xp0  # Initialize particle positions
    # xmean[:, 0] = xp  # Initialize particle positions
    h5file = h5py.File(f"{path}/job{job}_dat{j+1}.h5", "w")  # HDF5 file for storing data
    for i in tqdm(range(1, steps+1)):
        try:# Compute the analytical solution at each time step
            ud = det_dk(ud)  # Update u using the diffusion equation
            us = stoc_dk(us)
        except OverflowError:
            print(f"Overflow encountered at step {i}. Stopping the loop.")
            break
        # ud = f_det_dk(ud)  # Update stochastic diffusion
        # us = f_st_dk(us)  # Update deterministic diffusion



        # threshold = 1e-3  # Set a small threshold for "nonzero"
        # first_non_zero_index = np.argmax(ud > threshold)
        # front_position = x[first_non_zero_index-1]  # assuming x is your spatial grid
        # xfrontd[i] += (abs(front_position))
        #
        # threshold = 1e-3  # Set a small threshold for "nonzero"
        # first_non_zero_index = np.argmax(us > threshold)
        # front_position = x[first_non_zero_index-1]  # assuming x is your spatial grid
        # xfronts[i] += abs(front_position)

        '''
        Particles
        '''
        # _, Nr = compute_Nr_with_xr(xp, nx, bounds)
        # Nr = compute_Nr_numba(x, dx, L, N)
        # print(xr)
        # Update particle positions
        # jump = np.sqrt(2 * D0 * dt * ((Nr/N)*(L/dx))**p) # Jump size based on Nr
        # xp += jump * np.random.normal(size=xp.shape)  # (gaussian noise)
        if i % snaps == 0 and jj < counts-1:
            jj+=1
            time_array.append(t0+ i*dt)
            # print(f"Time step {i}, jj = {jj}, time = {t0 + i * dt:.4f}")

            usmean[:,jj] += us
            udmean[:,jj] += ud
            # xmean[:, jj] += xp  # Store particle positions


        # Apply periodic boundary conditions (PBC)
        # xp = bounds[0] + (xp - bounds[0]) % (bounds[1] - bounds[0])

        # density, bin_edges = np.histogram(xp, bins=int(nx), range=hist_range, density=True)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        #
        #
        # threshold = 1e-7  # Set a small threshold for "nonzero"
        # first_non_zero_index = np.argmax(density > threshold)
        # front_position = bin_centers[first_non_zero_index-1]
        # xfrontp[i] += abs(front_position)

        # if i % 100000 == 0:
        #     plt.plot(x, us, '.-')
        #     plt.plot(x, ud, '.-')
        #     # plt.hist(xp, bins=int(nx), density=True,alpha = 0.5)
        #     plt.plot(x, analytical_solution(t0+ dt* i, A, x), '.-', alpha = .5, color = 'r')
        #     plt.title(f't={10000* dt +t0:.4f}')
        #     # plt.ylim([-0.5, 2])
        #     plt.xlim([-0.8, 0.8])
        #     plt.show()
        #     plt.close()
        #
        # if i % 5000 == 0:
        #     plt.plot(x, us, '.-')
        #     plt.plot(x, ud, '.-')
        #     plt.plot(x, analytical_solution(t0 + dt * i, A, x), '.-', alpha=.5, color='r')
        #     plt.title(f't={np.mean(ud):.4f}')
        #     # plt.ylim([-0.5, 2])
        #     plt.xlim([-0.1, 0.1])

        # plt.show()
    result = {
        'ud': udmean,
        'us': usmean,
        'time': time_array,
    }

    # Save results to HDF5 file
    for key, value in result.items():
        h5file.create_dataset(key, data=value)
    h5file.close()  # Close the HDF5 file after writing




# Divide all inputs of usmean and udmean by number of simulations
# usmean[:,1:-1] /= run
# udmean[:,1:-1] /= run
#
# plt.plot(x, usmean[:,10000], '.-')
# plt.plot(x, udmean[:,10000], '.-')
#
# plt.plot(x, usmean[:,60000], '.-')
# plt.plot(x, udmean[:,60000], '.-')
# plt.plot(x, usmean[:,-1000], '.-')
# plt.plot(x, udmean[:,-1000], '.-')
#
# plt.title(f't={np.mean(ud):.4f}')
# # plt.ylim([-0.5, 2])
# plt.xlim([-0.1, 0.1])
# plt.show()
#
# xfrontd[1:] /= run
# xfronts[1:] /= run
# xfrontp[1:] /= run
#
#
# ''''
# Fitting curve
# '''
#
# from scipy.optimize import curve_fit
#
# # Power-law model
# def power_law(x, A, B,):
#     return A * (x ** B)
#
# # Fit the model
# params, covariance = curve_fit(power_law, t[1:steps], xfrontp[1:], p0=[1, 1])
# A, B = params
#
# print(f"Ap = {A:.3f}, Bp = {B:.3f},fit:{1/3}")
#
# param, cov = curve_fit(power_law, t[1:steps], xfronts[1:], p0=[1, 1])
# A_s, B_s = param
# print(f"As = {A_s:.3f}, Bs = {B_s:.3f},fit:{1/3}")
#
# param, cov = curve_fit(power_law, t[1:steps], xfrontd[1:], p0=[1, 1])
# A_d, B_d = param
# print(f"Ad = {A_d:.3f}, Bd = {B_d:.3f},fit:{1/3}")
#
# t_rescaled = t * (D0 * L**2)
# xfrontd_re = np.array(xfrontd)*L
# xfronts_re = np.array(xfronts)*L
#
# xanal = [A * (t0 + dt * i*10) ** (1 / 3) for i in range(int(steps/10))]
#
#
# plt.plot(t_rescaled[:steps], xfrontd_re, '.-', label='Deterministic ')
# plt.plot(t_rescaled[:steps], xfronts_re, '.-', label='Stochastic ')
# plt.plot(t[:steps], xfrontp, '.-', label='Particles ')
# plt.plot(t[:steps: 10], xanal, '.-', label='Analytical Solution', alpha=0.5, color='r')
# plt.legend()
# plt.show()


