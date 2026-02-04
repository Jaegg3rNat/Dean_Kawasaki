"""
SPDE1.py

Author: Nathan Silvano
Date: [2025-May-DD]
Description:
    .
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rc
import sys

# Choose the font for the figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters
def SPDE1(number):
    # ---------------------------
    # Parameters
    # ---------------------------
    N = 300          # Number of intervals → N+1 grid points
    L = 1.0
    dx = L / N
    x = np.linspace(0, L, N + 1)
    Nparticles = int(sys.argv[2])  # From command-line
    D = 0.1 + 1 * x

    print('Nh =', Nparticles * dx)

    # Initial condition: Gaussian
    mu = 0.5
    sigma = 0.05
    rho = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Normalize if desired
    rho /= np.sum(rho) * dx

    plt.plot(x, rho, color='r', lw=2, label='Initial condition')

    # ---------------------------
    # Time Setup
    # ---------------------------
    T = 0.5
    dt = 1e-6
    t = np.arange(0, T + dt, dt)
    nt = len(t)
    model = 'noise' #sys.argv[1]  # From command-line, 'det' or 'noise'
    # ---------------------------
    # Time Evolution
    # ---------------------------
    if model == 'det':
        # for n in tqdm(range(nt)):
        #     # Trho = D(x) * rho(x)
        #     Trho = D * rho
        #
        #     # --- Diffusion term with Neumann BC using ghost padding ---
        #     # Ghost padding: replicate edge values to enforce zero-gradient (Neumann BCs)
        #     Trho_ext = np.pad(Trho, pad_width=1, mode='edge')
        #     d2 = (Trho_ext[2:] - 2 * Trho_ext[1:-1] + Trho_ext[:-2]) / dx**2
        #
        #     # --- Euler-Maruyama Update ---
        #     rho += dt * d2
        for n in tqdm(range(nt)):
            Trho = D * rho
            Trhop = np.maximum(0, Trho)

            ##############################################
            # Compute second derivative with Neumann BCs
            d2 = np.zeros_like(rho)
            # Interior points
            d2[1:-1] = (Trho[2:] - 2 * Trho[1:-1] + Trho[:-2]) / dx ** 2
            # Boundaries
            d2[0] = 2 * (Trho[1] - Trho[0]) / dx ** 2
            d2[-1] = 2 * (Trho[-2] - Trho[-1]) / dx ** 2
            # Update (Euler-Maruyama)
            rho += dt * d2

            # print('rho', np.mean(rho))
    # # ---------------------------
    # # Plot Final Result
    # # ---------------------------
    # plt.plot(x, rho, color='b', lw=2, label='Final')
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("rho")
    # plt.title("Density evolution with Neumann BCs (ghost padding)")
    # plt.show()



    else:

        # for n in tqdm(range(nt)):
        #     Trho = D * rho
        #     Trhop = np.maximum(0, Trho)
        #
        #     ##############################################
        #     # Compute second derivative with Neumann BCs
        #     d2 = np.zeros_like(rho)
        #     # Interior points
        #     d2[1:-1] = (Trho[2:] - 2 * Trho[1:-1] + Trho[:-2]) / dx ** 2
        #     # Boundaries
        #     d2[0] = 2 * (Trho[1] - Trho[0]) / dx ** 2
        #     d2[-1] = 2 * (Trho[-2] - Trho[-1]) / dx ** 2
        #     ##############################################
        #     # # --- Stochastic part (Itô noise) ---
        #     noise = np.zeros_like(rho)
        #
        #     # Generate Gaussian noise
        #     xi = np.random.normal(0, 1, size=N + 1)  # N+1 points (same grid as rho)
        #
        #     # Itô evaluation: sqrt(T*rho) at current time
        #     sqrt_Trho = np.sqrt(Trhop)
        #
        #     # Centered difference for noise term
        #     noise[1:-1] = (sqrt_Trho[2:] * xi[2:] - sqrt_Trho[:-2] * xi[:-2]) / (2 * dx)
        #
        #     # Reflective boundaries (no flux)
        #     noise[0] = 0.0  # J(0) = 0
        #     noise[-1] = 0.0  # J(L) = 0
        #
        #     ##########################################################
        #     # Update (Euler-Maruyama)
        #     rho += dt * d2 + np.sqrt(2* dt / (Nparticles * dx)) * noise

        ######################
        # # --- Plot every 1000 steps with gray-to-black gradient ---
        # if n % 10000 == 0:
        #     # Normalized progress (0 to 1)
        #     progress = n / nt
        #     # RGB value: from light gray (0.7,0.7,0.7) to black (0,0,0)
        #     rgb_value = 0.7 * (1 - progress)
        #     color = (rgb_value, rgb_value, rgb_value)
        #     plt.plot(x, rho, color=color)

        for n in tqdm(range(nt)):
        #     # Trho = D(x) * rho(x)
        #     Trho = D * rho
        #     Trhop = np.maximum(0, Trho)
        # #
        # #     # --- Diffusion term with Neumann BC using ghost padding ---
        # #     # Ghost padding: replicate edge values to enforce zero-gradient (Neumann BCs)
        # #     Trho_ext = np.pad(Trho, pad_width=1, mode='edge')
        # #     d2 = (Trho_ext[2:] - 2 * Trho_ext[1:-1] + Trho_ext[:-2]) / dx ** 2
        # #
        #     # --- Stochastic term (Itô noise) ---
        #     xi = np.random.normal(0, 1, size=N + 1)
        #     sqrt_Trho = np.sqrt(2 *Trhop)
        #     sqrt_Trho_xi_ext = np.pad(sqrt_Trho*xi, pad_width=1, mode='edge')
        #     # xi_ext = np.pad(xi, pad_width=1, mode='edge')
        #
        #     # Compute centered stochastic flux derivative
        #     # noise = (sqrt_Trho_xi_ext[2:]  - sqrt_Trho_xi_ext[:-2]) / (2 * dx)
        # #
        #
        #
        #     # --- Diffusion term with reflection ghost cells ---
        #     Trho_ext = np.zeros(N + 2)
        #     Trho_ext[1:-1] = Trho
        #     Trho_ext[0] = Trho[1]  # left ghost = reflect
        #     Trho_ext[-1] = Trho[-2]  # right ghost = reflect
        #     d2 = (Trho_ext[2:] - 2 * Trho_ext[1:-1] + Trho_ext[:-2]) / dx ** 2
        #
        #     # --- Noise term with reflection ghost cells ---
        #     noise_ext = np.zeros(N + 2)
        #     noise_ext[1:-1] = sqrt_Trho * xi
        #     noise_ext[0] = noise_ext[1]  # reflect
        #     noise_ext[-1] = noise_ext[-2]  # reflect
        #     noise = (noise_ext[2:] - noise_ext[:-2]) / (2 * dx)
        #
        #     # --- Euler-Maruyama Update ---
        #     rho += dt * d2 + np.sqrt( dt /(Nparticles * dx)) * noise

            # Transport density
            Trho = D * rho
            Trhop = np.maximum(0, Trho)

            # --- Diffusion term with reflection ghost cells ---
            Trho_ext = np.zeros(N + 3)  # we need N+3 because original has N+1 points
            Trho_ext[1:-1] = Trho
            Trho_ext[0] = Trho[1]  # left ghost = reflect
            Trho_ext[-1] = Trho[-2]  # right ghost = reflect
            d2 = (Trho_ext[2:] - 2 * Trho_ext[1:-1] + Trho_ext[:-2]) / dx ** 2

            # --- Stochastic term (Itô noise) ---
            xi = np.random.normal(0, 1, size=N + 1)
            sqrt_Trho = np.sqrt(2 * Trhop)

            noise_ext = np.zeros(N + 3)
            noise_ext[1:-1] = sqrt_Trho * xi
            noise_ext[0] = noise_ext[1]  # reflect left
            noise_ext[-1] = noise_ext[-2]  # reflect right

            noise = (noise_ext[2:] - noise_ext[:-2]) / (2 * dx)

            # --- Euler-Maruyama Update ---
            rho += dt * d2 + np.sqrt(dt / (Nparticles * dx)) * noise
        print('rho', np.mean(rho))
    '''
    Save Data
    '''
    data = np.column_stack((x, rho))

    # Save to ASCII file
    if model == 'det':

        np.savetxt(f'../Data/Model1/L2/Model1det_N{Nparticles}.txt', data, fmt='%.6e', header='x y')
    else:
        np.savetxt(f'../Data/Model1/L2/Model1noise_N{Nparticles}_{number}.txt', data, fmt='%.6e', header='x y')

# plt.xlabel(r'$x$', fontsize=14)
# plt.ylabel(r'$\rho(x,t)$', fontsize=14)
# plt.title(r'Diffusion with spatially varying $D(x)$',fontsize = 14)
# plt.legend()
# plt.show()
SPDE1(sys.argv[1])