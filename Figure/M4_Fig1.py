import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from matplotlib import rc
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import os
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

# Choose the font for the figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


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
    power_spectrum = np.abs(fft_vals) ** 2 / ( N**2)  # Normalize by total number of points

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


def compute_S(snaps, N, D, bins,base_folder):
    powerList = np.zeros(int(bins / 2 - 1))
    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/_N{N}_D{D:.4f}/dat1.h5'
    with h5py.File(file_path, 'r') as f:
        # f.visititems(print_name)
        time_ = time_hdf5(file_path)  # Export time array from simulation
        for i in range(snaps):
            group_name = f"t{time_[-(i + 1)]}"  # Example group
            if group_name in f:
                group = f[group_name]  # Access the group
                x = group["x"][:]  # Read the dataset for x
                y = group["y"][:]  # Read the dataset for y
            # Compute 2D histogram
            counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], density=True)
            # plt.imshow(counts, cmap='hot')
            # plt.show()
            # plt.close()
            # Compute power spectrum
            k_values, power_spectrum = compute_power_spectrum_2d(counts, N, bins)
            # print(k_values[0:10])
            powerList += power_spectrum
        powerList /= snaps
        return powerList, k_values

# Primary gradient: dark to light blue
# primary_colors = ['#003C6C', '#0072B2', '#56B4E9']

# Secondary gradient: dark to light orange
# secondary_colors = ['#8C510A', '#E69F00', '#F6C141']

secondary_colors = ['#004949', '#009292', '#FFB20F']
primary_colors = ['#000000', '#4ECDC4', '#FF6B6B']

# ==================================================================================
# Begin Figure
def fig():
    # Begin figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=900, figsize=(4.2, 4.3),
                                   sharex=True)  # Height doubled for two rows

    xc = (0.82 / 1000) / 0.1 ** 2
    line4 = ax1.axvline(x=xc, color='k',linestyle='dotted', label=r'$\tilde{D}^d_c=$'f'${round(xc, 4)}$',
                alpha=0.4)
    ax2.axvline(x=xc, color='k',linestyle='dotted', label=r'$\tilde{D}^d_c=$'f'${round(xc, 4) }$',
                alpha=0.4)
    xc = (0.878 / 1000) / 0.1 ** 2
    # line5 = ax2.axvline(x=xc, color=primary_colors[1], linestyle='-.', label=r'$\tilde{D}^s_c=$'f'${round(xc, 4)}$',
    #             alpha=0.4)
    xc = (0.892 / 1000) / 0.1 ** 2
    # ax1.axvline(x=xc, color=primary_colors[1], linestyle='-.', label=r'$\tilde{D}^s_c=$'f'${round(xc, 4) }$',
    #             alpha=0.4)
    ax1.axhline(y=0.0, color='k', linestyle='--', alpha=0.6)
    ax2.axhline(y=0.0, color='k', linestyle='--', alpha=0.6)
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)

    '''
    10K particles determinitic model
    '''

    # Open DET File
    base_folder = '../Data/Model0/det/p100'
    # Open the HDF5 file and inspect contents
    p_list = np.arange(0.7, 1., 0.002)
    s_list = []
    snaps, bins, N = 1, 100, 10000
    for p in tqdm(p_list):
        # compute smax
        smax = compute_smax(base_folder, N, p, snaps, bins)
        # save smax
        s_list.append(smax)

    dd = (np.array(p_list) / 1000) / 0.1 ** 2
    ax1.plot(dd, s_list, '-', lw =3, color=primary_colors[0], label=f'deterministic DK')
    # ====================================================================
    # Open DET File
    base_folder = '../Data/Model0/det/p100_cont'
    # Open the HDF5 file and inspect contents
    p_list = np.array(
        [0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.97, 0.972, 0.974, 0.976, 0.978,
         0.98, 0.99,
         1.0])

    print(f"p_list: {p_list}")
    s_list = []
    for p in tqdm(p_list):
        smax = compute_smax(base_folder, N, p, 1, bins)

        s_list.append(smax)

    dd = (np.array(p_list) / 1000) / 0.1 ** 2
    ax1.plot(dd, s_list, '-',lw =3, color=primary_colors[0])
    '''
    DETERMINISTIC 2K PARTICLES
    '''
    base_folder = '../Data/Model0/det/p100'
    # Open the HDF5 file and inspect contents
    p_list = np.arange(0.7, 1., 0.002)
    s_list = []
    snaps, bins, N = 1, 100, 20000
    for p in tqdm(p_list):
        # compute smax
        smax = compute_smax(base_folder, N, p, snaps, bins)
        # save smax
        s_list.append(smax)

    dd = (np.array(p_list) / 1000) / 0.1 ** 2
    line3, = ax2.plot(dd, s_list, '-', lw =3, color=primary_colors[0], label=f'deterministic DKTE')
    # # ====================================================================
    # ====================================================================
    # Open DET File
    base_folder = '../Data/Model0/det/p100_cont'
    # Open the HDF5 file and inspect contents
    p_list = np.array(
        [0.82, 0.84, 0.85, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.972, 0.974, 0.976, 0.98, 0.99,
         1.0])

    print(f"p_list: {p_list}")
    s_list = []
    for p in tqdm(p_list):
        smax = compute_smax(base_folder, 20000, p, 1, bins)

        s_list.append(smax)

    dd = (np.array(p_list) / 1000) / 0.1 ** 2
    ax2.plot(dd, s_list, '-', lw =3,color=primary_colors[0])
    #
    # ====================================================================
    # Open NOISE File
    base_folder = '../Data/Model0/noise/p100'
    # Open the HDF5 file and inspect contents
    p_list = np.arange(0.7, 1, 0.002)
    # p_list = np.append(p_list, np.arange(0.966, 1, 0.002))

    s_list = []
    N = 20000
    snaps, bins = 100, 100
    for i, p in enumerate(p_list):
        # print(i, p)
        smax = compute_smax(base_folder, N, p, snaps, bins)

        s_list.append(smax)

    dd = np.array(p_list) / 1000 / 0.1 ** 2
    ax2.plot(dd[1::4], s_list[1::4], '^-',markersize = 5, color=primary_colors[1], label='DK')
    #
    # # ====================================================================
    # ====================================================================
    # Open NOISE File
    base_folder = '../Data/Model0/noise/p100'
    # Open the HDF5 file and inspect contents
    p_list = np.arange(0.7, 1, 0.002)

    # print(f"p_list: {p_list}")
    s_list = []
    snaps, bins = 10, 100
    for i, p in enumerate(p_list):
        # print(i, p)
        smax = compute_smax(base_folder, 10000, p, snaps, bins)

        s_list.append(smax)

    dd = np.array(p_list) / 1000 / 0.1 ** 2
    line2, = ax1.plot(dd[::4], s_list[::4], '^-',markersize = 5, color=primary_colors[1], label='DKTE')
    #
    # ====================================================================
    # ====================================================================
    # Open NOISE File
    base_folder = '../Data/Model0/noise/p100_cont'
    # Open the HDF5 file and inspect contents
    p_list = np.array(
        [0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.964, 0.965, 0.966, 0.967, 0.968, 0.969, 0.97, 0.98])
    #
    s_list = []
    snaps, bins = 100, 100
    for i, p in enumerate(p_list):
        # print(i, p)
        smax = compute_smax(base_folder, 10000, p, snaps, bins)

        s_list.append(smax)

    dd = np.array(p_list) / 1000 / 0.1 ** 2
    ax1.plot(dd, s_list, '^-', mfc='none',markersize = 5, color=primary_colors[1])
    #
    # ====================================================================
    # ====================================================================
    # Open NOISE File
    base_folder = '../Data/Model0/noise/p100_cont'
    # Open the HDF5 file and inspect contents
    p_list = np.array(
        [0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])
    #
    s_list = []
    snaps, bins = 100, 100
    for i, p in enumerate(p_list):
        # print(i, p)
        smax = compute_smax(base_folder, 20000, p, snaps, bins)

        s_list.append(smax)

    dd = np.array(p_list) / 1000 / 0.1 ** 2
    ax2.plot(dd, s_list, '^-', mfc='none',markersize = 5, color=primary_colors[1])
    #
    # ====================================================================
    # Open Particles File
    N = 10000
    base_folder = '../Data/Model0/Bugs_2d/Model4'
    p_list2 = np.array([0.7,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.852,0.854,0.858,0.86,0.862,0.864,0.866,0.868,0.87,0.872,0.874,0.88, 0.9])
    s_peak = []
    for D in tqdm(p_list2):
        power_spectrum, k_values = compute_S(100, N, D, 100,base_folder)
        #     # Find max characteristic frequency after k=0
        k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
        s_peak.append(s_max)

    dd = np.array(p_list2) / 1000 / 0.1 ** 2
    ax1.plot(dd, s_peak, '.-',markersize = 8, color=primary_colors[2], label='Particles')
    # ====================================================================
    '''
    /////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////Bugs Continuation/////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////
    '''
    # ====================================================================
    # Open Particles File
    N = 10000
    base_folder = '../Data/Model0/Bugs_2d/Model4_cont'
    p_list2 = np.array([0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98])
    s_peak = []
    for D in tqdm(p_list2):
        power_spectrum, k_values = compute_S(100, N, D, 100,base_folder)
        #     # Find max characteristic frequency after k=0
        k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
        s_peak.append(s_max)

    dd = np.array(p_list2) / 1000 / 0.1 ** 2
    ax1.plot(dd, s_peak, '.-',markersize = 8, mfc='none', color=primary_colors[2])
    # ====================================================================
    # Open Particles File
    N = 20000
    base_folder = '../Data/Model0/Bugs_2d/Model4'

    p_list2 = np.array([0.7,0.75,0.82 ,0.85,0.86,0.87,0.88,0.89,0.9])

    s_peak = []
    for D in tqdm(p_list2):
        power_spectrum, k_values = compute_S(500, N, D, 100,base_folder)
        #     # Find max characteristic frequency after k=0
        k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
        s_peak.append(s_max)

    dd = np.array(p_list2) / 1000 / 0.1 ** 2
    line1,= ax2.plot(dd, s_peak, '.-',markersize = 8,  color=primary_colors[2], label='Particles')
    # ====================================================================
    # Open Particles File
    N = 20000
    base_folder = '../Data/Model0/Bugs_2d/Model4_cont'
    p_list2 = np.array([0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,.96,.97,.98])
    s_peak = []
    for D in tqdm(p_list2):
        power_spectrum, k_values = compute_S(100, N, D, 100,base_folder)
        #     # Find max characteristic frequency after k=0
        k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
        s_peak.append(s_max)

    dd = np.array(p_list2) / 1000 / 0.1 ** 2
    ax2.plot(dd, s_peak, '.-',markersize = 8, mfc='none', color=primary_colors[2])
    # ====================================================================


    # --------------------------------------------------

    ax2.set_xlabel(r'Normalized Diffusion, $\tilde{D}$', fontsize=14)
    ax1.set_ylabel(r'$\tilde{S}_{max}$', fontsize=14)
    ax2.set_ylabel(r'$\tilde{S}_{max}$', fontsize=14)

    # ax1.grid(True)

    # plt.axvline(x=0.892/1000, color='r', linestyle='--',alpha = 0.4)
    # ax1.legend(fontsize = 8)
    # ax2.legend(fontsize = 8)
    ax1.text(0.065, 0.058, s=r'($\textbf{a}$)', fontweight='black', fontsize=15,
             )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
    ax2.text(0.065, 0.014, s=r'($\textbf{b}$)', fontweight='black', fontsize=15,
             )# bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))

    # After creating your plots...
    fig.legend(handles=[line3, line2,line1,line4],
               loc='upper center',
               bbox_to_anchor=(0.5, 0.),
               frameon=True,
               fancybox=True,
               shadow=False,
               ncol=2,
               fontsize=10)
    # ==================================================
    # ==================================================
    # if N == 10000:
    #     plt.savefig('Fig_1.png', bbox_inches="tight")  #
    #     plt.savefig('Fig_1.pdf', bbox_inches="tight")  #
    # else:
    plt.savefig('M4_fig1.png', bbox_inches="tight")  #
    plt.savefig('M4_fig1.pdf', bbox_inches="tight")  #


fig()
