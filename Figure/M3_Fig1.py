import numpy as np
import re
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy.stats import stats
from tqdm import tqdm
import os
from matplotlib import rc


# Choose the font for the figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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


# =================================================================================
# Fourier Transform of Fields
# =================================================================================

def compute_power_spectrum_2d(field, N, bins):
    """Compute the 2D power spectrum of the field and return the frequencies and power spectrum."""
    L = 1
    nx = bins
    dx = L / nx
    fft_vals = np.fft.fft2(field)
    power_spectrum = np.abs(fft_vals) ** 2 / (nx*nx*N)  # Normalize by total number of points

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

def compute_smax(base_folder, N, p, snaps,nx):
    file_path = f'{base_folder}/_N{N}_p{p:.1f}/dat.h5'
    power_mean = np.zeros(nx //2-1)
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
def compute_S(snaps, N, p, bins):
    powerList = np.zeros(int(bins / 2 - 1))

    base_folder = '../Data/Model3/Bugs'

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/_N{N}_p{p:.2f}/dat1.h5'
    x = np.zeros(N)
    y = np.zeros(N)

    with h5py.File(file_path, 'r') as f:
        # f.visititems(print_name)
        time_ = time_hdf5(file_path)  # Export time array from simulation
        # print(time_[-1])
        for i in range(snaps):
            group_name = f"t{int(time_[-(i + 1)])}"  # Example group
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

primary_colors = ['#000000', '#4ECDC4', '#FF6B6B']
'''
#######################################################################################################################
#######################################################################################################################
####################################### BEGIN THE FIGURE ENVIRONMENT #######################################
#######################################################################################################################
#######################################################################################################################
'''

# Begin figure
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=900, figsize=(5.5, 5.8),
                               sharex=True)  # Height doubled for two rows


xc = 7.6
line1 = ax1.axvline(x=7.6, color='k', linestyle='dotted',lw= 2, label='Critical $p_c$',alpha = 0.7)
ax2.axvline(x=xc, color='k', linestyle='dotted', lw=2, label=f'$p_c={xc}$',alpha = 0.7)
# ax1.axhline(y=0, color='k', linestyle='--',alpha = 0.6)
# ax2.axhline(y=0, color='k', linestyle='--',alpha = 0.6)


# Open DET File
base_folder = '../Data/Model3/det/p128'
# Open the HDF5 file and inspect contents
p_list = np.arange(5., 9.9, 0.1)
N, snaps, bins = 10000, 1, 128
s_list = []
for p in tqdm(p_list):
    smax = compute_smax(base_folder, N, p, snaps,bins)
    s_list.append(smax)

dd = np.array(p_list)

ax1.plot(dd, s_list, '-',lw=2,color = primary_colors[0], label=f'deterministic DK')
line2,= ax2.plot(dd, [x/2 for x in s_list], '-',lw=2,color = primary_colors[0], label='deterministic DKTE')
# # ====================================================================



# Open Noise File
base_folder = '../Data/Model3/noise/p128'
# Open the HDF5 file and inspect contents
p_list = np.arange(5., 9.9, 0.1)
# p_list = np.append(p_list, np.array([8.6]))


# print(f"p_list: {p_list}")
s_list = []
for i,p in enumerate(p_list):
    # print(i,p)
    smax = compute_smax(base_folder, 10000, p, 50, bins)

    s_list.append(smax)

dd = np.array(p_list)
line3, = ax1.plot(dd, s_list, '^-', color=primary_colors[1], label='DKTE')

# ====================================================================
# # ====================================================================
#
# # Open Noise File
# base_folder = '../Data/Model3/noise/p128'
# # Open the HDF5 file and inspect contents
# p_list = np.arange(5., 9.9, 0.1)
# # p_list = np.append(p_list, np.array([8.6]))
#
#
# # print(f"p_list: {p_list}")
# s_list = []
# for i,p in enumerate(p_list):
#     print(i,p)
#     smax = compute_smax(base_folder, 20000, p, 50, bins)
#
#     s_list.append(smax)
#
# dd = np.array(p_list)
# ax2.plot(dd, s_list, '^-', color=primary_colors[1], label=f'Stochastic')
# #
# # # # ====================================================================

# Open the HDF5 file and inspect contents
p_list2 = np.array([6,6.5,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.,8.3,8.4,8.5,8.8,9.,9.5,9.9])

s_peak = []
for p in tqdm(p_list2):
    power_spectrum, k_values = compute_S(1, 10000, p, 128)

    k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
    # #
    s_peak.append(s_max)

dd = np.array(p_list2)
line4, =ax1.plot(dd, s_peak, '.-',markersize = 8, color=primary_colors[2], label='Particles')

# # # ====================================================================
#
# # Open the HDF5 file and inspect contents
# p_list2 = np.array([6.5,7.,7.1,7.2,7.3,7.4,7.5,7.6])
#
# s_peak = []
# for i,p in enumerate(p_list2):
#     print(p,i)
#     power_spectrum, k_values = compute_S(1, 20000, p, 100)
#
#     k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
#     # #
#     s_peak.append(s_max)
#
# dd = np.array(p_list2)
# print(s_peak)
# ax2.plot(dd, s_peak, '.-',markersize = 8, color=primary_colors[2], label='Particles')
# dx = 1 / 128
# print(f'Nh^2 = {N *dx**2}')



# Open the HDF5 file and inspect contents
p_list2 = np.array([5,5.5,6,6.5,6.8,7.,7.3,7.4,7.6,7.8,8,8.5,8.8,9,9.2,9.5,9.8])

s_peak = []
for i,p in enumerate(p_list2):
    print(p,i)
    power_spectrum, k_values = compute_S(1, 20000, p, 128)

    k_max, s_max = find_max_characteristic_frequency_2d(np.array(k_values), np.array(power_spectrum))
    # #
    s_peak.append(s_max)

dd = np.array(p_list2)
# print(s_peak)
ax2.plot(dd, s_peak, '.-',markersize = 8, color=primary_colors[2], label='Particles')
# ====================================================================

# Open Noise File
base_folder = '../Data/Model3/noise/p128'
# Open the HDF5 file and inspect contents
p_list = np.array([6.5,6.6,6.7,6.8,6.9,7.,7.1,7.2,7.3,7.4,7.6,7.8,8.,8.2,8.5,8.8,9,9.2,9.5,9.8])
# p_list = np.append(p_list, np.array([8.6]))


# print(f"p_list: {p_list}")
s_list = []
for i,p in enumerate(p_list):
    print(i,p)
    smax = compute_smax(base_folder, 20000, p, 50, 128)

    s_list.append(smax)

dd = np.array(p_list)
ax2.plot(dd, s_list, '^-', color=primary_colors[1], label=f'DK')
#
# # # ====================================================================




ax2.set_xlabel(r'$p$', fontsize=17)
ax1.set_ylabel(r'$\tilde{S}_{max}$', fontsize=17)
ax2.set_ylabel(r'$\tilde{S}_{max}$', fontsize=17)
# ax1.legend(fontsize = 10)
# ax2.legend(fontsize = 10)
ax1.tick_params(axis='both', labelsize=15)
ax2.tick_params(axis='both', labelsize=15)
ax2.set_ylim([-0.001,0.06])
ax1.set_ylim([-0.001,0.06])

ax1.text(3.8, 0.068, s=r'($\textbf{a}$)', fontweight='black', fontsize=17,
         )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
ax2.text(3.8, 0.066, s=r'($\textbf{b}$)', fontweight='black', fontsize=17,
         )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))


# After creating your plots...
fig.legend(handles =[line2, line3 ,line4,line1],
          loc='upper center',
          bbox_to_anchor=(0.5, 0.),
          frameon=True,
          fancybox=True,
          shadow=False,
          ncol=2,
           fontsize = 12)


plt.savefig('M3_Fig1.png', bbox_inches="tight")#
plt.savefig('M3_Fig1.pdf', bbox_inches="tight") #