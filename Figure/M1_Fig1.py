import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Choose the font for the figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
primary_colors = ['#000000', '#4ECDC4', '#FF6B6B']
# set figure format
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, dpi=900, figsize=(12, 6), sharey=True)  # Height doubled for two rows
#
# #------------------------------------------------------------
# # READ DETERMINISTIC DATA
# N = 1000
# data = np.loadtxt(f'../Data/Model1/Model1det_N{N}.txt', skiprows=1)
# # Split the data into X and Y
# x = data[:, 0]  # First column as X
# y = data[:, 1]  # Second column as Y
# print(f'Mean Density for N={N}: {np.mean(y)}')
# # -----------------------------------------------------------
# # Parameters
# hist_range = (0, 1)
# bins = 300
#
# # READ STOCHASTIC DATA
# data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}.txt', skiprows=1)  # Split the data into X and Y
# # Compute histogram manually
# rho, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
# print(f'Mean Density for N={N}: {np.mean(rho)}')
# # -----------------------------------------------------------
# # Plot the data
# count = 0
# x1 = np.zeros_like(x)
# y1 = np.zeros_like(y)
#
# # READ STOCHASTIC DATA
# data1 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}.txt', skiprows=1)
# # Split the data into X and Y
# x1 += data1[:, 0]  # First column as X
# y1 += data1[:, 1]  # Second column as Y
# print(f'Mean Density for N={N}: {np.mean(y1)}')
#
#
#
# ax1.plot(x1[::2], y1[::2], marker='^', markersize=5, linestyle='-', color=primary_colors[1], label='DK')
# # ax1.hist(data2, bins=bins, range=hist_range, density=True, alpha=0.5,label = 'Particles',color = 'b')
# ax1.plot(bin_centers[::2], rho[::2], '.-', markersize=10, label='Particles', color=primary_colors[2], lw=2)
# ax1.plot(x, y, marker='none', linestyle='-', lw='3', color=primary_colors[0], label='deterministic DK')
#
#
# ######################################################
# N = 2000
#
# #------------------------------------------------
# data = np.loadtxt(f'../Data/Model1/Model1det_N{N}.txt', skiprows=1)
# # Split the data into X and Y
# x = data[:, 0]  # First column as X
# y = data[:, 1]  # Second column as Y
# print(f'Mean Density for N={N}: {np.mean(y)}')
# # -----------------------------------------------
# # -----------------------------------------------------------
# # Parameters
# hist_range = (0, 1)
# bins = 300
#
# # READ STOCHASTIC DATA
# data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}.txt', skiprows=1)  # Split the data into X and Y
# # Compute histogram manually
# rho, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
# print(f'Mean Density for N={N}: {np.mean(rho)}')
# # -----------------------------------------------------------
# # Plot the data
# count = 0
# x1 = np.zeros_like(x)
# y1 = np.zeros_like(y)
#
# # READ STOCHASTIC DATA
# data1 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}.txt', skiprows=1)
# # Split the data into X and Y
# x1 += data1[:, 0]  # First column as X
# y1 += data1[:, 1]  # Second column as Y
# print(f'Mean Density for N={N}: {np.mean(y1)}')
#
# ax2.plot(x1[::2], y1[::2], marker='^', markersize=5, linestyle='-', color=primary_colors[1], label='DK')
# # Plot as a curve instead of bars
# ax2.plot(bin_centers[::2], rho[::2], '.-', markersize=10, label='Particles', color=primary_colors[2], lw=2)
# # ax2.hist(data2, bins=bins, range=hist_range, density=True, alpha=0.5,label = 'Particles',color = 'b')
# ax2.plot(x, y, marker='none', linestyle='-', lw='3', color=primary_colors[0], label='deterministic DK')
# #-------------------
# #--------------------
# ax2.set_xlabel(r'System length, $x$', fontsize=17)
# ax1.set_xlabel(r'System length, $x$', fontsize=17)
# # ax2.ylabel(r'Normalized Density Distribution, $\rho(x)$')
# ax1.set_ylabel(r'Normalized Density Distribution, $\rho(x)$', fontsize=17)
# # plt.title('X-Y Data Plot')
# ax1.set_xlim(0, 0.5)
# ax2.set_xlim(0, 0.5)
# ax1.set_ylim(-0.1,9)
# ax1.legend(fontsize=12, loc='upper left')
# ax2.legend(fontsize=12, loc='upper left')
# # ax1.grid(True)
# # ax2.grid(True)
# ax2.tick_params(axis='both', which='major', labelsize=15, length=6, width=1.5)
# ax1.tick_params(axis='both', which='major', labelsize=15, length=6, width=1.5)
#
# ax1.text(-0.08, 9.5, s='(a)', fontweight='black', fontsize=25,
#          )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
# ax2.text(-0.06, 9.5, s='(b)', fontweight='black', fontsize=25,
#          )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
#
#
#
# ############################################################
# ############################################################
# ############################################################
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
#
# # # ----- Inset for ax2 -----
# axins1 = inset_axes(
#     ax1,
#     width="50%",
#     height="30%",
#     loc='center right',
#     bbox_to_anchor=(0, 0.15, 1, 1),  # (x0, y0, width, height) shift
#     bbox_transform=ax1.transAxes,   # coordinates in axes fraction
#     borderpad=1
# )
# # axins2.plot(x1[::2], y1[::2], marker='^', markersize=3, linestyle='-',lw= 0.5, color=primary_colors[1])
# # axins2.plot(bin_centers[::2], rho[::2], '.-', markersize=5,  color=primary_colors[2], lw=0.5)
#
# N = 1000
# count = 0
#
# # First load one file to get shape
# data0 = np.loadtxt(f'../Data/Model1/L2/Model1noise_N{N}_1.txt', skiprows=1)
# n_points = data0.shape[0]
#
# # Store all realizations
# y_all = np.zeros((200-1, n_points))  # rows = realizations, cols = points
# x_ref = data0[:, 0]  # x values are the same for all, assume
#
# # Loop over files
# for i in range(1, 200):
#     data = np.loadtxt(f'../Data/Model1/L2/Model1noise_N{N}_{i}.txt', skiprows=1)
#     y_all[count, :] = data[:, 1]
#     count += 1
#
# # Compute mean and std across realizations (axis=0 = across runs)
# y_mean = np.mean(y_all, axis=0)
# y_std = np.std(y_all, axis=0)
#
# print(f"Mean Density for N={N}: {np.mean(y_mean)}")
#
#
#
# # -----------------------------------------------------------
# # Parameters
# hist_range = (0, 1)
# bins = 300
# count = 0
#
# # allocate array for all realizations
# rho_all = np.zeros((100-1, bins))
#
# for i in range(1, 100):
#     data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}_{i}.txt', skiprows=1)
#     # Compute histogram (normalized density=True)
#     counts1, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
#     rho_all[count, :] = counts1
#     count += 1
#
# # compute mean and std across realizations
# rho_mean = np.mean(rho_all, axis=0)
# rho_std  = np.std(rho_all, axis=0)
#
# # bin centers for plotting
# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#
# print(f"Mean density from particles for N={N}: {np.mean(rho_mean)}")
#
# # Plot mean + std
#
#
# axins1.fill_between(x_ref, y_mean-y_std, y_mean+y_std, alpha=0.5,  color=primary_colors[1])
# axins1.fill_between(bin_centers, rho_mean-rho_std, rho_mean+rho_std,  alpha=0.5,color=primary_colors[2])
# axins1.plot(x_ref[::4], y_mean[::4] ,marker='^', markersize=3, linestyle='-',lw= 0.5,  color=primary_colors[1])
# axins1.plot(bin_centers[::4], rho_mean[::4], '.-', markersize=5,  color=primary_colors[2], lw=0.5)
# axins1.plot(x, y, color=primary_colors[0], linestyle='-', lw=1)
#
#
# # Optional: Zoomed-in region
# axins1.set_xlim(0., 0.9)
# axins1.set_ylim(0.1, 4)
# axins1.tick_params(axis='both', which='major', labelsize=6)
# #=========================================================
# # ------------------------------------------------------------
# # N = 2000
# # # Plot the data
# # count = 0
# # x1 = np.zeros_like(x)
# # y1 = np.zeros_like(y)
# # for i in range(1, 100):
# #     number = i
# #     count += 1
# #     # READ STOCHASTIC DATA
# #     data1 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}_{number}.txt', skiprows=1)
# #     # Split the data into X and Y
# #     x1 += data1[:, 0]  # First column as X
# #     y1 += data1[:, 1]  # Second column as Y
# #
# # x1 /= count
# # y1 /= count
# # print(f'Mean Density for N={N}: {np.mean(y1)}')
# # print(np.std(y1))
# # -----------------------------------------------------------
# # # Parameters
# # hist_range = (0, 1)
# # bins = 300
# # count = 0
# # rho = np.zeros(300)
# # for i in range(1, 100):
# #     number = i
# #     count += 1
# #     # READ STOCHASTIC DATA
# #     data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}_{number}.txt', skiprows=1)  # Split the data into X and Y
# #     # Compute histogram manually
# #     counts1, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
# #     rho += counts1
# # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
# # rho /= count
# # print(f'Mean Density for N={N}: {np.mean(rho)}')
# # # ----- Inset for ax2 -----
# axins2 = inset_axes(
#     ax2,
#     width="60%",
#     height="40%",
#     loc='center right',
#     bbox_to_anchor=(0.1, 0.1, 1., 1),  # (x0, y0, width, height) shift
#     bbox_transform=ax2.transAxes,   # coordinates in axes fraction
#     borderpad=1
# )
# # axins2.plot(x1[::2], y1[::2], marker='^', markersize=3, linestyle='-',lw= 0.5, color=primary_colors[1])
# # axins2.plot(bin_centers[::2], rho[::2], '.-', markersize=5,  color=primary_colors[2], lw=0.5)
#
# N = 2000
# count = 0
#
# # First load one file to get shape
# data0 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}_1.txt', skiprows=1)
# n_points = data0.shape[0]
#
# # Store all realizations
# y_all = np.zeros((200-1, n_points))  # rows = realizations, cols = points
# x_ref = data0[:, 0]  # x values are the same for all, assume
#
# # Loop over files
# for i in range(1, 200):
#     data = np.loadtxt(f'../Data/Model1/L2/Model1noise_N{N}_{i}.txt', skiprows=1)
#     y_all[count, :] = data[:, 1]
#     count += 1
#
# # Compute mean and std across realizations (axis=0 = across runs)
# y_mean = np.mean(y_all, axis=0)
# y_std = np.std(y_all, axis=0)
#
# print(f"Mean Density for N={N}: {np.mean(y_mean)}")
# print(np.mean(y_std))
#
#
#
# # -----------------------------------------------------------
# # Parameters
# hist_range = (0, 1)
# bins = 300
# count = 0
#
# # allocate array for all realizations
# rho_all = np.zeros((100-1, bins))
#
# for i in range(1, 100):
#     data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}_{i}.txt', skiprows=1)
#     # Compute histogram (normalized density=True)
#     counts1, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
#     rho_all[count, :] = counts1
#     count += 1
#
# # compute mean and std across realizations
# rho_mean = np.mean(rho_all, axis=0)
# rho_std  = np.std(rho_all, axis=0)
#
# # bin centers for plotting
# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#
# print(f"Mean density from particles for N={N}: {np.mean(rho_mean)}")
# print(np.mean(rho_std))
#
# # Plot mean + std
# # axins2.fill_between(x_ref, y_mean-y_std, y_mean+y_std, alpha=0.01,  color=primary_colors[1])
# axins2.fill_between(bin_centers, rho_mean-rho_std, rho_mean+rho_std,lw =1, alpha=0.6,color=primary_colors[2])
# axins2.plot(x_ref,y_mean-y_std, alpha=1, linestyle='-',lw= 1,  color=primary_colors[1])
# axins2.plot(x_ref,y_mean+y_std, alpha=1, linestyle='-',lw= 1,  color=primary_colors[1])
#
#
# axins2.plot(x_ref[::6], y_mean[::6] ,marker='^', markersize=5, linestyle='-',lw= 0.5,  color=primary_colors[1])
# axins2.plot(bin_centers[::6], rho_mean[::6], '.-', markersize=7,  color=primary_colors[2], lw=0.5)
# axins2.plot(x, y, color=primary_colors[0], linestyle='-', lw=1)
#
#
# # Optional: Zoomed-in region
# axins2.set_xlim(0., 0.9)
# axins2.set_ylim(0.1, 4)
# axins2.tick_params(axis='both', which='major', labelsize=6)
#
# axins1.set_xlabel(r'$x$', fontsize=12)
# axins1.set_ylabel(r'$\langle\rho(x)\rangle$', fontsize=12)
#
# axins2.set_xlabel(r'$x$', fontsize=12)
# axins2.set_ylabel(r'$\langle\rho(x)\rangle$', fontsize=12)



# set figure format
fig, ((ax1, ax2),(axins1,axins2)) = plt.subplots(nrows=2, ncols=2, dpi=900, figsize=(4.9, 3.9),sharex = True, sharey=True)  # Height doubled for two rows
plt.subplots_adjust(hspace=0.4)  # Increase space (default is around 0.2)

#------------------------------------------------------------
# READ DETERMINISTIC DATA
N = 1000
data = np.loadtxt(f'../Data/Model1/Model1det_N{N}.txt', skiprows=1)
# Split the data into X and Y
x = data[:, 0]  # First column as X
y = data[:, 1]  # Second column as Y
print(f'Mean Density for N={N}: {np.mean(y)}')
# -----------------------------------------------------------
# Parameters
hist_range = (0, 1)
bins = 300

# READ STOCHASTIC DATA
data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}.txt', skiprows=1)  # Split the data into X and Y
# Compute histogram manually
rho, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
print(f'Mean Density for N={N}: {np.mean(rho)}')
# -----------------------------------------------------------
# Plot the data
count = 0
x1 = np.zeros_like(x)
y1 = np.zeros_like(y)

# READ STOCHASTIC DATA
data1 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}.txt', skiprows=1)
# Split the data into X and Y
x1 += data1[:, 0]  # First column as X
y1 += data1[:, 1]  # Second column as Y
print(f'Mean Density for N={N}: {np.mean(y1)}')



line2, = ax1.plot(x1[::3], y1[::3], marker='^', markersize=5, linestyle='-', color=primary_colors[1], label='DKTE')
# ax1.hist(data2, bins=bins, range=hist_range, density=True, alpha=0.5,label = 'Particles',color = 'b')
line3, = ax1.plot(bin_centers[::3], rho[::3], '.-', markersize=6, label='Particles', color=primary_colors[2], lw=1)
line1, = ax1.plot(x, y, marker='none', linestyle='-', lw='1.5', color=primary_colors[0], label='deterministic DKTE')

######################################################
N = 2000

#------------------------------------------------
data = np.loadtxt(f'../Data/Model1/Model1det_N{N}.txt', skiprows=1)
# Split the data into X and Y
x = data[:, 0]  # First column as X
y = data[:, 1]  # Second column as Y
print(f'Mean Density for N={N}: {np.mean(y)}')
# -----------------------------------------------
# -----------------------------------------------------------
# Parameters
hist_range = (0, 1)
bins = 300

# READ STOCHASTIC DATA
data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}.txt', skiprows=1)  # Split the data into X and Y
# Compute histogram manually
rho, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
print(f'Mean Density for N={N}: {np.mean(rho)}')
# -----------------------------------------------------------
# Plot the data
count = 0
x1 = np.zeros_like(x)
y1 = np.zeros_like(y)

# READ STOCHASTIC DATA
data1 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}.txt', skiprows=1)
# Split the data into X and Y
x1 += data1[:, 0]  # First column as X
y1 += data1[:, 1]  # Second column as Y
print(f'Mean Density for N={N}: {np.mean(y1)}')

ax2.plot(x1[::3], y1[::3], marker='^', markersize=4, linestyle='-',lw= 0.8, color=primary_colors[1], label='DK')
# Plot as a curve instead of bars
ax2.plot(bin_centers[::3], rho[::3], '.-', markersize=6,lw= 0.8, label='Particles', color=primary_colors[2])
# ax2.hist(data2, bins=bins, range=hist_range, density=True, alpha=0.5,label = 'Particles',color = 'b')
ax2.plot(x, y, marker='none', linestyle='-', lw='1.5', color=primary_colors[0], label='deterministic DK')
######################################################


# #-------------------
# #--------------------
# ax2.set_xlabel(r'System length, $x$', fontsize=17)
# ax1.set_xlabel(r'System length, $x$', fontsize=17)
# # ax2.ylabel(r'Normalized Density Distribution, $\rho(x)$')
ax1.set_ylabel(r'$\rho(x)$', fontsize=12)
# # plt.title('X-Y Data Plot')
ax1.set_xlim(0, 0.5)
ax2.set_xlim(0, 0.5)
# ax1.set_ylim(-0.1,9)
# ax1.legend(fontsize=12, loc='upper left')
# ax2.legend(fontsize=12, loc='upper left')
# # ax1.grid(True)
# # ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)
#







'''' 
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''

from mpl_toolkits.axes_grid1.inset_locator import inset_axes



N = 1000
count = 0

# First load one file to get shape
data0 = np.loadtxt(f'../Data/Model1/L2/Model1noise_N{N}_1.txt', skiprows=1)
n_points = data0.shape[0]

# Store all realizations
y_all = np.zeros((200-1, n_points))  # rows = realizations, cols = points
x_ref = data0[:, 0]  # x values are the same for all, assume

# Loop over files
for i in range(1, 200):
    data = np.loadtxt(f'../Data/Model1/L2/Model1noise_N{N}_{i}.txt', skiprows=1)
    y_all[count, :] = data[:, 1]
    count += 1

# Compute mean and std across realizations (axis=0 = across runs)
y_mean = np.mean(y_all, axis=0)
y_std = np.std(y_all, axis=0)

print(f"Mean Density for N={N}: {np.mean(y_mean)}")



# -----------------------------------------------------------
# Parameters
hist_range = (0, 1)
bins = 300
count = 0

# allocate array for all realizations
rho_all = np.zeros((100-1, bins))

for i in range(1, 100):
    data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}_{i}.txt', skiprows=1)
    # Compute histogram (normalized density=True)
    counts1, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
    rho_all[count, :] = counts1
    count += 1

# compute mean and std across realizations
rho_mean = np.mean(rho_all, axis=0)
rho_std  = np.std(rho_all, axis=0)

# bin centers for plotting
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

print(f"Mean density from particles for N={N}: {np.mean(rho_mean)}")

# Plot mean + std

axins1.plot(x, y, color=primary_colors[0], linestyle='-', lw=2.5)
# axins1.fill_between(x_ref, y_mean-y_std, y_mean+y_std, alpha=0.7,   color=primary_colors[1])
# axins1.fill_between(bin_centers, rho_mean-rho_std, rho_mean+rho_std,   alpha=0.4,color=primary_colors[2])

# axins1.plot(x_ref,y_mean-y_std, alpha=1, linestyle='-',lw=1.1,  color=primary_colors[1])
# axins1.plot(x_ref,y_mean+y_std, alpha=1, linestyle='-',lw= 1.1,  color=primary_colors[1])
# axins1.plot(bin_centers, rho_mean-rho_std, alpha=1, linestyle='-',lw=1.,  color=primary_colors[2])
# axins1.plot(bin_centers,rho_mean+rho_std, alpha=1, linestyle='-',lw= 1.,  color=primary_colors[2])

ss =10

axins1.plot(x, y, color=primary_colors[0], linestyle='-', lw=2)
axins1.errorbar(x_ref[::ss], y_mean[::ss], yerr=y_std[::ss],capsize=5,
            capthick=1,
            elinewidth=1, markersize=5, linestyle='-', lw=0.5, color=primary_colors[1], alpha=1)
axins1.errorbar(bin_centers[::ss], rho_mean[::ss], yerr=rho_std[::ss], capsize=3,capthick=1,
            elinewidth=1, markersize=5, linestyle='-', lw=0.5, color=primary_colors[2], alpha=1)

axins1.plot(x_ref[::ss], y_mean[::ss] ,marker='^', markersize=5, linestyle='-',lw= 0.5,  color=primary_colors[1])
axins1.plot(bin_centers[::ss], rho_mean[::ss], '.-', markersize=5,  color=primary_colors[2], lw=0.5)




#=========================================================



N = 2000
count = 0

# First load one file to get shape
data0 = np.loadtxt(f'../Data/Model1/Model1noise_N{N}_1.txt', skiprows=1)
n_points = data0.shape[0]

# Store all realizations
y_all = np.zeros((200-1, n_points))  # rows = realizations, cols = points
x_ref = data0[:, 0]  # x values are the same for all, assume

# Loop over files
for i in range(1, 200):
    data = np.loadtxt(f'../Data/Model1/L2/Model1noise_N{N}_{i}.txt', skiprows=1)
    y_all[count, :] = data[:, 1]
    count += 1

# Compute mean and std across realizations (axis=0 = across runs)
y_mean = np.mean(y_all, axis=0)
y_std = np.std(y_all, axis=0)

print(f"Mean Density for N={N}: {np.mean(y_mean)}")
print(np.mean(y_std))



# -----------------------------------------------------------
# Parameters
hist_range = (0, 1)
bins = 300
count = 0

# allocate array for all realizations
rho_all = np.zeros((100-1, bins))

for i in range(1, 100):
    data2 = np.loadtxt(f'../Data/Model1/Model1Bugs_N{N}_{i}.txt', skiprows=1)
    # Compute histogram (normalized density=True)
    counts1, bin_edges = np.histogram(data2, bins=bins, range=hist_range, density=True)
    rho_all[count, :] = counts1
    count += 1

# compute mean and std across realizations
rho_mean = np.mean(rho_all, axis=0)
rho_std  = np.std(rho_all, axis=0)

# bin centers for plotting
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

print(f"Mean density from particles for N={N}: {np.mean(rho_mean)}")
print(np.mean(rho_std))

# Plot mean + std
# axins2.fill_between(x_ref, y_mean-y_std, y_mean+y_std, alpha=0.01,  color=primary_colors[1])
# axins2.fill_between(bin_centers, rho_mean-rho_std, rho_mean+rho_std,lw =1, alpha=0.6,color=primary_colors[2])
axins2.plot(x, y, color=primary_colors[0], linestyle='-', lw=2)
axins2.errorbar(x_ref[::ss], y_mean[::ss], yerr=y_std[::ss],capsize=5,
            capthick=1,
            elinewidth=1, markersize=5, linestyle='-', lw=0.5, color=primary_colors[1], alpha=1)
axins2.errorbar(bin_centers[::ss], rho_mean[::ss], yerr=rho_std[::ss], capsize=3,capthick=1,
            elinewidth=1, markersize=5, linestyle='-', lw=0.5, color=primary_colors[2], alpha=1)


axins2.plot(x_ref[::ss], y_mean[::ss] ,marker='^', markersize=4, linestyle='-',lw= 0.5,  color=primary_colors[1])
axins2.plot(bin_centers[::ss], rho_mean[::ss], '.-', markersize=6,  color=primary_colors[2], lw=0.5)




# Optional: Zoomed-in region
axins2.set_xlim(0., 0.8)
axins2.set_ylim(0.1, 4)
axins1.set_ylim(0.1, 4)
axins1.set_xlim(0., 0.8)
axins2.tick_params(axis='both', which='major', labelsize=9)
axins1.tick_params(axis='both', which='major', labelsize=9)

axins1.set_xlabel(r'$x$', fontsize=12)
axins1.set_ylabel(r'$\langle\rho(x)\rangle$', fontsize=12)

axins2.set_xlabel(r'$x$', fontsize=12)
# axins2.set_ylabel(r'$\langle\rho(x)\rangle$', fontsize=10)



ax1.text(-0.12, 4.5, s=r'($\textbf{a}$)', fontweight='black', fontsize=15,
         )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
ax2.text(-0.09,4.5, s=r'($\textbf{b}$)', fontweight='black', fontsize=15,
         )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
axins1.text(-0.12, 4.5, s=r'($\textbf{c}$)', fontweight='black', fontsize=15,
         )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))
axins2.text(-0.09, 4.5, s=r'($\textbf{d}$)', fontweight='black', fontsize=15,
         )#bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))


# ax1.legend(fontsize=6, loc='upper right')
# ax2.legend(fontsize=6, loc='upper right')

# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
#           ncol=2, frameon=False)  # Adjust position as needed

# After creating your plots...
fig.legend(handles =[line1,line2,line3],
          loc='upper center',
          bbox_to_anchor=(0.5, 0.),
          frameon=True,
          fancybox=True,
          shadow=False,
          ncol=3)

###################################################
##############################################################
############################################################


plt.savefig('M1_Fig1.png', bbox_inches="tight")  #
plt.savefig('M1_Fig1.pdf', bbox_inches="tight")  #
plt.close()







