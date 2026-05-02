import h5py
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import FancyBboxPatch
import os
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import numpy as np
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

# Choose the font for the figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

bounds = np.array([0, 10])  # Domain bounds
L = bounds[1] - bounds[0]  # Length of the domain
x_min, x_max = bounds[0], bounds[1]
y_min, y_max = bounds[0], bounds[1]
extent = [x_min, x_max, y_min, y_max]
# Create a reusable 2x3 figure environment with placeholder panels
def make_2x3_figure(save_path=None,  figsize=(6.5, 4.3), suptitle=None):
    cmap = 'inferno'
    boundaries = [0, 1, 3, 6, 15]
    norm = BoundaryNorm(boundaries, ncolors=256)
    fig, axes = plt.subplots(2, 3, figsize=figsize,sharex=True, sharey=True)

    N = 20000
    D = [0.9, 0.7]
    base_folder = ['Model0/det/p100', 'Model0/noise/p100', 'Model0/Bugs_2d/Model4']

    i = 0
    for d in D:
        for base in base_folder:


            if base == base_folder[-1]:
                file_path = f'../Data/{base}/_N{N}_D{d:.4f}/dat1.h5'
                with h5py.File(file_path, 'r') as f:
                    group = f['t0.01']
                    x = group['x'][:]
                    y = group['y'][:]
                    # set axis facecolor to black and draw reddish points sampled from the inferno cmap
                    ax = axes.flat[i]
                    ax.set_facecolor('black')
                    cmap_obj = plt.get_cmap(cmap)
                    # pick a reddish sample from inferno (~0.8-0.9)
                    color = cmap_obj(0.85)
                    ax.scatter(y/0.1, x/0.1, s=1, color=color, edgecolors='none', alpha=0.9)
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    i += 1
            elif base == base_folder[0]:
                N = 20000
                file_path = f'../Data/{base}/_N{N}_D{d:.4f}/dat.h5'
                with h5py.File(file_path, 'r') as f:
                    # print_hdf5_contents(file_path)
                    time = f['t50.0'][:]
                    if d == D[0]:
                        axes.flat[i].imshow(time, aspect='auto', cmap=cmap,norm= norm,extent = extent)
                    else:
                        axes.flat[i].imshow(time, aspect='auto', cmap=cmap,norm= norm,extent  = extent)
                    i+=1
            else :
                    # print_hdf5_contents(file_path)
                N = 20000
                file_path = f'../Data/{base}/_N{N}_D{d:.4f}/dat.h5'
                with h5py.File(file_path, 'r') as f:
                    # print_hdf5_contents(file_path)
                    time = f['t0.999'][:]
                    print(f'Vmax of noise is {np.max(time)}')
                    if d == D[0]:
                        axes.flat[i].imshow(time, aspect='auto',cmap =cmap,norm =norm,extent=extent)
                    else:
                        axes.flat[i].imshow(time, aspect='auto',cmap =cmap,norm= norm,extent = extent)
                    i+=1

    text_labels = [r'(\textbf{a})', r'(\textbf{b})', r'(\textbf{c})', r'(\textbf{d})', r'(\textbf{e})', r'(\textbf{f})']
    for i, ax in enumerate(axes.flat, start=1):
        ax.text(0.024, 0.975, text_labels[i-1], transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.23",
                          fc="white", ec="black", lw=0.6, alpha=1))
        ax.set_xticks([0,5,10])
        ax.set_yticks([0,5,10])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_alpha(0.6)

    axes[0, 0].set_ylabel(r'$\tilde{D}= 0.09$' '\n ' r'$y/R$', fontsize=13 ,labelpad=8)
    axes[1, 0].set_ylabel(r'$\tilde{D}=0.07$' '\n ' r'$y/R$', fontsize=13 ,labelpad=8)
    axes[1, 1].set_xlabel(r'$x/R$', fontsize=13)
    axes[1, 2].set_xlabel(r'$x/R$', fontsize=13)
    axes[1, 0].set_xlabel(r'$x/R$', fontsize=13)

    # Column labels to display inside boxes that join the label and the column
    col_labels = ["deterministic \n DKTE", "DKTE", "Particles"]

    plt.subplots_adjust(hspace=0.12, wspace=0.2)

    # Draw a rounded box around each column (in figure coordinates) and place the label centered at the top inside it.
    for col in range(3):

        top_ax = axes[0, col]
        bot_ax = axes[1, col]
        # positions are Bbox in figure coordinates
        top_pos = top_ax.get_position()
        bot_pos = bot_ax.get_position()
        x0 = min(top_pos.x0, bot_pos.x0) - 0.001
        x1 = max(top_pos.x1, bot_pos.x1) + 0.001
        y0 = min(top_pos.y0, bot_pos.y0) - 0.07
        # small extra top padding so the label fits comfortably inside the box
        y1 = max(top_pos.y1, bot_pos.y1) + 0.05

        width = x1 - x0
        height = y1 - y0

        # create the rounded rectangle patch in figure coordinates
        box = FancyBboxPatch((x0, y0), width, height,
                             boxstyle="round,pad=0.02",
                             transform=fig.transFigure,
                             linewidth=1.1,
                             edgecolor='black',alpha =0.,
                             facecolor='none',
                             zorder=2)
        fig.add_artist(box)

        # place the label centered near the top inside the box
        x_center = x0 + width / 2.0
        y_text = y1 + 0.05
        fig.text(x_center, y_text, col_labels[col], ha='center', va='top', fontsize=15)

    cbar_ax = fig.add_axes([0.12, -0.03, 0.5, 0.02])  # Adjust as needed

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,  # Use cax instead of ax
        orientation='horizontal'
    )
    cbar.set_label(
        r'Normalized density, $\tilde\rho$',
        fontsize=14, )
    tick_labels = ['0', '1', '3', '6', r'$\geq$15']
    cbar.set_ticklabels(tick_labels)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    if save_path:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=900, bbox_inches='tight')

    return fig, axes


if __name__ == '__main__':
    # Save next to this script by default
    out_path = os.path.join('..\..\IFISC\DK_project\Draft\V6_PRE _Referee', 'Fig1.pdf')
    fig, axes = make_2x3_figure(save_path=out_path)
    print(f"Saved example figure to: {out_path}")
    plt.close(fig)


