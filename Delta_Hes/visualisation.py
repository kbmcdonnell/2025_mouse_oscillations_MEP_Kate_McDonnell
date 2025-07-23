""" This module provides functions to visualize 2D animations of grid data and generate distinguishable colors from colormaps."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Define colors
colors = ['#fdc776', '#fc4e2a', '#8b1a0e']

# Create colormap
hes_cmap = LinearSegmentedColormap.from_list("hes_cmap", colors)
plt.register_cmap(name='hes_cmap', cmap=hes_cmap)

def animate_2D(data, interval=5, plot_save=False, filename='Animation_grid.gif', title = ''):
    """
    Create a 2D animation of a grid from a 3D array (time x grid).
    
    Parameters:
    - data: 3D numpy array of shape (grid_size, grid_size, num_tsteps)
    - interval: Time between frames in milliseconds
    - plot_save: If True, saves the animation as a GIF file
    - filename: Name of the file to save the animation
    - title: Title for the plot
    
    Returns:
    - HTML object to display in Jupyter Notebook
    """

    # transpose axis 1 and 2 of data
    data = np.transpose(data, axes = (0, 2, 1))
    # data = np.transpose(data, axes = (1, 0, 2))

    num_tsteps = data.shape[0]

    # Set up the figure and two axes for vertically stacked images
    fig, ax = plt.subplots(1, 1, figsize=(data.shape[2]/3, data.shape[1]/3 + 2), constrained_layout=True)

    # Plot the first image
    im = ax.imshow(data[0,:,:], cmap='YlOrRd', aspect='equal', vmin = np.min(data), vmax = np.max(data))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Posterior PSM --> Anterior PSM')
    ax.set_ylabel('Width of PSM')
    ax.set_title(title)
    
    # Add colorbar for the first image
    cbar1 = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.1, pad=0.2)
    cbar1.set_label('Hes concentration')

    # Add a text annotation for timeframe
    timeframe_text = ax.text(0.05, 0.95, f'Time: 0', color='white', fontsize=12, transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5))

    # Update function for animation
    def update(frame):
        im.set_array(data[frame,:,:])
        timeframe_text.set_text(f'Time: {frame*interval}')  # Update timeframe text
        return [im]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=num_tsteps, interval=interval, blit=True)
    if plot_save:
        ani.save(filename, writer='imagemagick', fps=10)
    
    # Convert animation to HTML for Jupyter Notebook
    return HTML(ani.to_jshtml())

def get_distinguishable_ylgn_colors(k, min_val=0.3, max_val=0.8):
    """
    Returns k distinguishable colors from the 'YlGn' colormap.

    Parameters:
    - k: Number of colors.
    - min_val: Minimum position in the colormap (0 = lightest, 1 = darkest).
    - max_val: Maximum position in the colormap.
    """
    assert 0 <= min_val < max_val <= 1, "min_val and max_val must be in (0, 1]"
    cmap = cm.get_cmap('YlGn')
    values = np.linspace(min_val, max_val, k)
    colors = [cmap(v) for v in values]
    return colors

def get_hes_colors(k):
    cmap = cm.get_cmap('hes_cmap')
    values = np.linspace(0, 1, k)
    colors = [cmap(v) for v in values]
    return colors



def get_two_colormaps(k, avoid_white=True):
    """
    Returns two lists of k colors each:
    - One from the 'Blues' colormap
    - One from the 'Reds' colormap
    Colors are spaced and trimmed to avoid white and black extremes.
    """
    # Trim range to avoid white (too light) and near-black (too dark)
    min_val, max_val = (0.3, 0.85) if avoid_white else (0.0, 1.0)
    sample_points = np.linspace(min_val, max_val, k)

    blues_cmap = plt.cm.Blues(sample_points)
    reds_cmap = plt.cm.Reds(sample_points)

    # Convert to RGB tuples (remove alpha)
    blues_rgb = [tuple(color[:3]) for color in blues_cmap]
    reds_rgb = [tuple(color[:3]) for color in reds_cmap]

    return blues_rgb, reds_rgb