"""Library to perform visualisation functions in delayed coupling model of segmentation clock"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_line(data, interval=5, plot_save=False, filename='Animation_grid.gif'):
    """
    Create a 2D animation of a grid from a 3D array.
    
    Parameters:
    - data: 3D numpy array of shape (grid_size, grid_size, num_tsteps)
    - interval: Time between frames in milliseconds
    
    Returns:
    - HTML object to display in Jupyter Notebook
    """

    grid_size = data.shape[0]
    num_tsteps = data.shape[1]

    # Set up the figure and two axes for vertically stacked images
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)

    # Plot the first image
    im = ax.imshow(data[:,0].reshape(1,-1), cmap='Purples', aspect='equal', vmin = np.min(data), vmax = np.max(data))
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar for the first image
    cbar1 = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.1, pad=0.2)
    cbar1.set_label('phase')

    # Update function for animation
    def update(frame):
        im.set_array(data[:,frame].reshape(1,-1))
        return [im]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=num_tsteps, interval=interval, blit=True)
    if plot_save:
        ani.save(filename, writer='imagemagick', fps=10)
        
    # Convert animation to HTML for Jupyter Notebook
    return HTML(ani.to_jshtml())

def animate_2D(data, interval=5, plot_save=False, filename='Animation_grid.gif', title = ''):
    """
    Create a 2D animation of a grid from a 3D array.
    
    Parameters:
    - data: 3D numpy array of shape (grid_size, grid_size, num_tsteps)
    - interval: Time between frames in milliseconds
    
    Returns:
    - HTML object to display in Jupyter Notebook
    """

    # transpose axis 0 and 1 of data
    data = np.transpose(data, axes = (1, 0, 2))

    num_tsteps = data.shape[2]

    # Set up the figure and two axes for vertically stacked images
    fig, ax = plt.subplots(1, 1, figsize=(data.shape[1]/4, data.shape[0]/4+3), constrained_layout=True)

    # Plot the first image
    im = ax.imshow(data[:,:,0], cmap='Purples', aspect='equal', vmin = np.min(data), vmax = np.max(data))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('NMP cells --> Posterior PSM --> Anterior PSM')
    ax.set_ylabel('Width of PSM')
    ax.set_title(title)
    
    # Add colorbar for the first image
    cbar1 = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.1, pad=0.2)
    cbar1.set_label('phase')

    # Update function for animation
    def update(frame):
        im.set_array(data[:,:,frame])
        return [im]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=num_tsteps, interval=interval, blit=True)
    if plot_save:
        ani.save(filename, writer='imagemagick', fps=10)
    
    # Convert animation to HTML for Jupyter Notebook
    return HTML(ani.to_jshtml())