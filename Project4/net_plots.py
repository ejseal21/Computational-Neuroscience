'''net_plots.py
Plots for visualization neural network activity
CS 443: Computational Neuroscience
Alice Cole Ethan
Project 4: Motion estimation
'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def plot_act_image_single(act, pause=0.001, cmap='bone'):
    '''Shows a video of images of the neural activity `act` across time (i.e. the same image plot
    updates to show a sequence of images that looks like a video)

    Parameters:
    -----------
    act: ndarray. shape=(n_frames, height, width).
        Activation values in a spatial image at different times. Could be the output of the neural
        network or it could be a RDK to show/visualize.
    pause: float.
        How long to pause between drawing successive frames (i.e. controls frame rate)
    cmap: str.
        Matplotlib color scheme. 'bone' is a good choice for making 0 values black, 1 values white.

    TODO:
    - If `act` doesn't have a time dimension, add a leading singleton dimension.
    - I'll sparse you the trouble of figuring out how to get the basic animation to work with
    Jupyter notebook. Put the following figure creation code outside your main loop:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    Then inside your loop, after all your plotting code, put:
        display(fig)
        clear_output(wait=True)
        plt.pause(pause)
    '''
    if len(act.shape) == 2:
        act = np.expand_dims(act, 0)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for n in range(act.shape[0]):
        plt.imshow(act[n], cmap=cmap)
        # display(fig)
        clear_output(wait=True)
        plt.pause(pause)


def plot_act_image_grid(act, n_rows=2, n_cols=4, pause=0.001, cmap='bone', figSz=(18, 9)):
    '''Generate a grid of plots (generally 2x4 arrangement for 8 directions in 45 deg increments),
    animated when there are more than 1 frame.

    Parameters:
    -----------
    act: ndarray. shape=(n_frames, n_dirs, height, width).
        Activation values in a spatial direction maps at different times.
        Could be the output of the neural network or it could be kernels to visualize.
    n_rows. int.
        Number of rows in the grid plot to place plot panels.
    n_cols. int.
        Number of columns in the grid plot to place plot panels.
    pause: float.
        How long to pause between drawing successive frames (i.e. controls frame rate)
    cmap: str.
        Matplotlib color scheme. 'bone' is a good choice for making 0 values black, 1 values white.
    figSz: tuple of ints.
        (width, height) for the matplotlib figure size.

    TODO:
    - If `act` doesn't have a time dimension, add a leading singleton dimension.
    - Check `n_dirs` and throw an error if it mismatches the number of rows/cols.
    - To make an animated grid of plots, define your plt.figure outside the main loop.
        - At each time, clear the figure (`fig.clf()`).
        - In your deepest loop, create a new axis:
        (`ax = fig.add_subplot(something, something, something)`).
        - After your loops for rows/cols, put the following code:
            display(fig)
            clear_output(wait=True)
            fig.tight_layout()
            plt.pause(pause)
    '''
    if len(act.shape)==3:
        act = np.expand_dims(act, axis=0)
    
    (n_frames, n_dirs, height, width) = act.shape
    if n_dirs != n_cols * n_rows:
        print("Mismatch of n_dirs and n_rows/n_cols")
        return
    
    fig = plt.figure(figsize=figSz)
    for n in range(act.shape[0]):
        fig.clf()
        for d in range(n_dirs):
            ax = fig.add_subplot(n_rows, n_cols, d+1)
            ax.imshow(act[n,d, :, :], cmap=cmap)
        display(fig)
        clear_output(wait=True)
        fig.tight_layout()
        plt.pause(pause)

def vector_sum_plot(act, figSz=(18, 9), pause=0.01):
    '''Visualize the combined activity of all 8 direction cells as a single vector at every position
    (sum of 8 vectors coming out of (x, y), where the 8 receptive fields are positioned).
    Animates over time.

    Goal: Use quiver to plot X, Y, U, and V at each time to visualize the direction cell activity
    in a layer as a vector field.
        - X is the x-coordinates of the cell receptive fields
        - Y is the y-coordinates of the cell receptive fields
        - U is the "u component" of the vector sum at each location (see equation in notebook)
        - V is the "v component" of the vector sum at each location (see equation in notebook)

    Parameters:
    -----------
    act: ndarray. shape=(n_frames, n_dirs, height, width).
        Activation values in a spatial direction maps at different times.
        Output of direction cells.
    figSz: tuple of ints.
        (width, height) for the matplotlib figure size.
    pause: float.
        How long to pause between drawing successive frames (i.e. controls frame rate)

    TODO:
    - If `act` doesn't have a time dimension, add a leading singleton dimension.
    - Compute U and V for each time step.
        - NOTE: You probably want to do this before the main animation loop due to the next
        bullet point...
    - Normalize U and V globally based on the max u/v component achieved across all time.
    - Plot the vector field at each time step using quiver.
        - NOTE: `np.meshgrid` is helpful for setting up the (x, y) coordinates for the vector that
        quiver plots.
    '''
    if len(act.shape) == 3:
        act = np.expand_dims(act, axis=0)
    
    (n_frames, n_dirs, height, width) = act.shape
    U = np.zeros((n_frames, height, width))
    V = np.zeros((n_frames, height, width))
    
    #compute U and V
    for i in range (n_frames):
        for j in range (height):
            for k in range (width):
                for m in range (n_dirs):
                    U[i, j, k] += act[i, m, j, k] * np.cos(2*np.pi*(m+1)/n_dirs)
                    V[i, j, k] += act[i, m, j, k] * np.sin(2*np.pi*(m+1)/n_dirs)
    U = U/np.max(U)
    V = V/np.max(V)
    X = np.arange(0, width)
    Y = np.arange(0, height)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=figSz)
    ax = fig.add_subplot(1, 1, 1)
    for n in range(n_frames):
        plt.quiver(X, Y, U[n], V[n])
        clear_output(wait=True)
        plt.pause(pause)