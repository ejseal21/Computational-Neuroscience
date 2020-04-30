'''filters.py
Creates Gaussian filters for convolution between network areas
CS 443: Computational Neuroscience
Alice Cole Ethan
Project 4: Motion estimation
'''
import numpy as np


def iso_gauss(sz=(5, 5), sigma=1, gain=1, offset=(0, 0)):
    '''Creates a square (`sz` x `sz`) 2D Gaussian kernel with standard deviation `sigma`.

    Parameters:
    -----------
    sz: tuple. Height and width dimensions of the Gaussian kernel (h, w).
        Usually height = width
    sigma: float. Standard deviation (NOT variance) of the 2D Gaussian.
    gain: float. Filter gain — what the filter sums to.
    offset: tuple. Shift in the center of the kernel in rows and cols: (row, col)
        NOTE: "row" means y and "col" means x when it comes to the Gaussian equation

    Returns:
    -----------
    ndarray. shape=(sz, sz).
        The Gaussian kernel (for convolution). Normalized to sum to 1  (before gain applied).

    NOTE:
    - Please implement this from scratch using only numpy.
    - You did something similar to this in the SOM project.

    HINT: np.meshgrid or outer product can be helpful.
    '''
    #controls height of kernel
    one_d = np.empty((sz[0], 1))
    for i in range(-offset[0], sz[0]-offset[0]):
        one_d[i + offset[0], :] = np.power(np.e, (-1/sigma**2) * (i - (sz[0] // 2)) ** 2)
    
    one_d2 = np.empty((sz[0], 1))
    for i in range(-offset[1], sz[1]-offset[1]):
        one_d2[i+offset[1], :] = np.power(np.e, (-1/sigma**2) * (i - (sz[1] // 2)) ** 2)
    
    return one_d @ one_d2.T

def aniso_gauss(k, sigmas=(3, 1), sz=(15, 15), n_dirs=8, gain=1):
    '''Creates an anisotropic 2D Gaussian kernel elongated in the direction k*2*pi/n_dirs.
    The principal (elongated) axis has standard deviation sigma[0], the minor (orthogonal) axis
    has standard deviation sigma[1].

    Parameters:
    -----------
    k: int.
        Between 0-n_dirs-1.  The k in k*2*pi/n_dirs. Determines the principal axis of elongation.
    sz: tuple.
        Height and width dimensions of the Gaussian kernel (h, w). Usually height = width
    n_dirs: int.
        Angle quanization of the principal axis of elongation around the circle (0, 2*pi).
    gain: float.
        Filter gain — what the filter sums to.

    Returns:
    -----------
    ndarray. shape=(sz, sz).
        The Gaussian kernel (for convolution). Normalized to sum to 1 (before gain applied).

    NOTE:
    - Please implement this from scratch using only numpy.

    HINT: np.meshgrid can be helpful.
    '''
    pass
