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
    #controls y position of kernel
    one_d = np.empty((sz[0], 1))
    for i in range(-offset[0], sz[0]-offset[0]):
        one_d[i + offset[0], :] = np.power(np.e, (-1/sigma**2) * (i - (sz[0] // 2)) ** 2)
 
    #controls x position of kernel
    one_d2 = np.empty((sz[0], 1))
    for i in range(-offset[1], sz[1]-offset[1]):
        one_d2[i+offset[1], :] = np.power(np.e, (-1/sigma**2) * (i - (sz[1] // 2)) ** 2)

    ker = one_d @ one_d2.T
    return (ker / np.sum(ker)) * gain

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
    gauss = np.empty(sz)
    for m in range(sz[0]):
        for n in range(sz[1]):
            #the part before the exp
            left = 1/(2 * np.pi * sigmas[0] * sigmas[1])
            
            #the first half of part in the exp
            left_exp = -1/2 * (((m-sz[0]//2) * np.cos(2 * k * np.pi / n_dirs) - (n-sz[1] // 2) * np.sin(2 * k * np.pi / n_dirs)) / sigmas[0])**2
            
            #the second half of the part in the exp
            right_exp = - 1/2 * (((m-sz[0]//2) * np.sin(2 * k * np.pi / n_dirs) + (n-sz[1] // 2) * np.cos(2 * k * np.pi / n_dirs)) / sigmas[1])**2
            
            #combining everything together, indices are flipped here because it needed to be transposed for whatever reason
            gauss[n, m] = left * np.exp(left_exp + right_exp)
    return gauss

