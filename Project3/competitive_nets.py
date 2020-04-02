'''competitive_nets.py
Simulates various competitive networks
CS443: Computational Neuroscience
YOUR NAMES HERE
Project 3: Competitive Networks
'''
import numpy as np


def leaky_integrator(I, A, B, t_max, dt):
    '''A layer of leaky integrator neurons with shunting excitation.

    Uses Euler's Method for numerical integration.

    Parameters:
    -----------
    I: ndarray. shape=(N,).
        Input vector (assumed to not vary with time here). Components map 1-to-1 to units.
        For example, neuron 0 gets I[0], etc.
    A: float.
        Passive decay rate >= 0.
    B: float.
        Excitatory upper bound of each cell > 0.
    t_max: float.
        Maximum time ("real continuous time", not time steps) to simulate the network > 0.
    dt: float.
        Integration time step > 0.

    Returns:
    -----------
    ndarray. shape=(n_steps, N).
        Each unit in the network's activation at all the integration time steps.
    '''
    pass


def sum_not_I(I):
    '''Sums all the other elements in `I` across all dimensions except for the one in each position

    Parameters:
    -----------
    I: ndarray. shape=(anything).
        Input vector in any number of dimensions

    Returns:
    -----------
    ndarray. shape=shape(I).
    '''
    pass


def lateral_inhibition(I, A, B, t_max, dt):
    '''Shunting network with lateral inhibition

    Parameters:
    -----------
    I: ndarray. shape=(N,).
        Input vector (assumed to not vary with time here). Components map 1-to-1 to units.
        For example, neuron 0 gets I[0], etc.
    A: float.
        Passive decay rate >= 0.
    B: float.
        Excitatory upper bound of each cell > 0.
    t_max: float.
        Maximum time ("real continuous time", not time steps) to simulate the network > 0.
    dt: float.
        Integration time step > 0.

    Returns:
    -----------
    ndarray. shape=(n_steps, N).
        Each unit in the network's activation at all the integration time steps.
    '''
    pass


def dist_dep_net(I, A, B, C, e_sigma, i_sigma, kerSz, t_max, dt):
    '''Distant-dependent (convolutional) 1D shunting network

    Parameters:
    -----------
    I: ndarray. shape=(N,).
        Input vector (assumed to not vary with time here). Component i is CENTERED on cell i,
        but due to convolution there is no longer a 1-to-1 mapping input-to-unit.
    A: float.
        Passive decay rate >= 0.
    B: float.
        Excitatory upper bound of each cell > 0.
    C: float.
        Inhibitory lower bound constant of each cell > 0.
    e_sigma: float.
        Standard deviation of the excitatory Gaussian convolution kernel
    i_sigma: float.
        Standard deviation of the inhibitory Gaussian convolution kernel
    kerSz: int.
        Length of the 1D convolution kernels
    t_max: float.
        Maximum time ("real continuous time", not time steps) to simulate the network > 0.
    dt: float.
        Integration time step > 0.

    Returns:
    -----------
    ndarray. shape=(n_steps, N).
        Each unit in the network's activation at all the integration time steps.

    TODO:
    - Create two small 1D 3x1 Gaussian kernels with different sigma values (see parameters above).
    Select `kerSz` equally spaced sample points between -(`kerSz`-1)/2 and (`kerSz`-1)/2 when making
    your kernel.
    - Do separate 1D convolutions on the raw input to get the excitatory and inhibitory network
    inputs (`same` boundary conditions; you do not need to implement this from scratch).
    The rest should be the same as in previous simulations.
    - Remember to add the inhibitory lower bound C to the network. For now set C=0
    (to focus on other properties of the network).

    NOTE: You may either write your own convolution code (e.g. based on last semester) or use
    the built-in one in scipy.
    '''
    pass


def dist_dep_net_image(I, A, i_sigma, kerSz, t_max, dt):
    '''Distant-dependent (convolutional) 2D shunting network

    NOTE: If the network simulation is too slow on your machine (e.g. you are using very large images),
    you can solve for and replace the ODE with the steady state solution.

    Parameters:
    -----------
    I: ndarray. shape=(N, img_height, img_width).
        Input vector (assumed to not vary with time here).
    A: float.
        Passive decay rate >= 0.
    i_sigma: float.
        Standard deviation of the inhibitory Gaussian convolution kernel
    kerSz: int.
        Length of the 2D convolution kernels
    t_max: float.
        Maximum time ("real continuous time", not time steps) to simulate the network > 0.
    dt: float.
        Integration time step > 0.

    Returns:
    -----------
    ndarray. shape=(n_steps, img_height, img_width).
        Each unit in the network's activation at all the integration time steps.
        NOTE: If you have issues holding all the time steps in memory, you can just return the return
        at the final time step.

    TODO:
    - Adapt your previous distance dependent network code to 2D and the modified equation.
    - Be sure to replace the excitatory convolution with I_ij. The logic is that we don't want to
    blur individual pixel values in the image.
    - To generate a 2D Gaussian kernel, generate a 1D one like before then use the matrix
    multiplication "trick" (outer product) from the end of Project 0 to make a symmetric 2D Gaussian
    kernel (a 1x25 1D kernel should make a 25x25 2D kernel).
    I suggest plotting it to make sure this worked!

    NOTE: You may either write your own convolution code (e.g. based on last semester) or use
    the built-in one in scipy.
    '''
    pass


def rcf(I, A, B, fun_str, t_max, dt, F=0):
    '''Recurrent competitive field network

    Parameters:
    -----------
    I: ndarray. shape=(N,).
        Input vector (assumed to not vary with time here). Components map 1-to-1 to units.
        For example, neuron 0 gets I[0], etc.
    A: float.
        Passive decay rate >= 0.
    B: float.
        Excitatory upper bound of each cell > 0.
    fun_str: str.
        Name of recurrent feedback function to use in the network. Options are:
        'linear', 'faster_than_linear', 'slower_than_linear', 'sigmoid'
    t_max: float.
        Maximum time ("real continuous time", not time steps) to simulate the network > 0.
    dt: float.
        Integration time step > 0.
    F: float.
        Parameter in slower-than-linear and sigmoid functions that controls inflection point > 0.

    Returns:
    -----------
    ndarray. shape=(n_steps, N).
        Each unit in the network's activation at all the integration time steps.
    '''
    pass
