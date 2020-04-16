'''competitive_nets.py
Simulates various competitive networks
CS443: Computational Neuroscience
Alice Cole Ethan
Project 3: Competitive Networks
'''
import numpy as np
from scipy import ndimage


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
    ret = np.empty((1, I.shape[0]))
    x = np.zeros((1, I.shape[0]))
    #time
    t = 0
    #while time is less than max time do the following
    while t < t_max:
        #time increase in iteration
        t += dt
        #iterate over all Inputs
        for i in range(I.shape[0]):
            #notebook equation to calculate change
            change = (-A * x[:, i]) + ((B - x[:, i]) * I[i])
            #add change every time
            x[:, i] = x[:, i] + change * dt
        #add the new neurons back to the return every time
        ret = np.vstack((ret, x))
    return ret

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
    return np.sum(I) - I

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
    ret = np.empty((1, I.shape[0]))
    x = np.zeros((1, I.shape[0]))
    #time
    t = 0
    #while time is less than max time do the following
    while t < t_max:
        #time increase in iteration
        t += dt
        #iterate over all Inputs
        for i in range(I.shape[0]):
            #notebook equation to calculate change
            not_i = sum_not_I(I)[i]
            change = (-A * x[:, i]) + ((B - x[:, i]) * I[i]) - (x[:, i] * not_i)
            #add change every time
            x[:, i] = x[:, i] + change * dt
        #add the new neurons back to the return every time
        ret = np.vstack((ret, x))
    return ret


def dist_dep_net(I, A=1, B=1, C=0, exc_sigma=0.1, inh_sigma=3.0, kerSz=3, t_max=3, dt=0.001):
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
    exc = np.empty((kerSz, 1))
    for k in range(kerSz):
        exc[k, :] = np.power(np.e, (-1/exc_sigma**2) * (k - (kerSz // 2)) ** 2)

    inh = np.empty((kerSz, 1))
    for k in range(kerSz):
        inh[k, :] = np.power(np.e, (-1/inh_sigma**2) * (k - (kerSz // 2)) ** 2)


    pad = int(np.ceil((kerSz - 1) / 2))
    I = np.expand_dims(np.pad(np.squeeze(I), pad), 1)

    ret = np.empty((1, I.shape[0]))
    x = np.zeros((1, I.shape[0]))
    t = 0
    while t < t_max:
        t += dt
        #iterate over all Inputs
        for i in range(pad, I.shape[0] - pad):
            #notebook equation to calculate change
            inhibitory = (-A * x[:, i])
            
            #convolution
            Esum = 0
            Ssum = 0
            for j in range(kerSz):
                Esum += I[i+j-1, :] * exc[j]
                Ssum += I[i+j-1, :] * inh[j]

            excitatory = (B - x[:, i]) * Esum#np.sum(ndimage.convolve(I[i-pad: i+pad], exc))
            inhibitory2 = (C + x[:, i]) * Ssum#np.sum(ndimage.convolve(I[i-pad: i+pad], inh)))
            change = inhibitory + excitatory - inhibitory2
            
            #add change every time
            x[:, i] = x[:, i] + change * dt
        
        #add the new activations back to the return every time
        ret = np.vstack((ret, x))
    return ret[:, pad:-pad]


def dist_dep_net_image(I, A, inh_sigma, kerSz, t_max, dt):
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
    inh = np.empty((kerSz, 1))
    for k in range(kerSz):
        inh[k, :] = np.power(np.e, (-1/inh_sigma**2) * (k - (kerSz // 2)) ** 2)
    inh = inh @ inh.T
    print(inh)

    pad = int(np.ceil((kerSz - 1) / 2))
    I = np.pad(I, pad)

    ret = np.empty((1, I.shape[0], I.shape[1]))
    x = np.zeros((1, I.shape[0], I.shape[1]))
    t = 0
    while t < t_max:
        t += dt
        #iterate over all Inputs
        for i in range(pad, I.shape[0] - pad):
            for j in range(pad, I.shape[1] - pad):
                #notebook equation to calculate change
                inhibitory = (-A * x[:, i])
                Ssum = 0
                #convolution?
                for k in range(-pad, pad):
                    for l in range(-pad, pad):
                        Ssum += I[i+k, j+l] * inh[k+pad, l+pad]

                excitatory = I[i, j]
                inhibitory2 = x[:, i] * Ssum
                change = inhibitory + excitatory - inhibitory2
                
                #add change every time
                x[:, i] = x[:, i] + change * dt
        
        #add the new activations back to the return every time
        ret = np.vstack((ret, x))
    return ret[:, pad:-pad, pad:-pad]


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
    I = np.asarray(I)
    ret = np.empty((1, I.shape[0]))
    x = I
    #time
    t = 0
    #while time is less than max time do the following
    while t < t_max:
        #time increase in iteration
        t += dt
        f = f_function(fun_str, x, F)
        #iterate over all Inputs
        for i in range(I.shape[0]):
            #notebook equation to calculate change
            change = (-A * x[i]) + ((B - x[i]) * f[i] - x[i] * sum_not_I(f)[i])
            #add change every time
            x[i] = x[i] + change * dt
        #add the new neurons back to the return every time
        ret = np.vstack((ret, x))
    return ret

def f_function(fun_str, x, F=0):
    '''
    '''
    if fun_str == 'linear':
        f = x
    elif fun_str == 'faster_than_linear':
        f = np.square(x)
    elif fun_str == 'slower_than_linear':
        f = np.true_divide(x, x+F)
    else:           #sigmoid
        f = np.true_divide(np.square(x), F+np.square(x))
    return f