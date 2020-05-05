'''motion.py
Network that detects motions and estimates the dominant direction
CS 443: Computational Neuroscience
Alice Cole Ethan
Project 4: Motion estimation
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


import filters
import math


class KernelParams():
    '''Organizes parameters of ONE excitatory/inhibitory kernel (e.g. Layer 4 excitatory)
    '''
    def __init__(self, sigma, sz, gain=1):
        '''
        Parameters:
        -----------
        sigma. float or tuple.
            Standard deviation of the Gaussian kernel. Float in the case of isotropic Gaussian.
            Tuple of floats in the case of anisotropic kernel.
        sz: tuple.
            Height and width dimensions of the Gaussian kernel (h, w). Usually height = width
        gain: float.
            Filter gain — what the filter sums to.
        '''
        self.sigma = sigma
        self.sz = sz
        self.gain = gain

    def get_sigma(self):
        return self.sigma

    def get_size(self):
        return self.sz

    def get_gain(self):
        return self.gain

    def get_all(self):
        '''Convenience method to get all the parameter values'''
        return self.sigma, self.sz, self.gain


class LayerParams():
    '''Organizes parameters of ONE neural network layer (e.g. Layer 5: Long-range filter (MT))
    '''
    def __init__(self, tau=5, A=1, B=1, C=0, output_thres=0, excit_g=1, inhib_g=1):
        '''
        NOTE: Not all parameters necessarily apply to each layer.
        Pay attention to the network equations.

        Parameters:
        -----------
        tau. float.
            Inverse cell time constant. Higher values speed up cell dynamics (but may make
            numerical integration unstable).
        A: float.
            Passive decay rate.
        B: float.
            Excitatory upper bound
        C: float.
            Inhibitory lower bound
        output_thres: float.
            Activation threshold to apply to the output signals coming out of a layer.
            Subtracted from the cell activity and anything less than 0 is set to 0.
        excit_g: float:
            Gain on the feedforward excitatory input from the previous layer.
        inhib_g: float:
            Gain on the inhibitory input.
        '''
        self.tau = tau
        self.A = A
        self.B = B
        self.C = C
        self.output_thres = output_thres
        self.excit_g = excit_g
        self.inhib_g = inhib_g

    def get_time_const(self):
        # print(self.tau)
        return self.tau

    def get_decay(self):
        return self.A

    def get_upper_bound(self):
        return self.B

    def get_lower_bound(self):
        return self.C

    def get_output_thres(self):
        return self.output_thres

    def get_excit_gain(self):
        return self.excit_g

    def get_inhib_gain(self):
        return self.inhib_g

    def get_all(self):
        '''Convenience method to get all the parameter values'''
        return self.tau, self.A, self.B, self.C, self.output_thres, self.excit_g, self.inhib_g


class HGateParams():
    '''Organizes parameters of a habituative gate (depressing synapse)
    '''
    def __init__(self, tau, A=1, B=1, K=50):
        '''
        Parameters:
        -----------
        tau. float.
            Inverse cell time constant. Higher values speed up cell dynamics (but may make
            numerical integration unstable).
        A: float.
            Passive decay rate.
        B: float.
            Excitatory upper bound
        K: float.
            Rate of synaptic depression to 0 (drop in synapse efficiacy)
        '''
        self.tau = tau
        self.A = A
        self.B = B
        self.K = K

    def get_time_const(self):
        return self.tau

    def get_decay(self):
        return self.A

    def get_upper_bound(self):
        return self.B

    def get_depression_rate(self):
        return self.K

    def get_all(self):
        '''Convenience method to get all the parameter values'''
        return self.tau, self.A, self.B, self.K


class MotionNet:
    '''6-Layer Network model of how primate visual system (doral stream) detects and processes
    motion in areas LGN, V1, MT, and MSTd.
    '''
    def __init__(self, dt, n_dirs, lvl1_params, lv1_hgate_params, lvl2_inter_params=None,
                 lvl2_params=None, lvl3_params=None, lvl4_params=None, lvl5_params=None,
                 lvl6_params=None,
                 lvl3_excit_ker_params=None, lvl4_excit_ker_params=None, lvl4_inhib_ker_params=None,
                 lvl5_excit_ker_params=None, lvl6_inhib_ker_params=None,
                 do_lvl1=True, do_lvl2=True, do_lvl3=True, do_lvl4=True, do_lvl5=True, do_lvl6=True):
        '''
        Parameters:
        -----------
        dt. float.
            Integration time step in "seconds", where we assume that network integrates one input
            video frame per "second".
            i.e. If dt = 0.1, then there are 10 time steps per "second" of simulation.
        n_dirs: int.
            Number of motion directions cells prefer equally spaced 0-360.
        lvl1_params: LayerParams.
            Collection of parameters relating to Layer 1 of the net: Non-directional transient cells
        lv1_hgate_params: HGateParams.
            Collection of parameters relating to habituative gates in Layer 1.
        lvl2_inter_params: LayerParams.
            Collection of parameters relating to inhibitory interneurons in Layer 2 of the net:
            Directional transient cells
        lvl2_params: LayerParams.
            Collection of parameters relating to Layer 2 of the net: Directional transient cells
        lvl3_params: LayerParams.
            Collection of parameters relating to Layer 3 of the net: Short-range filter cells
        lvl4_params: LayerParams.
            Collection of parameters relating to Layer 4 of the net: Competition cells in MT
        lvl5_params: LayerParams.
            Collection of parameters relating to Layer 5 of the net: Long-range filter cells
        lvl6_params: LayerParams.
            Collection of parameters relating to Layer 6 of the net: Direction grouping cells in MSTd
        lvl3_excit_ker_params: KernelParams.
            Collection of parameters relating to the excitatory input (from Layer 2) Gaussian kernel
            for Layer 3 cells.
        lvl4_excit_ker_params: KernelParams.
            Collection of parameters relating to the excitatory input (from Layer 3) Gaussian kernel
            for Layer 4 cells.
        lvl4_inhib_ker_params: KernelParams.
            Collection of parameters relating to the inhibitory input (from Layer 3) Gaussian kernel
            for Layer 4 cells.
        lvl5_excit_ker_params: KernelParams.
            Collection of parameters relating to the excitatory input (from Layer 4) Gaussian kernel
            for Layer 5 cells.
        lvl6_excit_ker_params: KernelParams.
            Collection of parameters relating to the Gaussian kernel that applies recurrent inhibition
            to Layer 6 cells and feedback inhibition to Layer 5 cells.
        do_lvli. boolean.
            Toggle network levels on/off during simulation. Used for debugging/test code
            Assumption is that we always compute up from layer 1 to layer N
            For example, lvl1 = True, lvl2 = True, is possible, but lvl1 = False, lvl2 = True, is NOT
            possible because higher layers require feedforward input from lower ones.

        TODO:
        - Save parameters as instance variables.
            - NOTE: Yes, this is a little tedious but this should help get you familar with
            the variables and structure.
        - Compute `self.n_steps_per_frame`, the number of time steps per "second"/frame of simulation.
        '''
        # Input placeholders
        self.inputs = None
        self.height = None
        self.width = None

        # Layer activity placeholders: these are cell populations whose activations we solve for
        # via numerical integration of ODEs
        # Layer 1
        self.x = None  # input integrator cells
        self.z = None  # input habituative gates
        # Layer 2
        self.dir_trans_inter_cells = None  # directional transient inhibitory interneurons
        self.dir_trans_cells = None  # directional transient cells
        # Layer 3
        self.srf_cells = None  # short-range filter cells
        # Layer 4
        self.comp_cells = None  # MT competition cells
        # Layer 5
        self.lr_cells = None  # long-range filter cells
        # Layer 6
        self.mstd_cells = None  # global grouping direction cells

        # Layer output activity placeholders: these are for saving the output of a layer at all times
        # (e.g. after thresholding). These are NOT computed via numerical integration / ODEs.
        # Layer 1
        self.y = None  # Non-direction transient cells
        # Layer 2
        self.dir_trans_out = None  # output of directional transient cells
        # Layer 3
        self.srf_out = None  # output of short-range filter cells
        # Layer 4
        self.comp_out = None  # output of MT competition cells
        # Layer 5
        self.lr_out = None  # output of long-range filter cells
        # Layer 6
        self.mstd_out = None  # output of global grouping direction cells

        self.dt = dt
        self.n_dirs = n_dirs
        self.inhib_dir_shift_map = self.make_inhib_dir_shift_map(self.n_dirs)

        # Excitatory/inhibitory kernel placeholders
        self.short_range_excit_ker = None
        self.comp_excit_ker = None
        self.comp_inhib_ker = None
        self.long_range_excit_ker = None
        self.mstd_inhib_ker = None

        
        # boolean: controls which layer is turned on or off
        self.do_lvl1 = do_lvl1
        self.do_lvl2 = do_lvl2
        self.do_lvl3 = do_lvl3
        self.do_lvl4 = do_lvl4
        self.do_lvl5 = do_lvl5
        self.do_lvl6 = do_lvl6

        # layer parameters
        self.layer1 = lvl1_params
        self.hgate = lv1_hgate_params
        self.layer2 = lvl2_params
        self.layer2_inhib = lvl2_inter_params
        self.layer3 = lvl3_params
        self.layer3_excite = lvl3_excit_ker_params
        self.layer4 = lvl4_params
        self.layer4_excite = lvl4_excit_ker_params
        self.layer4_inhib = lvl4_inhib_ker_params
        self.layer5 = lvl5_params
        self.layer5_excite = lvl5_excit_ker_params
        

        self.make_kernels()



    def get_input(self, t):
        '''Get the appropriate external input signal frame at the current time step `t`.

        For example: if dt = 0.1,
        t=0 should return frame 1,
        t=1 should return frame 1
        t=2 should return frame 1
        t=9 should return frame 1
        t=10 should return frame 2
        t=11 should return frame 2
        ...

        Parameters:
        -----------
        t: int.
            Current time step of the simulation. e.g. 1, 2, 3, ..., 99, 100, 101, ...

        Returns:
        -----------
        ndarray. shape=(Iy, Ix)
            The external input signal at the current time step `t`

        NOTE: Assumes that constructor has been called and
            self.inputs (input video) is defined. shape=(n_frames, n_rows, n_cols).
        NOTE: Should throw a helpful error if self.inputs is None.
        '''
        if self.inputs.any() == None:
            print("ERROR: The constructor hasn't been called or there are no inputs")
            return 
        else:
            dt_magnitude = 1/self.dt
            return self.inputs[math.floor(t/dt_magnitude)]


    def make_inhib_dir_shift_map(self, n_dirs):
        '''Needed in motion detection / nulling inhibition (Layer 2: Directional transient cells).
        Maps each motion direction index (0, ..., 7) to a tuple indicating the row/col offsets
        needed to "take a step" in the direction of the opponent motion direction. For example a
        cell that prefers northeast motion (+45 deg) has an opponent motion direction of
        southwest (+225 deg). The shifts therefore are (1, -1) (i.e. step down 1 row, then step
        left one column). In this case, the key/value entry in the dictionary would be `1 -> (1, -1)`.

        NOTE: Assumes 0 deg angle is aligned with + x axis and increases positively CCW.

        Parameters:
        -----------
        n_dirs: int.
            Number of motion directions cells prefer equally spaced 0-360.

        Returns:
        -----------
        Python dictionary.
            keys: int. direction indices 0,...,7
            values: tuples of 2 ints. Shifts in the direction of each motion direction angle's
                opponent direction (+180 deg)

        TODO:
        - Aside from all the above, be sure to add a call in your constructor to make this dictionary
        and save the dictionary as an instance variable. You will need to use this dictionary more
        than once.
        '''
        dir_map = {}
        dir_map[0] = (0, -1)
        dir_map[1] = (1, -1)
        dir_map[2] = (1, 0)
        dir_map[3] = (1, 1)
        dir_map[4] = (0, 1)
        dir_map[5] = (-1, 1)
        dir_map[6] = (-1, 0)
        dir_map[7] = (-1, -1)
        return dir_map

    def make_kernels(self):
        '''Makes all the excitatory/inhibitory convolutional kernels in Layers 3+. See constructor
        for kernel instance variables to set. Here is a summary of what the kernels should be in
        each area:

        Layer 3: Short-range filter.
            - Excitatory kernel. Anisotropic Gaussian aligned with each direction preference.
            shape=(n_dirs, sz_rows, sz_cols).
        Layer 4: Competition cells in MT
            - Excitatory kernel. Anisotropic Gaussian aligned with each direction preference.
            shape=(n_dirs, sz_rows_excit, sz_cols_excit).
            - Inhibitory kernel. Isotropic Gaussian offset in the opponent direction.
            shape=(n_dirs, sz_rows_inhib, sz_cols_inhib).
        Layer 5: Long-range filter.
            - Excitatory kernel. Anisotropic Gaussian aligned with each direction preference.
            shape=(n_dirs, sz_rows, sz_cols).
        Layer 6: Direction grouping in MSTd
            - Inhibitory kernel. Isotropic Gaussian.
            shape=(n_dirs, sz_rows, sz_cols).
        '''
        # layer 3 helloooooo
        if self.do_lvl3:
            self.short_range_excit_ker = np.zeros((self.n_dirs, self.layer3_excite.get_size()[0], self.layer3_excite.get_size()[1]))
            for i in range(self.n_dirs):
                self.short_range_excit_ker[i, :, :] = filters.aniso_gauss(i, sigmas=self.layer3_excite.get_sigma(), sz=self.layer3_excite.get_size())
        #layer 4
        if self.do_lvl4:
            self.comp_excit_ker = np.zeros((self.n_dirs, self.layer4_excite.get_size()[0], self.layer4_excite.get_size()[1]))
            for i in range(self.n_dirs):
                self.comp_excit_ker[i, :, :] = filters.aniso_gauss(i, sigmas=self.layer4_excite.get_sigma(), sz=self.layer4_excite.get_size())
            self.comp_inhib_ker = np.zeros((self.n_dirs, self.layer4_inhib.get_size()[0], self.layer4_inhib.get_size()[1]))
            for i in range(self.n_dirs):
                self.comp_inhib_ker[i, :, :] = filters.iso_gauss(sigma=self.layer4_inhib.get_sigma(), sz=self.layer4_inhib.get_size())
        #layer 5
        if self.do_lvl5:
            self.long_range_excit_ker = np.zeros((self.n_dirs, self.layer5_excite.get_size()[0], self.layer5_excite.get_size()[1]))
            for i in range(self.n_dirs):
                self.long_range_excit_ker[i, :, :] = filters.aniso_gauss(i, sigmas=self.layer5_excite.get_sigma(), sz=self.layer5_excite.get_size())
        #layer 6
        self.mstd_inhib_ker = None

        

    def make_mstd_fb_wts(self):
        '''Weights for directional inhibition in MSTd and feedback from MSTd to MT
        (long-range filters). Each cell recieves no inhibition from other neurons with the
        same direction preference (0 wt). Each cell recieves the most inhibition from the
        opponent direction (2 wt). All other directions recieve equally weighted moderate
        inhibition (1 wt).

        In summary, the inhibitory weights are:
        d = d_preferred -> 0
        d = d_opponent -> 2
        d != d_preferred or d_opponent -> 1

        Returns:
        -----------
        ndarray. shape=(n_dir, n_dir).
            Symmetric weights matrix to scale the inhibition MSTd and long-range filter neurons
            recieve from other nearby direction neurons.

        NOTE: This should be algorithmically generated, NOT hard coded. i.e. should work if
        n_dirs were not 8.

        NOTE: The only instance variable that you should need to compute this is `self.n_dirs`
        '''
        pass

    def initialize(self, n_steps, height, width):
        '''Instantiates each of the network data structures to hold network activity and layer
        output signals (i.e. initializes all variables declared in constructor in sections
        *Layer activity placeholders* and *Layer output activity placeholders*).

        Parameters:
        -----------
        n_steps. int.
            Number of simulation time steps (not frames).
        height. int.
            Height of each input video frame in pixels.
        width. int.
            Width of each input video frame in pixels.

        TODO:
        - Set the instance variables `self.height` and `self.width`.
        - Instantiate all network activity data structures.

        NOTE: Every structure before Layer 2: Directional transients has shape (n_steps, height, width)
            [We haven't detected motion directions yet!]
        NOTE: Every structure Layer 2+ has shape (n_steps, n_dirs, height, width)
        NOTE: Remember that habituative gates should have different initial conditions than all the
        rest.
        '''
        self.height = int(height)
        self.width = int(width)
        n_steps = int(n_steps)
        #layer 1
        self.x = np.zeros((n_steps, height, width))
        self.z = np.ones((n_steps, height, width)) #habituative gates starts with 1s
        self.y = np.zeros((n_steps, height, width))

        #layer 2
        self.dir_trans_inter_cells = np.zeros((n_steps, self.n_dirs, height, width))
        self.dir_trans_cells = np.zeros((n_steps, self.n_dirs, height, width))
        self.dir_trans_out = np.zeros((n_steps, self.n_dirs, height, width))

        # #layer 3
        self.srf_cells = np.zeros((n_steps, self.n_dirs, height, width))
        self.srf_out = np.zeros((n_steps, self.n_dirs, height, width))

        # #layer 4
        self.comp_cells = np.zeros((n_steps, self.n_dirs, height, width))
        self.comp_out = np.zeros((n_steps, self.n_dirs, height, width))

        #layer 5
        self.lr_cells = np.zeros((n_steps, n_dirs, height, width))
        self.lr_out = np.zeros((n_steps, n_dirs, height, width))

        # #layer 6
        # self.mstd_cells = np.zeros((n_steps, n_dirs, height, width))
        # self.mstd_out = np.zeros((n_steps, n_dirs, height, width))

    def get_opponent_direction(self, dir):
        '''Given the direction index `dir`, return the index of the opponent direction
        (180 deg away).

        Parameters:
        -----------
        dir. int.
            Direction index 0, ..., n_dirs-1

        Returns:
        -----------
        int. Index of the opponent direction.

        NOTE: The only instance variable that you should need to compute this is `self.n_dirs`
        '''
        if dir >= self.n_dirs/2:
            return int(dir-self.n_dirs/2)
        else:
            return int(dir+self.n_dirs/2)

    def d_non_dir_transient_cells(self, t):
        '''Compute the change in the Layer 1 cells: Non-directional Transient Cells.

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        Returns:
        -----------
        d_x: ndarray. shape=(height, width)
            Derivative of the input leaky integrator cells at time t
            Derivative of the input leaky integrator cells at time t
        d_z: ndarray. shape=(height, width)
            Derivative of the habituative gates at time t
        '''

        d_x = -self.layer1.get_decay() * self.x[t-1] + (self.layer1.get_upper_bound() - self.x[t-1]) * self.get_input(t)
        d_z = 1 - self.z[t-1] - self.hgate.get_depression_rate() * self.x[t-1] * self.z[t-1]

        return d_x, d_z
        

    def d_dir_transient_cells(self, t):
        '''Compute the change in the Layer 2 cells: Directional Transient Cells
        (and their inhibitory interneurons).

        This is the nulling inhibition stage where motion direction is first detected!

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        Returns:
        -----------
        d_dir_trans_inter_cells: ndarray. shape=(n_dirs, height, width)
            Derivative of the directional transient inhibitory interneurons at time t
        d_dir_trans_cells: ndarray. shape=(n_dirs, height, width)
            Derivative of the directional transient cells at time t

        NOTE:
        - When computing the output signal from Layer 1, remember to use the newly updated signals
        at time t (not t-1). We want the effects due to the input to sweep through the entire network
        in 1 time step.
            - However...you should use Layer 2 values from t-1 in your derivatives
            (we are in the process of computing the Layer 2 values for time t!)
        - The directional transient cells and interneurons follow almost the same equation, except
        that the parameters you pick for the interneurons need to operate on a
        significantly SLOWER timescale than the directional transient cells (i.e. taus). This leads
        to more robust motion direction.
        - The main "trick" with this Layer of the network is that both dir trans cells and interneurons
        recieve inhibition from the cells that prefer the opponent direction at an OFFSET position
        (a step toward the opponent direction). So there are 2 things going on here:
            - A spatial shift
            - indicating that the inhibition is recieved from cells tuned to d+180 deg

        Hint: make use of `self.inhib_dir_shift_map`
        '''
        #initialize
        d_dir_trans_inter_cells = np.zeros((self.n_dirs, self.height, self.width))
        d_dir_trans_cells = np.zeros((self.n_dirs, self.height, self.width))

        #work on each direction
        for d in range(self.n_dirs):
            #gets the opposite of each given direction
            oppo = self.get_opponent_direction(d)
            #gets the given directional shift from a dict
            dir_shift = self.inhib_dir_shift_map[d]

            for i in range(self.height):
                for j in range(self.width):
                    offset_i = (i - dir_shift[0]) % self.height
                    offset_j = (j - dir_shift[1]) % self.width
                    d_dir_trans_inter_cells[d, i, j] = self.layer2_inhib.get_time_const() * (-self.dir_trans_inter_cells[t-1, d, i, j] + self.layer2_inhib.get_excit_gain()*self.y[t, i, j] - self.layer2_inhib.get_excit_gain()*np.maximum(self.dir_trans_inter_cells[t-1, oppo, offset_i, offset_j], 0))

                    d_dir_trans_cells[d, i, j] = self.layer2.get_time_const() * (-self.dir_trans_cells[t-1, d, i, j] + self.layer2.get_excit_gain()*self.y[t, i, j] - self.layer2.get_inhib_gain()*np.maximum(self.dir_trans_inter_cells[t-1, oppo, offset_i, offset_j], 0))

        return d_dir_trans_inter_cells, d_dir_trans_cells

    
    def d_short_range_filter(self, t):
        '''Compute the change in the Layer 3 cells: Short-range filter cells

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        Returns:
        -----------
        d_srf: ndarray. shape=(n_dirs, height, width)
            Derivative of the short-range filter cells at time t

        TODO:
        - Aside from the usual stuff, make sure that you convolve the excitatory "netIn" kernels
        with the directional transient cell output signal.
        '''
        d_srf = np.zeros((self.n_dirs, self.height, self.width))
        for k in range(self.n_dirs):
            d_srf[k] = self.layer3.get_time_const() * (- np.expand_dims(self.srf_cells[t-1, k, :, :], 0) + signal.convolve(self.dir_trans_out[t, k], self.short_range_excit_ker[k], "same"))
                        

        return d_srf

    def d_competition_layer(self, t):
        '''Compute the change in the Layer 4 cells: Spatial and directional competition in MT

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        Returns:
        -----------
        d_comp: ndarray. shape=(n_dirs, height, width)
            Derivative of the competition cells at time t

        TODO:
        - Make sure that you remember to include all 3 types of inputs to these competition neurons:
            1. Excitatory "netIn" kernel convolution with Layer 3 output
            2. Inhibitory "netIn" kernel convolution with Layer 3 output
            3. Inhibitory "netIn" with Layer 3 output from the opponent direction (DIRECTIONAL competition)
             (no convolution)
        '''
        d_comp = np.zeros((self.n_dirs, self.height, self.width))
        for k in range(self.n_dirs):
            opposite = self.get_opponent_direction(k)
            d_comp[k] = self.layer4.get_time_const() * (- np.expand_dims(self.comp_cells[t-1, k, :, :], 0) + (1-self.comp_cells[t-1, k, :, :]) * signal.convolve(self.srf_out[t, k], self.comp_excit_ker[k], "same") - (self.comp_cells[t-1, k, :, :] + self.layer4.get_lower_bound())*(signal.convolve(self.srf_out[t, k], self.comp_excit_ker[k], "same") + self.srf_out[t, opposite, :, :]) )

        return d_comp
        
    def d_long_range(self, t):
        '''Compute the change in the Layer 5 cells: Long range filter cells in MT

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        Returns:
        -----------
        d_lr: ndarray. shape=(n_dirs, height, width)
            Derivative of the long range filter cells at time t

        TODO:
        - Usual stuff
        - If Layer 6 (MSTd) not simulated, make the feedback signal 0.
        - Excitatory "netIn" kernel convolution with Layer 4 output
        '''
        pass

    def d_mstd_grouping(self, t):
        '''Compute the change in the Layer 6 cells: Direction grouping cells in MSTd

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        Returns:
        -----------
        d_comp: ndarray. shape=(n_dirs, height, width)
            Derivative of the competition cells at time t

        TODO:
        - Usual stuff
        - Use self.mstd_fb() to compute the inhibitory feedback signal. It should be based on the
        MSTd output signal at t-1.
        '''
        pass

    def mstd_fb(self, t, curr_mstd_out):
        '''Computes feedback signal from MSTd.
        Used as an inhibitory signal in both Layer 5 and Layer 6.

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...
        curr_mstd_out: ndarray. shape=(n_dirs, height, width)
            Output signal from MSTd at the current time (self.mstd_out[t])

        TODO:
        - Convolve each direction in the current MSTd output signal with the MSTd inhibitory kernel
        (i.e. smooth the signal within each direction map).
        - Apply the MSTd direction weight matrix (remember that? :) to the convolved signal result.
        This should lead to NO suppression from MSTd neurons tuned to the same preferred direction,
        maximal suppression from the opponent directioon, and moderate suppression for other
        non-preferred directions.
        HINT: broadcasting/new axes may be helpful here.
        '''
        pass

    def update_net(self, t):
        '''Solve for all the cell populations activity at the current time based on the previous
        time(s) and the incremental update — numerical integration (e.g. via Euler's Method).

        Goal: Solve for the derivatives at each level of the network then use numerical integation
        to compute the network activity at time t based on t-1 (and the derivatives).

        Parameters:
        -----------
        t: int.
            Current time step of the simulation: 1, 2, 3, ...

        NOTE:
        - You should minimally update all cells that have an associated ODE here.
        - You should also populate the "output" signal arrays (e.g. y, dir_trans_out, etc.). It
        is up to you where you want to do this...the output signals for Layer i are needed for
        Layer i+1.
        - To keep your code clean/modular, use the provided method stubs to solve for the derivatives
        in each area (rather than making this a gigantic unweldy method).
        - Use the level boolean variables (e.g. self.do_lvl1) to restrict the computation of
        different layers. This will be helpful for debugging.
        - IMPORTANT: You should solve for the derivatives and use numerical integration to get the
        activity of each level/area at the current time in an INTERLEAVED fashion.
        Example:
            - Solve for Layer 1 derivatives based on input at time t and cell activities at t-1.
            - Use Layer 1 derivatives to compute Layer 1 cells at time t
            - Use output of Layer 1 at time t to compute the Layer 2 derivatives
            ( their "input" at time t; like in Layer 1).
            - Use Layer 2 derivatives to compute Layer 2 cells at time t
            ...
        '''
        # layer 1
        if self.do_lvl1 == True:
            d_x, d_z = self.d_non_dir_transient_cells(t)
            self.x[t] = self.x[t-1] + d_x * self.dt
            self.z[t] = self.z[t-1] + d_z * self.dt
            self.y[t] = np.maximum(self.x[t] * self.z[t] - self.layer1.get_output_thres(), 0)

        # layer 2 
        if self.do_lvl2 == True:
            d_c, d_e = self.d_dir_transient_cells(t)
            
            self.dir_trans_inter_cells[t] = self.dir_trans_inter_cells[t-1] + d_c * self.dt
            self.dir_trans_cells[t] = self.dir_trans_cells[t-1] + d_e * self.dt
            self.dir_trans_out[t] = np.maximum(self.dir_trans_cells[t] - self.layer2.get_output_thres(), 0)

        # layer 3
        if self.do_lvl3 == True:
            d_srf = self.d_short_range_filter(t)
            self.srf_cells[t] = self.srf_cells[t-1] + d_srf * self.dt
            self.srf_out[t] = np.maximum(self.srf_cells[t] - self.layer3.get_output_thres(), 0)
            # print("self.srf_out[t] in update_net()", self.srf_out[t])

        # layer 4
        if self.do_lvl4:
            d_comp = self.d_competition_layer(t)
            self.comp_cells[t] = self.comp_cells[t-1] + d_comp * self.dt
            self.comp_out[t] = np.maximum(self.comp_cells[t] - self.layer4.get_output_thres(), 0)

        if self.do_lvl5:
            d_lr = self.d_long_range(t)
            self.lr_cells[t] = self.lr_cells[t-1] + d_lr * self.dt
            self.lr_out[t] = np.maximum(self.lr_cells[t] - self.layer5.get_output_thres(), 0)


        

    def simulate(self, inputs):
        '''Starts a simulation and have the network process the video (e.g. RDK).
        Initializes the network data structures then computes the network activity over time.

        Parameters:
        -----------
        inputs: ndarray. shape=(n_frames, height, width).
            The external input video (e.g. RDK)

        TODO:
        - Set the `self.inputs` instance variable.
        - Initialize the network structures (i.e. cell activity containers)
        - Start the temporal evolution of the network by updating the network activity at all
        time steps.
        '''
        self.inputs = inputs
        (n_frames, height, width) = inputs.shape
        n_steps = int(n_frames/self.dt)
        self.initialize(n_steps, height, width)
        for i in range (n_steps):
            self.update_net(i)


    def decode_direction(self, act, t, thres=0):
        '''Applies an activity (at time `t`) threshold to 2D spatial arrays of direction cells.
        Then decodes how much activation there is to each motion direction across the entire visual
        field in a simple way: sum of all the activity across all spatial positions (i.e. how much
        evidence is there that things are globally moving in each of these 8 motion directions?).

        Parameters:
        -----------
        act: ndarray. shape=(n_time_steps, n_dirs, n_rows (y), n_cols (x))
            Neural activity array.
        t: int.
            Simulation time step.
        thres: float >= 0.
            Value to subtract from the current activity at time t then values < 0 are set to 0.

        Returns:
        -----------
        ndarray. shape=(n_dirs,)
            Thresholded neural activation at time `t` summed across space for all preferred directions.
        '''
        return np.sum(np.maximum(act[t] - thres, 0), axis=(1, 2))
