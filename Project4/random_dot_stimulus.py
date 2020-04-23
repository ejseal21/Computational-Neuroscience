'''random_dot_stimulus.py
Creates noisy random dot kinematogram stimuli for motion decision making tasks
CS443: Computational Neuroscience
Alice Cole Ethan
Project 4: Motion detection and perception
'''
import numpy as np


def make_random_dot_stimulus(n_frames, height=50, width=100, dir_rc=(0, 1), n_dots=50,
                             dot_value=1, noise_prop=0):
    '''Makes a random dot kinematogram (RDK) stimulus video

    Parameters:
    -----------
    n_frames: int.
        Number of frames of the generated video
    height: int.
        Number of rows in the video.
    width: int.
        Number of cols in the video.
    dir_rc: tuple of int.
        The offset of the dots across successive frames. dir_rc[0] is shift in ROWS. +1 means down.
        dir_rc[1] is shift in COLS. +1 means right.
        Typical element values are -1, 0, +1 for single pixel shifts.
    n_dots: int.
        How many non-noise dots are placed in each frame.
    dot_value: float.
        Background values are 0. Dots are defined to have value `dot_value`.
        Example: white dots are 1, black background is 0.
    noise_prop. float.
        Proportion of n_dots are ADDED to each frame in random positions.
        Example: if `n_dots` = 50 and `noise_prop` = 0.1, that means 5 fots are placed randomly in
        each frame.
        NOTE: I assumed for simplicity that noise dots can coincide with non-noise dots
        (i.e. number of dots may not be constant across frames). You can get fancy and ensure that
        noise dots always occupy empty background positions if you'd like.

    Returns:
    -----------
    ndarray. shape=(n_frames, height, width).
        The RDK
    '''
    video = np.zeros((n_frames, height, width))
    frame = np.zeros((height * width))
    
    #set n_dots random dots to 1
    frame[np.random.choice(np.arange(height * width), size=n_dots)] = dot_value

    frame = np.reshape(frame, (height, width))

    for n in range(n_frames):
        #up
        if dir_rc[0] == -1:
            frame = np.vstack((frame[1:, :], frame[0, :]))
        #down
        if dir_rc[0] == 1:
            frame = np.vstack((frame[-1, :], frame[:-1, :]))
        #left
        if dir_rc[1] == -1:
            frame = np.hstack((frame[:, 1:], np.expand_dims(frame[:, 0], 1)))
        #right
        if dir_rc[1] == 1:
            frame = np.hstack((np.expand_dims(frame[:, -1], 1), frame[:, :-1]))
        
        add_noise = np.copy(frame)
        add_noise[np.unravel_index(np.random.choice(np.arange(height * width), size=int(n_dots*noise_prop)), shape=(height, width))] = dot_value
        video[n, :, :] = add_noise
        # mask = np.random.choice([True, False], (height, width), replace=True, p=(noise_prop, 1-noise_prop))
        # video[n, mask] = dot_value
        
        # print(video[0].size)
        # print(np.count_nonzero(video[0]))
        # break

    return video