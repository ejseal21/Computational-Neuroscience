'''preprocessing.py
Preprocess data before training neural network
CS443: Computational Neuroscience
Alice Zhang, Cole Turner, Ethan Seal
Project 2: Content Addressable Memory
'''
import numpy as np
from PIL import Image as im


def resize_imgs(imgs, width, height):
    '''Resizes list of PIL Image objects (`imgs`) to (`width` x `height`) resolution.
    Also, converts the images to grayscale if they have RGB color channels.

    Parameters:
    -----------
    imgs: Python list of PIL Image objects. len(imgs) = num_imgs. shape=variable.
        Each image may or may not have a RGB color depth dimension.
    width : int. Desired width with which to resize every image.
    height : int. Desired height with which to resize every image.

    Returns:
    -----------
    ndarray of uint8s. shape=(num_imgs, height, width).
        Grayscale images
    '''
    imgs = imgs.copy()
    for img in range(len(imgs)):
        imgs[img] = np.array(imgs[img].convert('L').resize((width,height)))

    return np.array(imgs)

def img2binaryvectors(data, bipolar=True):
    '''Transform grayscale images into normalized, centered, binarized 1D feature vectors with
    bipolar values (-1, +1)

    Parameters:
    -----------
    data: ndarray. shape=(N, Iy (height), Ix (width)).
        Grayscale images

    Returns:
    -----------
    ndarray of -1 and +1s only. shape=(N, Iy*Ix).

    TODO:
    - Normalize each image based on its dynamic range.
    - Center the image then threshold at 0 so that values are either -1 or +1.
    - Reshape so that the result is a 1D vector (see shape above)
    '''
    #normalize based on dynamic range
    data = data / np.expand_dims(np.max(data, axis=(1, 2)), axis=(1, 2))
    data = data - np.expand_dims(np.min(data, axis=(1, 2)), axis=(1, 2))
    
    #subtract median so half of the values are below 0 and half are above
    data = data - np.expand_dims(np.median(data, axis=(1, 2)), axis=(1, 2))

    data = np.sign(data)
    data = np.reshape(data, (data.shape[0], -1))
    return data.astype(np.int64)


def vec2img(feat_vecs, width, height):
    '''Inflates each 1D feature vector into a `width` x `height` grayscale image.

    Parameters:
    -----------
    feat_vecs: ndarray. shape=(N, height*width).
        1D feature vectors
    width : int. Original width of each image before it was flattened into a 1D vector.
    height : int. Original height of each image before it was flattened into a 1D vector.

    Returns:
    -----------
    ndarray. shape=(N, height, width).
        Inflated version of `feat_vecs` into images
    '''
    imgs = np.reshape(feat_vecs, (feat_vecs.shape[0], width, height))
    return imgs


def recall_error(orig_data, recovered_data, tol=0.5):
    '''Measure the error between training data `orig_data` and the memories recalled by the network
    `recovered_data`.

    Parameters:
    -----------
    orig_data: ndarray. shape=(N, height*width).
        1D feature vectors used to train network
    recovered_data: ndarray. shape=(N, height*width).
        1D vectors of recovered memories from trained network

    Returns:
    -----------
    float. error rate, a proportion between 0 and 1, of how many vector components are mismatched.
    '''
    [N, hw] = orig_data.shape
    count = 0
    for i in range(N):
        for j in range(hw):
            if orig_data[i, j] != recovered_data[i, j]:
                count = count + 1
    return count/(N*hw)
    # what is tol as a parameter??????