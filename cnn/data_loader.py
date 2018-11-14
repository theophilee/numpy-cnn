from __future__ import division
from scipy.io import loadmat
import numpy as np
import os


def convert_scipy_mat_to_numpy_arrays(filename, output_dir):
    """Converts the matlab mat file to numpy arrays

    Arguments:
        filename (str): The path to the "all_mnist.mat" file
        output_dir (str): The directory to store all the numpy arrays
    Returns:
        None
    """
    data = loadmat(filename)

    xtrain = np.concatenate([data['train{0}'.format(ix)] for ix in range(10)],
                            axis=0)
    ytrain = np.concatenate(
        [ix * np.ones((data['train{0}'.format(ix)].shape[0], 1))
         for ix in range(10)], axis=0
    )
    xtest = np.concatenate([data['test{0}'.format(ix)] for ix in range(10)],
                           axis=0)
    ytest = np.concatenate(
        [ix * np.ones((data['test{0}'.format(ix)].shape[0], 1))
         for ix in range(10)], axis=0
    )
    train_x_filename = os.path.join(output_dir, "train_x.npy")
    train_y_filename = os.path.join(output_dir, "train_y.npy")

    np.save(train_x_filename, xtrain.astype("uint8"))
    np.save(train_y_filename, ytrain.astype("uint8"))

    test_x_filename = os.path.join(output_dir, "test_x.npy")
    test_y_filename = os.path.join(output_dir, "test_y.npy")

    np.save(test_x_filename, xtest.astype("uint8"))
    np.save(test_y_filename, ytest.astype("uint8"))


def load_mnist(data_dir, dtype="float64"):
    """
    Load matrices from data_dir. Return matrices with dtype
    """
    train_x = np.load(os.path.join(data_dir, "train_x.npy")).astype(dtype)
    test_x = np.load(os.path.join(data_dir, "test_x.npy")).astype(dtype)

    train_y = np.load(os.path.join(data_dir, "train_y.npy")).astype(dtype)
    test_y = np.load(os.path.join(data_dir, "test_y.npy")).astype(dtype)

    # Shuffle the data and normalize
    p_ix = np.random.permutation(train_x.shape[0])
    train_x = train_x[p_ix] / 255.
    train_y = train_y[p_ix]

    # Now return
    return train_x, train_y.flatten(), test_x, test_y.flatten()