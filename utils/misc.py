import os

import errno
import pickle
import numpy as np
import random
import torch


def mkdir_p(path):
    """
    Make a directory.

    Parameters
    ----------
    path : str
        path to the directory to make.

    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_obj(obj, filename):
    """
    Save an object in pickle format.

    Parameters
    ----------
    obj : object
        Object to save.
    filename: str
        path/filename to saved the object.

    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """
    Load an object from pickle format.

    Parameters
    ----------
    filename: str
        path/filename of the saved object.

    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def fix_seed(seed):
    """
    Fix the seed.

    Parameters
    ----------
    seed: int
        The seed to use.

    """
    print('pytorch/random seed: {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
