from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict
import numpy as np


def cfg_from_file(filename):
    """
     Load a config file and merge it into the default options.

     Parameters
     ----------
     filename : string
         Path to filename.
     """
    import yaml
    with open(filename, 'r') as f:
        return edict(yaml.load(f))

