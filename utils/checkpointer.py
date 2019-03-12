import os

import skopt
import torch
import yaml

from utils.config import cfg_from_file
from utils.misc import mkdir_p


class CheckpointSaver:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def save(self, model, epoch, cfg):
        """
        Save the model and the config
        :param model: The PyTorch model
        :param epoch: The current epoch
        :param cfg: The training configuration of the model
        """
        mkdir_p(self.checkpoint_dir)

        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_epoch{}.pth".format(epoch))
        torch.save(model, checkpoint_path)

        # Augment the current config file with useful parameters for resuming training
        # Save current epoch in train and train_extra because we do not know which will be used
        cfg.TRAIN.CURRENT_EPOCH = epoch

        config_path = os.path.join(self.checkpoint_dir, "checkpoint_epoch{}.yml".format(epoch))
        with open(config_path, 'w') as config_file:
            yaml.dump(cfg, config_file, default_flow_style=False)

        print("Checkpointing new model ...")

    def load(self, checkpoint_name):
        """
        Load the model and the config based on the checkpoint name
        The config is also loaded and can be accessed with the cfg object
        :param checkpoint_name: The name of the checkpoint without the file extension
        :return: The PyTorch model
        """

        model_path = os.path.join(self.checkpoint_dir, "{0}.pth".format(checkpoint_name))
        config_path = os.path.join(self.checkpoint_dir, "{0}.yml".format(checkpoint_name))

        if not os.path.exists(model_path):
            raise FileNotFoundError("Cannot find model checkpoint at {}".format(model_path))
        if not os.path.exists(config_path):
            raise FileNotFoundError("Cannot find config checkpoint at {}".format(config_path))

        model = torch.load(model_path)

        cfg_from_file(config_path)

        return model


class CheckpointSaverCallback(object):
    """
    Save current state after each iteration with `skopt.dump`.
    Callback from skopt

    Example usage:
        import skopt

        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])

    Parameters
    ----------
    * `checkpoint_path`: location where checkpoint will be saved to;
    * `dump_options`: options to pass on to `skopt.dump`, like `compress=9`
    """

    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        skopt.dump(res, self.checkpoint_path, **self.dump_options)
