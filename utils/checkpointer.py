import torch
import os
import yaml
from easydict import EasyDict

from utils.misc import mkdir_p
from utils.config import cfg, cfg_from_file

class CheckpointSaver:
    def __init__(self, checkpoint_dir ):
        self.checkpoint_dir = checkpoint_dir

    def save(self, model, epoch):

        mkdir_p(self.checkpoint_dir)

        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_epoch{}.pth".format(epoch))
        torch.save(model, checkpoint_path)

        # Augment the current config file with useful parameters for resuming training
        cfg.TRAIN.CURRENT_EPOCH = epoch

        config_path = os.path.join(self.checkpoint_dir, "checkpoint_epoch{}.yml".format(epoch))
        with open(config_path, 'w') as config_file:
            yaml.dump(cfg, config_file, default_flow_style=False)


    def load(self, checkpoint_name):
        model_path = os.path.join(self.checkpoint_dir, "{0}.pth".format(checkpoint_name))
        config_path = os.path.join(self.checkpoint_dir, "{0}.yml".format(checkpoint_name))

        if not os.path.exists(model_path):
            raise FileNotFoundError("Cannot find model checkpoint at {}".format(model_path))
        if not os.path.exists(config_path):
            raise FileNotFoundError("Cannot find config checkpoint at {}".format(config_path))

        model = torch.load(model_path)

        cfg_from_file(config_path)

        return model
