import torch
import os
import yaml
from easydict import EasyDict

from utils.misc import mkdir_p

class CheckpointSaver:
    def __init__(self, checkpoint_dir ):
        self.checkpoint_dir = checkpoint_dir

    def save(self, model, epoch):

        mkdir_p(self.checkpoint_dir)

        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_epoch{}.pth".format(epoch))
        config_path = os.path.join(self.checkpoint_dir, "checkpoint_epoch{}.yaml".format(epoch))
        torch.save(model, checkpoint_path)

        config = {"TRAIN": {"CURRENT_EPOCH": epoch}}
        # TODO: Save other  parameters useful for restarting checkpoint

        with open(config_path, 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)


    def load(self, checkpoint_name):
        model_path = os.path.join(self.checkpoint_dir, "{0}.pth".format(checkpoint_name))
        config_path = os.path.join(self.checkpoint_dir, "{0}.yaml".format(checkpoint_name))

        if not os.path.exists(model_path):
            raise("Cannot find model checkpoint at {}".format(model_path))
        if not os.path.exists(config_path):
            raise("Cannot find config checkpoint at {}".format(config_path))

        model = torch.load(model_path)

        with open(config_path, "r") as config_file:
            config = EasyDict(yaml.load(config_file))

        return model, config
