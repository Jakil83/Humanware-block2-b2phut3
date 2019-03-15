from __future__ import print_function

import argparse
import datetime
import os
import sys
from shutil import copyfile

import dateutil.tz
import torch

from models.vgg import VGG
from trainer.trainer import train_model
from utils.checkpointer import CheckpointSaver
from utils.config import cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p, fix_seed

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    """
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    """
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', type=str,
                        default=None,
                        help='''optional config file,
                             e.g. config/base_config.yml''')

    parser.add_argument("--metadata_filename", nargs='+', type=str,
                        default=['data/SVHN/train_metadata.pkl', 'data/SVHN/extra_metadata.pkl'],
                        help='''metadata_filename will be the absolute
                                paths to the metadata files of the data (order is [train, extra] 
                                if both are provided).''')

    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints",
                        help='''checkpoint_dir will be the absolute path to the directory used for checkpointing''')

    parser.add_argument("--dataset_dir", nargs='+', type=str,
                        default=['data/SVHN/train', 'data/SVHN/extra'],
                        help='''dataset_dir will be the absolute path
                                 to the data to be used for
                                 training (order is [train, extra] if both are provided).''')

    parser.add_argument("--results_dir", type=str,
                        default='results/',
                        help='''results_dir will be the absolute
                        path to a directory where the output of
                        your training will be saved.''')

    parser.add_argument("--checkpoint_name", type=str,
                        default=None,
                        help='''the name of the checkpoint to resume training from.  
                        If set to None then the training will start from the beginning''')

    return parser.parse_args()


def load_config(args):
    """
    Load the config .yml file.

    """

    if args.cfg is None:
        raise Exception("No config file specified.")

    cfg = cfg_from_file(args.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    print('timestamp: {}'.format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.INPUT_DIR = args.dataset_dir
    cfg.METADATA_FILENAME = args.metadata_filename
    cfg.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.OUTPUT_DIR = os.path.join(
        args.results_dir,
        '%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp))

    mkdir_p(cfg.OUTPUT_DIR)
    copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    print('Data dir: {}'.format(cfg.INPUT_DIR))
    print('Output dir: {}'.format(cfg.OUTPUT_DIR))
    return cfg


if __name__ == '__main__':
    args = parse_args()

    if args.checkpoint_name:
        checkpoint = CheckpointSaver(args.checkpoint_dir)

        # Load model from checkpoint
        model, cfg = checkpoint.load(args.checkpoint_name)

        # Make results reproducible
        fix_seed(cfg.SEED)
    else:

        # Load the config file
        cfg = load_config(args)

        # Make results reproducible
        fix_seed(cfg.SEED)

        # Define model architecture
        model = VGG('VGG19', num_classes_length=7, num_classes_digits=10)

    # Prepare data
    (train_loader,
     valid_loader) = prepare_dataloaders(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    train_model(model,
                cfg=cfg,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=device)
