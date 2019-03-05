from __future__ import print_function
import os
import sys

import argparse
import dateutil.tz
import datetime
import numpy as np
import pprint
import random
from shutil import copyfile

import torch

import skopt


from utils.config import cfg, cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p, fix_seed
# from models.baselines import BaselineCNN, ConvNet, BaselineCNN_dropout
from models.vgg import VGG
# from models.resnet import ResNet18
from trainer.trainer import train_model

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', type=str,
                        default=None,
                        help='''optional config file,
                             e.g. config/base_config.yml''')
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints",
                        help='''checkpoint_dir will be the absolute path to the directory used for checkpointing''')
    parser.add_argument("--metadata_filename", type=str,
                        default='data/SVHN/train_metadata.pkl',
                        help='''metadata_filename will be the absolute
                                path to the directory to be used for
                                training.''')

    parser.add_argument("--dataset_dir", type=str,
                        default='data/SVHN/train/',
                        help='''dataset_dir will be the absolute path
                                to the directory to be used for
                                training''')

    parser.add_argument("--results_dir", type=str,
                        default='results/',
                        help='''results_dir will be the absolute
                        path to a directory where the output of
                        your training will be saved.''')

    args = parser.parse_args()
    return args


def load_config():
    '''
    Load the config .yml file.

    '''
    args = parse_args()

    if args.cfg is None:
        raise Exception("No config file specified.")

    cfg_from_file(args.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    print('timestamp: {}'.format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.INPUT_DIR = args.dataset_dir
    cfg.METADATA_FILENAME = args.metadata_filename
    cfg.OUTPUT_DIR = os.path.join(
        args.results_dir,
        '%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp))

    mkdir_p(cfg.OUTPUT_DIR)
    copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    print('Data dir: {}'.format(cfg.INPUT_DIR))
    print('Output dir: {}'.format(cfg.OUTPUT_DIR))

    print('Using config:')
    pprint.pprint(cfg)


def train_model_opt(parameters):
    lr = parameters[0]
    print(parameters)
    (train_loader,
     valid_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=1000,
        valid_split=cfg.TRAIN.VALID_SPLIT)

    vgg19 = VGG("VGG19", num_classes_length=7, num_classes_digits=10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    return -   train_model(vgg19,
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           # num_epochs=cfg.TRAIN.NUM_EPOCHS,
                           num_epochs=1,
                           device=device,
                           lr=lr,
                           output_dir=cfg.OUTPUT_DIR)


if __name__ == '__main__':
    # Load the config file
    load_config()

    # Make the results reproductible
    fix_seed(cfg.SEED)

    space = [skopt.space.Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),
             skopt.space.Categorical(["VGG11", "VGG13", "VGG16", "VGG19"])]

    res_gp = skopt.gp_minimize(train_model_opt, space, n_calls=10,
                               random_state=0)

    print("Best score: {0}".format(-res_gp.fun))
    print("Best lr: {0}".format(res_gp.x[0]))
