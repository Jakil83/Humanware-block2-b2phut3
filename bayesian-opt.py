from __future__ import print_function

import argparse
import datetime
import os
import sys
from shutil import copyfile

import dateutil.tz
import skopt
import torch

# from models.baselines import BaselineCNN, ConvNet, BaselineCNNDropout
from models.vgg import VGG
from trainer.trainer import train_model
from utils.checkpointer import CheckpointSaverCallback
from utils.config import cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p

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
                        help='''metadata_filename will be the absolute paths to the metadata files 
                        of the data (order is [train, extra] if both are provided).''')

    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints",
                        help='''checkpoint_dir will be the absolute path to the directory used 
                        for checkpointing''')

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

    parser.add_argument("--num_calls", type=int,
                        default=10,
                        help='''number of iteration to be performed by the bayesian optimization.
                          It should any number larger than 10''')

    parser.add_argument("--bayesian_checkpoint_name", type=str,
                        default=None,
                        help='''the name of the checkpoint to resume the bayesian optimization from.  
                        If set to None then the bayesian optimization will start from the beginning''')

    # parser.add_argument("--checkpoint_name", type=str,
    #                     default=None,
    #                     help='''the name of the checkpoint to resume training from.
    #                     If set to None then the training will start from the beginning''')

    args = parser.parse_args()
    return args


def load_config(args):
    """
    Load the config .yml file.

    """

    if args.cfg is None:
        raise Exception("No config file specified.")

    cfg = cfg_from_file(args.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # print('timestamp: {}'.format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.INPUT_DIR = args.dataset_dir
    cfg.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.METADATA_FILENAME = args.metadata_filename
    cfg.OUTPUT_DIR = os.path.join(
        args.results_dir,
        '%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp))

    mkdir_p(cfg.OUTPUT_DIR)
    copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    # print('Data dir: {}'.format(cfg.INPUT_DIR))
    # print('Output dir: {}'.format(cfg.OUTPUT_DIR))

    # print('Using config:')
    # pprint.pprint(cfg)
    return cfg


def train_model_opt(parameters):
    print("Training model with parameters: {}\n\n\n".format(parameters))
    args = parse_args()

    # Load the config file
    cfg = load_config(args)

    (train_loader,
     valid_loader) = prepare_dataloaders(cfg)

    vgg19 = VGG("VGG19", num_classes_length=7, num_classes_digits=10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    cfg.TRAIN.LR = parameters[0]
    cfg.TRAIN.BATCH_SIZE = parameters[1]

    return train_model(vgg19,
                       cfg=cfg,
                       train_loader=train_loader,
                       valid_loader=valid_loader,
                       device=device)


if __name__ == '__main__':
    args = parse_args()

    # Load the config file
    # cfg = load_config(args)

    # Make the results reproductible
    # fix_seed(cfg.SEED)

    space = [skopt.space.Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),
             skopt.space.Integer(16, 128, name='batch_size')
             ]

    checkpoint_path = os.path.join(args.checkpoint_dir, "bayesian_checkpoint.pkl")
    checkpoint_saver = CheckpointSaverCallback(checkpoint_path, compress=9)
    if args.bayesian_checkpoint_name:
        res_gp = skopt.load(checkpoint_path)
        print("Resuming from iteration: {}\n\n".format(len(res_gp.x_iters)))
        skopt.gp_minimize(train_model_opt, space, x0=res_gp.x_iters, y0=res_gp.func_vals, n_calls=args.num_calls,
                          callback=[CheckpointSaverCallback], random_state=0)
    else:
        print("Starting bayesian optimization\n\n")
        res_gp = skopt.gp_minimize(train_model_opt, space, n_calls=args.num_calls, callback=[checkpoint_saver],
                                   random_state=0)

    print("Best accuracy: {0}".format(-res_gp.fun))
    print("Best parameters: {0}".format(res_gp.x))
