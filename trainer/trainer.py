from __future__ import print_function

import copy
import time
import os
import torch
from tqdm import tqdm, tqdm_notebook

from utils.checkpointer import CheckpointSaver
from trainer.evaluator import Evaluator
from tensorboardX import SummaryWriter


def multi_loss(outputs, target_ndigits, target_digits):
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

    loss_seqlen = torch.nn.CrossEntropyLoss()(outputs[0], target_ndigits)
    loss_digit1 = loss_function(outputs[1], target_digits[:, 0])
    loss_digit2 = loss_function(outputs[2], target_digits[:, 1])
    loss_digit3 = loss_function(outputs[3], target_digits[:, 2])
    loss_digit4 = loss_function(outputs[4], target_digits[:, 3])
    loss_digit5 = loss_function(outputs[5], target_digits[:, 4])

    return loss_seqlen + loss_digit1 + loss_digit2 + loss_digit3 + loss_digit4 + loss_digit5


def to_np(x):
    """
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    :param x:
    :return:
    """
    return x.data.cpu().numpy()


def train_model(model, train_loader, valid_loader, device,cfg):

    current_epoch = cfg.TRAIN.CURRENT_EPOCH
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    lr = cfg.TRAIN.LR
    checkpoint_dir = cfg.CHECKPOINT_DIR
    output_dir = cfg.OUTPUT_DIR
    momentum = cfg.TRAIN.MOM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    '''
    Training loop.

    Parameters
    ----------
    model : obj
        The model.
    train_loader : obj
        The train data loader.
    valid_loader : obj
        The validation data loader.
    device : str
        The type of device to use ('cpu' or 'gpu').
    num_eopchs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    output_dir : str
        path to the directory where to save the model.

    '''

    checkpoint = CheckpointSaver(checkpoint_dir)

    since = time.time()
    dirName = 'run'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory", dirName,  "created ")
    else:
        print("Directory", dirName,  "already exists")

    writer1 = SummaryWriter('run/train_loss')

    writer2 = SummaryWriter('run/mtrain')

    writer3 = SummaryWriter('run/mtrain2')
    writer4 = SummaryWriter('run/mtrain_out')

    writer5 = SummaryWriter('run/Model_arch')

    model = model.to(device)
    train_loss_history = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                           verbose=True, min_lr=1e-5)
    print("# Start training #")
    for epoch in range(current_epoch, num_epochs):

        train_loss = 0
        train_n_iter = 0

        # Iterate over train data
        print("\n\nIterating over training data...")
        # set a progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, batch in pbar:
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)

            target_ndigits = targets[:, 0].long()
            target_ndigits = target_ndigits.to(device)

            target_digits = targets[:, 1:].long()
            target_digits = target_digits.to(device)

            # Forward
            outputs = model.train()(inputs)

            # Multi-task learning loss
            total_loss = multi_loss(outputs, target_ndigits, target_digits)

           # Zero the gradient buffer
            optimizer.zero_grad()

            # Backward
            total_loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += total_loss.item()
            train_n_iter += 1

            # add the model graph

            writer1.add_scalar('Train/loss', train_loss / train_n_iter, i)

            # log the layers and layers gradient histogram and distributions

            #for tag, value in model.named_parameters():
            #    tag = tag.replace('.', '/')
            #    writer2.add_histogram('model/(train)' + tag, to_np(value), i + 1)
            #    writer3.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)

            # log the outputs given by the model (The segmentation)
            #writer4.add_image('model/(train)output', make_grid(outputs[0].data), i + 1)


            # update progress bar status
            pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f(avg) '
                                 % (epoch + 1, num_epochs, train_loss / train_n_iter))

        #writer5.add_graph(model, inputs)

        train_loss_history.append(train_loss / train_n_iter)

        valid_loss_history, valid_accuracy, valid_accuracy_history, best_model = \
            Evaluator().evaluate(valid_loader, model, multi_loss, device, output_dir)

        scheduler.step(valid_loss_history[-1])

        if epoch % 5 == 0:
            checkpoint.save(model, epoch, cfg=cfg)

        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('\tTrain Loss: {:.4f}'.format(train_loss / train_n_iter))
        print('\tValid Loss: {:.4f}'.format(valid_loss_history[-1]))
        print('\tValid Accuracy: {:.4f}'.format(valid_accuracy))

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)

    # return score for bayesian optimization
    return valid_accuracy
