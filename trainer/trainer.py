from __future__ import print_function

import copy
import numpy as np
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


def train_model(model, train_loader, valid_loader, device, cfg):

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
    dir_name = 'run'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory", dir_name,  "created ")
    else:
        print("Directory", dir_name,  "already exists")

    writer1 = SummaryWriter('run/train_loss')
    writer2 = SummaryWriter('run/train_acc')
    writer3 = SummaryWriter('run/Model_arch')

    valid_best_accuracy = 0
    best_model = None

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5,
                                                           verbose=True, min_lr=1e-5)
    print("# Start training #")
    for epoch in range(current_epoch, num_epochs):

        train_loss = 0
        train_n_iter = 0
        train_n_samples = 0

        train_correct_seq = 0
        train_correct_length = 0
        train_correct_digits = np.array([0] * 5)
        train_digit_seq = np.array([0] * 5)

        train_detailed_accuracy = [None] * 6
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

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward
            outputs = model.train()(inputs)

            # Multi-task learning loss
            total_loss = multi_loss(outputs, target_ndigits, target_digits)

            # Backward
            total_loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += total_loss.item()
            train_n_iter += 1

            _, predicted_num_digits = torch.max(outputs[0].data, 1)

            predicted_digits_data = []

            for j in range(5):
                predicted_digits_data.append(outputs[j + 1].data)

            predicted_digits_data = torch.stack(predicted_digits_data, 1)
            _, predicted_digits = torch.max(predicted_digits_data, 2)

            for k in range(predicted_digits.size(0)):
                target_length = target_ndigits[k].item()

                train_digit_seq[:target_length] += 1

                if target_length == predicted_num_digits[k].item():
                    train_correct_length += 1

                curated_seq = predicted_digits[k]
                curated_seq[predicted_num_digits[k]:] = -1
                if curated_seq.eq(target_digits[k]).sum().item() == 5:
                    train_correct_seq += 1

                for j in range(5):

                    if curated_seq[j].item() != -1 and curated_seq[j].item() == target_digits[k][j].item():
                        train_correct_digits[j] += 1

            train_n_samples += predicted_digits.size(0)

            # update progress bar status
            pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f(avg) '
                                 % (epoch + 1, num_epochs, train_loss / train_n_iter))

        train_avg_loss = train_loss / train_n_iter
        train_accuracy = train_correct_seq / train_n_samples

        train_detailed_accuracy[0] = train_correct_length / train_n_samples

        for i in range(5):
            if train_digit_seq[i] != 0:
                train_detailed_accuracy[i+1] = train_correct_digits[i] / train_digit_seq[i]

        # adding log

        writer1.add_scalar('Loss', train_avg_loss, epoch + 1)
        writer2.add_scalar('Accuracy', train_accuracy, epoch + 1)
        writer2.add_scalar('Length Accuracy', train_detailed_accuracy[0], epoch + 1)

        for k in range(1, 6):
            if train_detailed_accuracy[k] is not None:
                writer2.add_scalar('Digit {} Accuracy'.format(k), train_detailed_accuracy[k],
                                   epoch + 1)

        valid_avg_loss, valid_accuracy, valid_detailed_accuracy = \
            Evaluator().evaluate(epoch, valid_loader, model, multi_loss, device)

        scheduler.step(valid_accuracy)

        if valid_accuracy > valid_best_accuracy:
            valid_best_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)

        if epoch % 5 == 0:
            checkpoint.save(model, epoch, cfg=cfg)

        # Add the model graph
        writer3.add_graph(model, inputs)

        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('\tTrain Loss: {:.4f}'.format(train_avg_loss))
        print('\tTrain Accuracy: {:.4f}'.format(train_accuracy))
        print('\tValid Loss: {:.4f}'.format(valid_avg_loss))
        print('\tValid Accuracy: {:.4f}'.format(valid_accuracy))

        print("\nDetailed train accuracies:\n")

        print('\tLength Accuracy: {:.4f}'.format(train_correct_length / train_n_samples))

        for i in range(5):
            if train_digit_seq[i] != 0:
                print('\tDigit {} Accuracy: {:.4f}'.format(i + 1, train_correct_digits[i] / train_digit_seq[i]))
            else:
                print('\tDigit {0} Accuracy: No examples of length {0} digit(s) or more.'.format(i + 1))

        print("\nDetailed valid accuracies:\n")

        print('\tLength Accuracy: {:.4f}'.format(valid_detailed_accuracy[0]))

        for i in range(5):
            print('\tDigit {} Accuracy: {}'.format(i + 1, valid_detailed_accuracy[i + 1]))

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)

    # return score for bayesian optimization
    return valid_best_accuracy
