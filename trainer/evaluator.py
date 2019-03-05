from __future__ import print_function
from tqdm import tqdm, tqdm_notebook
import copy
import numpy as np
import torch
from tensorboardX import SummaryWriter


class Evaluator(object):
    def __init__(self, volatile=True):

        self.volatile = volatile

    def evaluate(self, _loader, model, multi_loss, loss_ndigits, device, output_dir):

        writer6 = SummaryWriter('run/valid_loss')
        writer7 = SummaryWriter('run/vseq_acc')
        writer8 = SummaryWriter('run/vdigits_acc')
        valid_loss = 0
        valid_n_iter = 0
        valid_seqlen_correct = 0
        valid_all_digits_correct = 0
        valid_n_samples = 0
        valid_loss_history = []
        valid_accuracy_history = []
        valid_best_accuracy = 0

        # Set model to evaluate mode

        model.eval()

        # Iterate over valid data
        print("Iterating over validation data...")
        for i, batch in enumerate(tqdm(_loader)):
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)

            target_ndigits = targets[:, 0].long()
            target_ndigits = target_ndigits.to(device)

            target_digits = targets[:, 1:].long()
            target_digits = target_digits.to(device)

            # Forward
            output_seqlen, output_digits = model(inputs)

            total_loss = multi_loss(loss_ndigits, output_seqlen, output_digits, target_ndigits, target_digits)

            # Statistics
            valid_loss += total_loss.item()
            valid_n_iter += 1

            _, pred_seqlen = torch.max(output_seqlen.data, 1)
            _, pred_digit1 = torch.max(output_digits[0].data, 1)
            _, pred_digit2 = torch.max(output_digits[1].data, 1)
            _, pred_digit3 = torch.max(output_digits[2].data, 1)
            _, pred_digit4 = torch.max(output_digits[3].data, 1)
            _, pred_digit5 = torch.max(output_digits[4].data, 1)

            valid_seqlen_correct += (pred_seqlen == target_ndigits).sum().item()

            # TODO: Make it work only with torch tensors
            valid_digit1 = (pred_digit1 == target_digits[:, 0]).cpu().numpy().astype(bool)
            valid_digit2 = (pred_digit2 == target_digits[:, 1]).cpu().numpy().astype(bool)
            valid_digit3 = (pred_digit3 == target_digits[:, 2]).cpu().numpy().astype(bool)
            valid_digit4 = (pred_digit4 == target_digits[:, 3]).cpu().numpy().astype(bool)
            valid_digit5 = (pred_digit5 == target_digits[:, 4]).cpu().numpy().astype(bool)

            valid_all_digits_correct += np.logical_and.reduce(
                (valid_digit1, valid_digit2, valid_digit3, valid_digit4, valid_digit5)).sum()

            valid_n_samples += target_ndigits.size(0)

            # adding logg
            writer6.add_scalar('Valid Loss', valid_loss / valid_n_iter, i)
            writer7.add_scalar('Valid seq_acc', valid_seqlen_correct / valid_n_samples, i)
            writer8.add_scalar('Valid_digit_acc', valid_all_digits_correct / valid_n_samples, i)

        valid_loss_history.append(valid_loss / valid_n_iter)
        valid_accuracy = valid_seqlen_correct / valid_n_samples
        valid_digit_acc = valid_all_digits_correct / valid_n_samples

        if valid_accuracy > valid_best_accuracy:
            valid_best_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)
        valid_accuracy_history.append(valid_accuracy)

        return valid_loss_history, valid_accuracy, valid_digit_acc, valid_accuracy_history, best_model

