from __future__ import print_function
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

import numpy as np


class Evaluator(object):
    def __init__(self, volatile=True):

        self.volatile = volatile

    @staticmethod
    def evaluate(epoch, _loader, model, multi_loss, device):

        writer4 = SummaryWriter('run/valid_loss')
        writer5 = SummaryWriter('run/valid_acc')

        valid_loss = 0
        valid_n_iter = 0
        valid_n_samples = 0

        valid_correct_seq = 0
        valid_correct_length = 0
        valid_correct_digits = np.array([0] * 5)
        valid_digit_seq = np.array([0] * 5)

        valid_detailed_accuracy = [None] * 6

        # Iterate over valid data
        print("\n\nIterating over validation data...")
        for i, batch in enumerate(tqdm(_loader)):

            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)

            target_ndigits = targets[:, 0].long()
            target_ndigits = target_ndigits.to(device)

            target_digits = targets[:, 1:].long()
            target_digits = target_digits.to(device)

            # Set model to evaluate mode and do a forward pass
            outputs = model.eval()(inputs)

            total_loss = multi_loss(outputs, target_ndigits, target_digits)

            # Statistics
            valid_loss += total_loss.item()
            valid_n_iter += 1

            _, predicted_num_digits = torch.max(outputs[0].data, 1)

            predicted_digits_data = []

            for j in range(5):
                predicted_digits_data.append(outputs[j + 1].data)

            predicted_digits_data = torch.stack(predicted_digits_data, 1)
            _, predicted_digits = torch.max(predicted_digits_data, 2)

            for k in range(predicted_digits.size(0)):
                target_length = target_ndigits[k].item()

                valid_digit_seq[:target_length] += 1

                if target_length == predicted_num_digits[k].item():
                    valid_correct_length += 1

                curated_seq = predicted_digits[k]
                curated_seq[predicted_num_digits[k]:] = -1
                if curated_seq.eq(target_digits[k]).sum().item() == 5:
                    valid_correct_seq += 1

                for j in range(5):

                    if curated_seq[j].item() != -1 and curated_seq[j].item() == target_digits[k][j].item():
                        valid_correct_digits[j] += 1

            valid_n_samples += predicted_digits.size(0)

        valid_avg_loss = valid_loss / valid_n_iter
        valid_accuracy = valid_correct_seq / valid_n_samples

        valid_detailed_accuracy[0] = valid_correct_length / valid_n_samples

        for i in range(1, 6):
            if valid_digit_seq[i-1] != 0:
                valid_detailed_accuracy[i] = "{:.4f}".format((valid_correct_digits[i-1] / valid_digit_seq[i-1]))
            else:
                valid_detailed_accuracy[i] = "No examples of length {} digit(s) or more.".format(i)

        # adding log
        writer4.add_scalar('Loss', valid_avg_loss, epoch + 1)
        writer5.add_scalar('Accuracy', valid_accuracy, epoch + 1)
        writer5.add_scalar('Length Accuracy', valid_detailed_accuracy[0], epoch + 1)

        for k in range(5):
            if valid_digit_seq[k] != 0:
                writer5.add_scalar('Digit {} Accuracy'.format(k + 1), valid_correct_digits[k] / valid_digit_seq[k],
                                   epoch + 1)

        return valid_avg_loss, valid_accuracy, valid_detailed_accuracy

