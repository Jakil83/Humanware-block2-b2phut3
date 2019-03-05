from __future__ import print_function
from tqdm import tqdm, tqdm_notebook
import copy
import torch
from tensorboardX import SummaryWriter


class Evaluator(object):
    def __init__(self, volatile=True):

        self.volatile = volatile

    def evaluate(self, _loader, model, multi_loss, device, output_dir):

        writer6 = SummaryWriter('run/valid_lossr')
        writer7 = SummaryWriter('run/vseq_acc')

        valid_loss = 0
        valid_n_iter = 0
        valid_seq_correct = 0

        valid_n_samples = 0
        valid_loss_history = []
        valid_accuracy_history = []
        valid_best_accuracy = 0

        # Set model to evaluate mode

        model.eval()

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

            # Forward
            outputs = model(inputs)

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
                curated_seq = predicted_digits[k]
                curated_seq[predicted_num_digits[k]:] = -1
                if curated_seq.eq(target_digits[k]).sum().item() == 5:
                    valid_seq_correct += 1

            valid_n_samples += target_ndigits.size(0)

            # adding log
            writer6.add_scalar('Valid Loss', valid_loss / valid_n_iter, i)
            writer7.add_scalar('Valid seq_acc', valid_seq_correct / valid_n_samples, i)

        valid_loss_history.append(valid_loss / valid_n_iter)
        valid_accuracy = valid_seq_correct / valid_n_samples

        best_model = None

        if valid_accuracy > valid_best_accuracy:
            valid_best_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)
        valid_accuracy_history.append(valid_accuracy)

        return valid_loss_history, valid_accuracy, valid_accuracy_history, best_model

