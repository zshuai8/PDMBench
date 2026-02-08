import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch import nn

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model, path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, path):
        if self.verbose:
            print(f'Validation ACC increase ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_acc_min = val_acc


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def cal_f1(y_pred, y_true):
    return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted')


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.bin_lowers[0] = self.bin_lowers[0] - 1e-6

    def forward(self, logits_input, labels_input):
        # move to cpu
        logits = logits_input.cpu()
        labels = labels_input.cpu()

        if len(logits.shape) >1 and logits.shape[1]>1:
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)
        else:
            confidences = torch.sigmoid(logits)
            accuracies = labels
        
        
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def calculate_ece(logits_input,labels_input):
    logits = logits_input.cpu()
    labels = labels_input.cpu()
    ece = _ECELoss()
    ece_value = ece(logits,labels).item()
    return ece_value


def calculate_nll(logits_input,labels_input):
    logits = logits_input.cpu()
    labels = labels_input.cpu()
    if len(logits.shape) >1 and logits.shape[1]>1:
        nll = nn.CrossEntropyLoss()
        nll_value = nll(logits.float(),labels.long()).item()
    else:
        nll = nn.BCEWithLogitsLoss()
        # reshape logits and labels
        logits = logits.reshape(-1)
        labels = labels.float().reshape(-1)

        # avoid nan
        logits = logits.clamp(min=-1e6,max=1e6)

        nll_value = nll(logits.float(),labels.float()).item()
    return nll_value


def calculate_Brier(logits_input, labels_input):
    """
    Calculate the Brier score of a set of logits and labels
    logits (Tensor): a tensor of shape (N, C), where C is the number of classes
    labels (Tensor): a tensor of shape (N,), where each element is in [0, C-1]
    """
    logits = logits_input.cpu()
    labels = labels_input.cpu()
    if len(logits.shape) >1 and logits.shape[1]>1:
        # Convert labels to one-hot
        labels_one_hot = torch.zeros_like(logits)
        labels_one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        softmaxes = F.softmax(logits, dim=1)
        
        # Calculate Brier score
        brier_score = ((softmaxes - labels_one_hot)**2).mean()
        return brier_score.item()
    else:
        labels = labels.float()
        brier_score = ((torch.sigmoid(logits) - labels)**2).mean()
        return brier_score.item()


def calculate_nll_ece(logits,labels):
    ece = calculate_ece(logits,labels)
    nll = calculate_nll(logits,labels)
    return nll, ece


def evaluate_calibration(y_logits, labels):
    nll, ece = calculate_nll_ece(y_logits, labels)
    brier = calculate_Brier(y_logits, labels)
    return nll, ece, brier