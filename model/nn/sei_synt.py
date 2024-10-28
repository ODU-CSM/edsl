import torch
import torch.nn as nn
import torch.nn.functional as F
from ...train import weightedBCELoss
import numpy as np


class SeiSynt(nn.Module):
    def __init__(self, sequence_length, n_targets, mode = 'pred'):
        super(SeiSynt, self).__init__()
        conv_kernel_size = 9

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 24, kernel_size=9),
            nn.Conv1d(24, 24, kernel_size=9))

        self.conv1 = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=9, padding=4))
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)

        self.classifier = nn.Sequential(
            nn.Linear(24, n_targets),
            nn.GELU(),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        x = self.lconv1(x)
        out = x + torch.exp(self.conv1(x))
        reshape_out = torch.mean(self.maxpool(out), dim=2)
        predict = self.classifier(reshape_out)
        return predict


def criterion():
    """
    Specify the appropriate loss function (criterion) for this
    model.

    Returns
    -------
    torch.nn._Loss
    """
    return weightedBCELoss

def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.

    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.

    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})