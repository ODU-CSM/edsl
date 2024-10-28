import torch
import torch.nn as nn
import torch.nn.functional as F
from ...train import weightedBCELoss
import numpy as np


class DenseSynt(nn.Module):
    def __init__(self, sequence_length, n_targets, mode = 'pred'):
        super(DenseSynt, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 3

        self.conv1 = nn.Conv1d(4, 24, kernel_size=conv_kernel_size)
        self.pool1 = nn.MaxPool1d(
            kernel_size=pool_kernel_size, stride=pool_kernel_size)
        self.bn1 = nn.BatchNorm1d(24)

        self.conv2 = nn.Conv1d(24, 24, kernel_size=conv_kernel_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(
            kernel_size=pool_kernel_size, stride=pool_kernel_size)
        self.bn2 = nn.BatchNorm1d(24)

        self.conv3 = nn.Conv1d(24, 24, kernel_size=conv_kernel_size)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm1d(24)

        self.bn = nn.BatchNorm1d(72)

        self.classifier = nn.Sequential(
            nn.Linear(72, n_targets),
            nn.GELU(),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        x = torch.exp(self.conv1(x))
        x1 = self.bn1(self.pool1(x))
        x2 = self.bn2(self.pool2(self.relu2(self.conv2(x1))))
        x3 = self.bn3(self.relu3(self.conv3(x2)))
        reshape_out = torch.cat((torch.mean(x1, dim=2), torch.mean(x2, dim=2), torch.mean(x3, dim=2)), dim=1)

        predict = self.classifier(self.bn(reshape_out))
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