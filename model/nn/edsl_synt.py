import torch
import torch.nn as nn
import torch.nn.functional as F
from ...train import weightedBCELoss


class EDSLSynt(nn.Module):
    def __init__(self, sequence_length, n_targets, mode = 'pred'):
        super(EDSLSynt, self).__init__()
        # Define the motif convolutional layers
        self.conv1 = nn.Conv1d(4, 8, kernel_size=9, padding=0, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=17, padding=4, stride=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=25, padding=8, stride=1)
        self.bn3 = nn.BatchNorm1d(8)

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(24),
            #nn.Dropout(p=0.1),
            nn.Linear(24, n_targets),
            nn.GELU(),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())

    def conv_net(self, x):
        # Apply motif convolutional layers
        x1 = torch.exp(self.bn1(self.conv1(x)))
        x2 = torch.exp(self.bn2(self.conv2(x)))
        x3 = torch.exp(self.bn3(self.conv3(x)))

        # Concatenate the output from convolutional layers
        x = torch.cat((x1, x2, x3), dim=1)

        return x

    def forward(self, x):
        # Apply conv_net
        x = self.conv_net(x)

        # Apply classifier
        x = self.classifier(torch.mean(x, dim=2))

        return x


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