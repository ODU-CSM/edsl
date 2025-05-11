import torch
import torch.nn as nn
import torch.nn.functional as F
from ...train import weightedBCELoss


class EDSL(nn.Module):
    def __init__(self, sequence_length, n_targets, mode = 'pred'):
        super(EDSL, self).__init__()
        # Define the motif convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=9, padding=0, stride=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=640, kernel_size=17, padding=4, stride=1)
        self.bn2 = nn.BatchNorm1d(640)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=25, padding=8, stride=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn = nn.BatchNorm1d(832)
        self.maxpool = nn.MaxPool1d(kernel_size=9, stride=9)

        # Define the other convolutional and max pool layers
        self.layer1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=7, stride=7),
            nn.Conv1d(in_channels=832, out_channels=832, kernel_size=7),
            nn.LayerNorm([832, 135]),
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=7, stride=7)

        self.layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(in_channels=832, out_channels=832, kernel_size=5),
            nn.LayerNorm([832, 23]),
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=5, stride=5)

        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=832, out_channels=832, kernel_size=3),
            nn.LayerNorm([832, 5]),
        )
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Define batch normalizer and fully connected layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(3328),
            nn.Dropout(p=0.1),
            nn.Linear(3328, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, n_targets),
            nn.GELU(),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())


    def forward(self, x):
        # Apply motif convolutional layers
        x1 = torch.exp(self.bn1(self.conv1(x)))
        x2 = torch.exp(self.bn2(self.conv2(x)))
        x3 = torch.exp(self.bn3(self.conv3(x)))

        # Concatenate the output from motif convolutional layers
        x = torch.cat((x1, x2, x3), dim=1)

        # Apply other convolutional and max pool layers
        x = self.bn(x)
        x1 = F.gelu(self.layer1(x))
        x2 = F.gelu(self.layer2(x1))
        x3 = F.gelu(self.layer3(x2))

        # Concatenate the output from other convolutional layers
        x = torch.cat((
            torch.mean(self.maxpool(x), dim=2), torch.mean(self.maxpool1(x1), dim=2),
            torch.mean(self.maxpool2(x2), dim=2), torch.mean(self.maxpool3(x3), dim=2)), dim=1)

        # Pass through classifier
        x = self.classifier(x)

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