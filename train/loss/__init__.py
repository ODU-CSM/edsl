'''
Collection of implementations of loss functions for training
'''

from .multi_class import weightedBCELoss
from .semi_svsd import SemiSVSDLoss
from .contrastive import ConLoss
from .discriminative import DiscriminativeLoss

__all__ = ['weightedBCELoss',
           'SemiSVSDLoss',
           'DiscriminativeLoss',
           'ConLoss']