'''
Collection of implementations for model training
'''

from .trainer import StandardSGDTrainer
from .trainer import SemiSVSDTrainer
from .utils import LossTracker
from .loss import weightedBCELoss
from .loss import SemiSVSDLoss
from .loss import ConLoss

__all__ = ['StandardSGDTrainer', 
           'SemiSVSDTrainer',
           'LossTracker',
           'weightedBCELoss',
           'SemiSVSDLoss',
           'ConLoss']
