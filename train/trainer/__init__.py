'''
Collection of implementations of model trainers
'''

from .std_sgd import StandardSGDTrainer
from .deep_svsd import SemiSVSDTrainer

__all__ = ['StandardSGDTrainer',
           'SemiSVSDTrainer']
