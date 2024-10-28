"""
This is the main module for FuGEP.
"""

from .version import __version__
from .evaluate import ModelEvaluator
from .cli import main

__all__ = ["data", "model", "samplers", "utils",
           'train', "predict", "interpret", "__version__",
           'ModelEvaluator', 'main']