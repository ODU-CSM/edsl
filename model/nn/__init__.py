"""
This module provides classes for the implementation of built-in networks.

Created on Apr 28, 2021

@author: Javon
"""

from .non_strand_specific_module import NonStrandSpecific
from .danQ import DanQ
from .deeper_deepsea import DeeperDeepSEA
from .heartenn import HeartENN
from .deepsea import DeepSEA
from .multinet_wrapper import MultiNetWrapper
from .edsl import EDSL
from .sei import Sei


__all__ = ['NonStrandSpecific', 
           "danQ", 
           "deeper_deepsea", 
           "deepsea", 
           "heartenn",
           'MultiNetWrapper']

