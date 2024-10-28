'''
Setup and run fugep
'''

from .config import load
from .config import load_path
from .config import instantiate
from .run import initialize_model
from .run import execute
from .run import parse_configs_and_run

__all__ = ["load",
           "load_path",
           "instantiate",
           "initialize_model",
           "execute",
           "parse_configs_and_run"]