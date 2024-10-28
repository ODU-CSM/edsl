'''
Collection of models and wrappers
'''

from .nn import NonStrandSpecific
from .nn import DanQ
from .nn import DeeperDeepSEA
from .nn import HeartENN
from .nn import DeepSEA
from .nn import MultiNetWrapper
from .nn import EDSL
from .nn import Sei

from .wrappers import UniSeqMWrapper
from .wrappers import SemiSVSDMWrapper

from .utils import loadModel
from .utils import loadModelFromFile

__all__ = ['NonStrandSpecific', 
           "danQ", 
           "deeper_deepsea", 
           "deepsea", 
           "heartenn",
           "deepcpg"
           'UniSeqMWrapper',
           'SemiSVSDMWrapper', 
           'MultiNetWrapper', 
           'loadModel',
           'loadModelFromFile']

import importlib
def loadNnModule(className):
    '''
    Load network module by class name
    '''
    if className == 'DanQ':
        return importlib.import_module('fugep.model.nn.danQ')
    elif className == 'DeeperDeepSEA':
        return importlib.import_module('fugep.model.nn.deeper_deepsea')
    elif className == 'DeepSEA':
        return importlib.import_module('fugep.model.nn.deepsea')
    elif className == 'HeartENN':
        return importlib.import_module('fugep.model.nn.heatenn')
    elif className == 'DeepCpGDNA':
        return importlib.import_module('fugep.model.nn.deepcpg')
    elif className == 'DeepSVSD':
        return importlib.import_module('fugep.model.nn.deep_svsd')
    elif className == 'DeepSVSDORG':
        return importlib.import_module('fugep.model.nn.deep_svsd_org')
    elif className == 'DeepDSVSD':
        return importlib.import_module('fugep.model.nn.deep_dsvsd')
    elif className == 'ModifiedDeeperDeepSEA':
        return importlib.import_module('fugep.model.nn.mdds')
    elif className == 'EDSL':
        return importlib.import_module('fugep.model.nn.edsl')
    elif className == 'Sei':
        return importlib.import_module('fugep.model.nn.sei')
    else:
        raise ValueError("Unrecognized network class {0}".format(className))
    
    
def loadWrapperModule(className):
    '''
    Load model wrapper module by class name
    '''
    if className == 'UniSeqMWrapper':
        return importlib.import_module('fugep.model.wrappers.uni_seq')
    elif className == 'SemiSVSDMWrapper':
        return importlib.import_module('fugep.model.wrappers.semi_svsd')
    else:
        raise ValueError("Unrecognized model wrapper class {0}".format(className))
