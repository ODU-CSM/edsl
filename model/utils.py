'''
Utils related to models

Created on May 24, 2021

@author: jsun
'''

from collections import OrderedDict
import torch

from .nn import MultiNetWrapper

def loadModel(stateDict, model):
    """
    Loads model weights that were saved to a file previously by `torch.save`.
    This is a helper function to reconcile state dict keys where a model was
    saved with/without torch.nn.DataParallel and now must be loaded
    without/with torch.nn.DataParallel.

    Parameters
    ----------
    stateDict : collections.OrderedDict
        The state of the model.
    model : torch.nn.Module
        The PyTorch model, a module composed of submodules.

    Returns
    -------
    torch.nn.Module \
        The model with weights loaded from the state dict.

    Raises
    ------
    ValueError
        If model state dict keys do not match the keys in `stateDict`.
    
    Note
    ---------
    Selene's load_model_from_state_dict function
    
    """
    if 'state_dict' in stateDict:
        stateDict = stateDict['state_dict']

    model_keys = model.state_dict().keys()
    state_dict_keys = stateDict.keys()

    if len(model_keys) != len(state_dict_keys):
        try:
            model.load_state_dict(stateDict, strict=False)
            return model
        except Exception as e:
            raise ValueError("Loaded state dict does not match the model "
                "architecture specified - please check that you are "
                "using the correct architecture file and parameters.\n\n"
                "{0}".format(e))

    new_state_dict = OrderedDict()
    for (k1, k2) in zip(model_keys, state_dict_keys):
        value = stateDict[k2]
        try:
            new_state_dict[k1] = value
        except Exception as e:
            raise ValueError(
                "Failed to load weight from module {0} in model weights "
                "into model architecture module {1}. (If module name has "
                "an additional prefix `model.` it is because the model is "
                "wrapped in `fugep.utils.NonStrandSpecific`. This "
                "error was raised because the underlying module does "
                "not match that expected by the loaded model:\n"
                "{2}".format(k2, k1, e))
    model.load_state_dict(new_state_dict)
    return model

def loadModelFromFile(filepath, model):
    '''
    Load previously trained model(s) saved to file(s)
    '''
    
    if isinstance(filepath, str):
        trained_model = torch.load(filepath,
            map_location=lambda storage, location: storage)

        loadModel(trained_model, model)
    elif hasattr(filepath, '__len__'):
        state_dicts = []
        for mp in filepath:
            state_dict = torch.load(
                mp, map_location=lambda storage, location: storage)
            state_dicts.append(state_dict)

        for (sd, sub_model) in zip(state_dicts, model.sub_models):
            loadModel(sd, sub_model)
    else:
        raise ValueError(
            '`filepath` should be a str or list of strs '
            'specifying the full paths to model weights files, but was '
            'type {0}.'.format(type(filepath)))
