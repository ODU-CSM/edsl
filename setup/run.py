"""
Utilities for loading configurations, instantiating Python objects, and
running operations in _Selene_.

"""
import os
import importlib
import sys
from time import strftime
import types
import torch
import random
import numpy as np

from . import instantiate
from ..model import loadNnModule
from ..model import loadWrapperModule


def class_instantiate(classobj):
    """Not used currently, but might be useful later for recursive
    class instantiation
    """
    for attr, obj in classobj.__dict__.items():
        is_module = getattr(obj, '__module__', None)
        if is_module and "fugep" in is_module and attr is not "model":
            class_instantiate(obj)
    classobj.__init__(**classobj.__dict__)


def module_from_file(path):
    """
    Load a module created based on a Python file path.

    Parameters
    ----------
    path : str
        Path to the model architecture file.

    Returns
    -------
    The loaded module

    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(
        module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """
    This method expects that you pass in the path to a valid Python module,
    where the `__init__.py` file already imports the model class,
    `criterion`, and `get_optimizer` methods from the appropriate file
    (e.g. `__init__.py` contains the line `from <model_class_file> import
    <ModelClass>`).

    Parameters
    ----------
    path : str
        Path to the Python module containing the model class.

    Returns
    -------
    The loaded module
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)

def _getModelInfo(configs, sampler):
    '''
    Assemble model info from the config dictionary
    '''
    modelInfo = configs["model"]
    if not ('classArgs' in modelInfo):
        modelInfo['classArgs'] = dict()
    classArgs = modelInfo['classArgs']
    if not ('sequence_length' in classArgs):
        classArgs['sequence_length'] = sampler.getSequenceLength()
    if not ('n_targets' in classArgs):
        classArgs['n_targets'] = len(sampler.getFeatures())
    
    return modelInfo

def initialize_model(model_configs, train=True, lr=None):
    """
    Initialize model (and associated criterion, optimizer)

    Parameters
    ----------
    model_configs : dict
        Model-specific configuration
    train : bool, optional
        Default is True. If `train`, user-specified optimizer
        and optimizer class that can be found within the input model file 
        will be set to the model wrapper.
    lr : float or None, optional
        If `train`, a learning rate must be specified. Otherwise, None.

    Returns
    -------
    model, criterion : tuple(torch.nn.Module, torch.nn._Loss) or \
            model, criterion, optim_class, optim_kwargs : \
                tuple(torch.nn.Module, torch.nn._Loss, torch.optim, dict)

        * `torch.nn.Module` - the model architecture
        * `torch.nn._Loss` - the loss function associated with the model
        * `torch.optim` - the optimizer associated with the model
        * `dict` - the optimizer arguments

        The optimizer and its arguments are only returned if `train` is
        True.

    Raises
    ------
    ValueError
        If `train` but the `lr` specified is not a float.

    """
    model_class_name = model_configs["class"]
    
    module = None
    if 'path' in model_configs.keys():
        # load network module from user given file
        import_model_from = model_configs["path"]
        if os.path.isdir(import_model_from):
            module = module_from_dir(import_model_from)
        else:
            module = module_from_file(import_model_from)
    else:
        # load built in network module
        module = loadNnModule(model_class_name)
    
    model_class = getattr(module, model_class_name)

    model = model_class(**model_configs["classArgs"])
    if "non_strand_specific" in model_configs:
        from fugep.model import NonStrandSpecific
        model = NonStrandSpecific(
            model, mode=model_configs["non_strand_specific"])

    # loss function 
    if 'criterionArgs' in model_configs:
        criterionArgs = model_configs['criterionArgs']
    else:
        criterionArgs = dict()
    criterion = module.criterion(**criterionArgs)
    
    # optimizer for training
    optim_class, optim_kwargs = None, None
    mode = 'evaluate'
    if train and isinstance(lr, float):
        mode = 'train'
        optim_class, optim_kwargs = module.get_optimizer(lr)
    elif train:
        raise ValueError("Learning rate must be specified as a float "
                         "but was {0}".format(lr))
    
    # construct model wrapper    
    if 'wrapperArgs' in model_configs:
        wrapperArgs = model_configs['wrapperArgs']
    else:
        wrapperArgs = dict()
    modelWrapper = initializeWrapper(model_configs['wrapper'], 
         mode = mode, model = model, loss = criterion, wrapperArgs = wrapperArgs,
         optimizerClass = optim_class,  optimizerKwargs = optim_kwargs)
    return modelWrapper

def initializeWrapper(className, mode, model, loss, wrapperArgs, useCuda = False, 
          optimizerClass = None, optimizerKwargs = None, ):
    '''
    Initialize model wrapper
    '''
    wrapperClass = getattr(loadWrapperModule(className), className)
    wrapper = wrapperClass(model, mode = mode, lossCalculator = loss, 
             useCuda = useCuda, optimizerClass = optimizerClass, 
             optimizerKwargs = optimizerKwargs, **wrapperArgs)
    return wrapper
    
def execute(operations, configs, output_dir):
    """
    Execute operations in _Selene_.

    Parameters
    ----------
    operations : list(str)
        The list of operations to carry out in _Selene_.
    configs : dict or object
        The loaded configurations from a YAML file.
    output_dir : str or None
        The path to the directory where all outputs will be saved.
        If None, this means that an `output_dir` was not specified
        in the top-level configuration keys. `output_dir` must be
        specified in each class's individual configuration wherever
        it is required.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    Raises
    ------
    ValueError
        If an expected key in configuration is missing.

    """
    model = None
    modelTrainer = None
    for op in operations:
        if op == "train":
            # construct sampler
            sampler_info = configs["sampler"]
            if output_dir is not None:
                sampler_info.bind(output_dir=output_dir)
            sampler = instantiate(sampler_info)
            
            if 'bgSampler' in configs.keys():
                # construct background sampler for semi-supervised one-class learning
                bgSamplerInfo = configs['bgSampler']
                if output_dir is not None:
                    bgSamplerInfo.bind(output_dir = output_dir)
                bgSampler = instantiate(bgSamplerInfo)
            
            # construct model
            modelInfo = _getModelInfo(configs, sampler)
            model = initialize_model(modelInfo, train = True, lr = configs["lr"])
            
            # create trainer
            train_model_info = configs["train_model"]
            train_model_info.bind(model = model, dataSampler = sampler)
            if 'bgSampler' in configs.keys():
                train_model_info.bind(bgDataSampler = bgSampler)
            if sampler.getValOfMisInTarget() is not None:
                train_model_info.bind(valOfMisInTarget = sampler.getValOfMisInTarget())
            
            if output_dir is not None:
                train_model_info.bind(outputDir = output_dir)
            if "random_seed" in configs:
                train_model_info.bind(deterministic=True)

            modelTrainer = instantiate(train_model_info)
            
            # train
            modelTrainer.trainAndValidate()

        elif op == "evaluate":
            if modelTrainer is not None:
                modelTrainer.evaluate()
            else:
                # construct sampler
                sampler_info = configs["sampler"]
                sampler = instantiate(sampler_info)
                
                # construct model
                modelInfo = _getModelInfo(configs, sampler)
                model = initialize_model(modelInfo, train = False)
                
                # construct evaluator  
                evaluate_model_info = configs["evaluate_model"]  
                evaluate_model_info.bind(model = model, dataSampler = sampler)
                if output_dir is not None:
                    evaluate_model_info.bind(outputDir = output_dir)
                if sampler.getValOfMisInTarget() is not None:
                    evaluate_model_info.bind(valOfMisInTarget = sampler.getValOfMisInTarget())
                evaluate_model = instantiate(evaluate_model_info)
                
                # evaluate
                evaluate_model.evaluate()

        elif op == "analyze":
            if not model:
                model = initialize_model(configs["model"], train = False)
            
            # construct analyzer
            analyze_seqs_info = configs["analyzer"]
            analyze_seqs_info.bind(model = model)
            if output_dir is not None:
                    analyze_seqs_info.bind(outputDir = output_dir)
            analyze_seqs = instantiate(analyze_seqs_info)
            
            if "variant_effect_prediction" in configs:
                analyze_seqs.evaluate()
            elif "in_silico_mutagenesis" in configs:
                ism_info = configs["in_silico_mutagenesis"]
                analyze_seqs.evaluate(**ism_info)
            elif "prediction" in configs:
                predict_info = configs["prediction"]
                analyze_seqs.predict(**predict_info)
            else:
                raise ValueError('The type of analysis needs to be specified. It can '
                                 'either be variant_effect_prediction, in_silico_mutagenesis '
                                 'or prediction')

def parse_configs_and_run(configs,
                          create_subdirectory=True,
                          lr=None):
    """
    Method to parse the configuration YAML file and run each operation
    specified.

    Parameters
    ----------
    configs : dict
        The dictionary of nested configuration parameters. Will look
        for the following top-level parameters:

            * `ops`: A list of 1 or more of the values \
            {"train", "evaluate", "analyze"}. The operations specified\
            determine what objects and information we expect to parse\
            in order to run these operations. This is required.
            * `output_dir`: Output directory to use for all the operations.\
            If no `output_dir` is specified, assumes that all constructors\
            that will be initialized (which have their own configurations\
            in `configs`) have their own `output_dir` specified.\
            Optional.
            * `random_seed`: A random seed set for `torch` and `torch.cuda`\
            for reproducibility. Optional.
            * `lr`: The learning rate, if one of the operations in the list is\
            "train".
            * `load_test_set`: If `ops: [train, evaluate]`, you may set\
               this parameter to True if you would like to load the test\
               set into memory ahead of time--and therefore save the test\
               data to a .bed file at the start of training. This is only\
               useful if you have a machine that can support a large increase\
               (on the order of GBs) in memory usage and if you want to\
               create a test dataset early-on because you do not know if your\
               model will finish training and evaluation within the allotted\
               time that your job is run.

    create_subdirectory : bool, optional
        Default is True. If `create_subdirectory`, will create a directory
        within `output_dir` with the name formatted as "%Y-%m-%d-%H-%M-%S",
        the date/time this method was run.
    lr : float or None, optional
        Default is None. If "lr" (learning rate) is already specified as a
        top-level key in `configs`, there is no need to set `lr` to a value
        unless you want to override the value in `configs`. Otherwise,
        set `lr` to the desired learning rate if "train" is one of the
        operations to be executed.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    """
    operations = configs["ops"]

    if "train" in operations and "lr" not in configs and lr != "None":
        configs["lr"] = float(lr)
    elif "train" in operations and "lr" in configs and lr != "None":
        print("Warning: learning rate specified in both the "
              "configuration dict and this method's `lr` parameter. "
              "Using the `lr` value input to `parse_configs_and_run` "
              "({0}, not {1}).".format(lr, configs["lr"]))

    current_run_output_dir = None
    if "output_dir" not in configs and \
            ("train" in operations or "evaluate" in operations):
        print("No top-level output directory specified. All constructors "
              "to be initialized (e.g. Sampler, TrainModel) that require "
              "this parameter must have it specified in their individual "
              "parameter configuration.")
    elif "output_dir" in configs:
        current_run_output_dir = configs["output_dir"]
        os.makedirs(current_run_output_dir, exist_ok=True)
        if "create_subdirectory" in configs:
            create_subdirectory = configs["create_subdirectory"]
        if create_subdirectory:
            current_run_output_dir = os.path.join(
                current_run_output_dir, strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(current_run_output_dir)
        print("Outputs and logs saved to {0}".format(
            current_run_output_dir))

    if "random_seed" in configs:
        seed = configs["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print("Warning: no random seed specified in config file. "
              "Using a random seed ensures results are reproducible.")

    execute(operations, configs, current_run_output_dir)
