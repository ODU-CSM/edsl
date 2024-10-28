'''
Created on May 23, 2021

@author: jsun
'''

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
from ...train.loss import DiscriminativeLoss

class PredMWrapper(metaclass = ABCMeta):
    '''
    The base class of all predictive models
    '''
    
    BASE_MODES = ("train", "evaluate")
    """
    The types of modes that the `PredMWrapper` object can run in.
    """
    
    def __init__(self, model, mode = 'train', lossCalculator = None,
                 useCuda = False, optimizerClass = None, optimizerKwargs = None):
        '''
        Constructor for a new PredMWrapper object
        
        Parameters
        -----------------------
        mode: 
            mode of the model, can be either 'train' or 'evaluate',
            optional, 'train' by default
            
        lossCalculator: 
            function or class for computing loss
        '''
        self._model = model
        self._dataParallel = False
        
        if mode not in self.BASE_MODES:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.BASE_MODES))
            
        self._mode = mode
        self._lossCalculator = lossCalculator
        self._useCuda = useCuda
        if self._useCuda:
            self._mode.cuda()
            if self._lossCalculator is not None:
                self._lossCalculator.cuda()
        
        if optimizerClass is not None:
            self._optimizer = optimizerClass(
                self._model.parameters(), **optimizerKwargs)    
    
    def getLossCalculator(self):
        '''
        Return loss calculator
        '''
        return self._lossCalculator
    
    def setLossCalculator(self, lossCalculator):
        '''
        Set loss calculator
        '''
        self._lossCalculator = lossCalculator
        if self._useCuda:
            self._lossCalculator.cuda()
    
    def getStateDict(self):
        '''
        Return model state dictionary
        '''
        return self._model.state_dict()
    
    def getOptimizer(self):
        '''
        Return optimizer
        
        Raise error when optimizer has not been set
        '''
        if self._optimizer is None:
            raise ValueError('Optimizer has not been set')
        return self._optimizer
    
    def setOptimizer(self, optimizerClass, optimizerKwargs):
        '''
        Set optimizer
        '''
        self._optimizer = optimizerClass(
            self._model.parameters(), **optimizerKwargs)

    def setOptimizer_con(self, optimizerClass, optimizerKwargs):
        '''
        Set optimizer
        '''
        self._optimizer = optimizerClass([{
            'params': self._model.module.model.conv_net.parameters()},
            {'params': self._model.module.model.fc_net.parameters()}], **optimizerKwargs)

    def setOptimizer_classify(self, optimizerClass, optimizerKwargs):
        '''
        Set optimizer
        '''
        self._optimizer = optimizerClass(
            self._model.module.model.classifier.parameters(), **optimizerKwargs)

    def setOptimizer(self, optimizerClass, optimizerKwargs):
        '''
        Set optimizer
        '''
        self._optimizer = optimizerClass(
            self._model.parameters(), **optimizerKwargs)

    def initOptim(self, stateDict):
        '''
        Init optimizer from state dictionary
        
        Raise error when optimizer has not been set
        '''
        if self._optimizer is None:
            raise ValueError('Optimizer has not been set')
        
        self._optimizer.load_state_dict(stateDict)
        if self._useCuda:
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        
        
    def getOptimStateDict(self):
        '''
        Get the state dictionary of optimizer
        
        Raise error when optimizer has not been set
        '''
        if self._optimizer is None:
            raise ValueError('Optimizer has not been set')
        return self._optimizer.state_dict()
        
    def setMode(self, mode):
        """
        Sets the PredMWrapper mode.

        Parameters
        ----------
        mode : str
            The name of the mode to use. It must be one of
            `PredMWrapper.BASE_MODES`.

        Raises
        ------
        ValueError
            If `mode` is not a valid mode.

        """
        if mode not in self.BASE_MODES:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.BASE_MODES))
        self._mode = mode    
    
    def toDataParallel(self):
        '''
        set data parallelism at the model level
        '''
        self._model = nn.DataParallel(self._model)
        self._dataParallel = True
    
    def getReprDim(self):
        '''
        Get hidden representation, typically the activation of the layer
        right before classification layers, depending how the getReprDim()
        function of the network is implemented.
        '''
        if self._dataParallel:
            return self._model.module.getReprDim()
        else:
            return self._model.getReprDim()
    
    def toUseCuda(self):
        self._useCuda = True
        self._model.cuda()
        if self._lossCalculator is not None and \
            isinstance(self._lossCalculator, nn.Module):
            self._lossCalculator.cuda()
    
    @abstractmethod
    def fit(self, batchData, optimizer = None):
        """
        Fit the model with a batch of data

        Parameters
        ----------
        batchData : dict
            A dictionary that holds the data for training
        optimizer : torch.optim.Optimizer
            The optimizer to use for the fitting

        Returns
        -------
        float : sum
            The sum of the loss over the batch of the data
        int : nTerms
            The number of terms involved in calculated loss. 
            
        Note
        ------
        The current implementation of this function is one step of gradient update.
        Future implementation can include a nStep argument to support multiple
        steps update or even train to the end (until no improvement over 
        the input batch of data)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def validate(self, dataInBatches):
        """
        Validate the model with a batch of data

        Parameters
        ----------
        dataInBatches : []
            A list of dictionaries that hold data in batches for the validating

        Returns
        -------
        float : 
            The average loss over the batch of the data
        nArray :
            The prediction
            Note, this can be None in one-class classification models
        """
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, batchData):
        """
        Apply the model to make prediction for a batch of data

        Parameters
        ----------
        batchData : dict
            A dictionary that holds the input data for prediction

        Returns
        -------
        nArray :
            The prediction
        """
        raise NotImplementedError()
    
    @abstractmethod
    def init(self, stateDict = None):
        """
        Initialize the model before training or making prediction
        """
        raise NotImplementedError()
    
    @abstractmethod
    def initFromFile(self, filepath):
        '''
        Initialize the model by a previously trained model saved 
        to a file
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, outputDir, modelName = 'model'):
        """
        Save the model
        
        Parameters:
        --------------
        outputDir : str
            The path to the directory where to save the model
        """
        raise NotImplementedError()
    
    
    
    
    