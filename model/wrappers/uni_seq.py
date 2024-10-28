'''
Created on May 23, 2021

@author: jsun
'''

import torch
import numpy as np
import os
import pandas as pd
from .pred import PredMWrapper
from ...train import LossTracker
from ..utils import loadModel
from ..utils import loadModelFromFile
import torch.nn as nn


class UniSeqMWrapper(PredMWrapper):
    '''
    classdocs
    '''
    def __init__(self, model, mode = 'train', lossCalculator = None, 
             useCuda = False, optimizerClass = None, optimizerKwargs = None):
        '''
        Constructor
        '''
        super(UniSeqMWrapper, self).__init__(model, 
            mode = mode, lossCalculator = lossCalculator,
            useCuda = useCuda, optimizerClass = optimizerClass,
            optimizerKwargs = optimizerKwargs)

        self.firstTime = True
        
    def fit(self, batchData):
        """
        Fit the model with a batch of data

        Parameters
        ----------
        batchData : dict
            A dictionary that holds the data for training

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
        self._model.train()
        inputs = torch.Tensor(batchData['sequence'])
        targets = torch.Tensor(batchData['targets'])
        weights = None
        if 'weights' in batchData:
            weights = torch.Tensor(batchData['weights'])
        if self._useCuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            if torch.is_tensor(weights):
                weights = weights.cuda() # Added by Sanjeev 07/20/2021

        predictions = self._model(inputs.transpose(1, 2))
        aveLoss, sumOfLoss, nEffTerms = \
            self._lossCalculator(prediction = predictions, target = targets, weight = weights)
        print(aveLoss)
        self._optimizer.zero_grad()
        aveLoss.backward()
        #torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10)
        self._optimizer.step()
        
        return (sumOfLoss, nEffTerms)
    
    
    def validate(self, dataInBatches, evalType='validate', sampler=None, layer=0):
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
        """
        self._model.eval()

        batchLosses = LossTracker()
        allPreds = []
        features = []
        #features = np.zeros((947008, 64, 992))

        self.testDistNeg = np.array([])
        self.testDistPos = np.array([])
        self.testReprNeg = np.array([])
        self.testReprPos = np.array([])
        if evalType == 'test':
            #trainData, _ = sampler.getDataAndTargets(64, mode='train')
            #validationData, _ = sampler.getValidationSet(64)
            #allBatches = trainData + validationData
            #testData, _ = sampler.getDataAndTargets(64, mode='test')
            #center = self.initCenter(allBatches, eps=0.1)
            cent = pd.read_csv('/scratch/ml-csm/projects/fgenom/ocl/train-data/rmdnase/hg38/training/data/single/outputs/0/center.txt', header=None)
            center = torch.reshape(torch.tensor(cent.values), (self.getReprDim(), )).to('cuda:0')
        for batchData in dataInBatches:
            inputs = torch.Tensor(batchData['sequence'])
            targets = torch.Tensor(batchData['targets'])
            weights = None
            if 'weights' in batchData:
                weights = torch.Tensor(batchData['weights'])
            if self._useCuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                if torch.is_tensor(weights):
                    weights = weights.cuda() # Added by Sanjeev 07/20/2021

            with torch.no_grad():
                if layer:
                    feature = self._model.module.model.lconv1(inputs.transpose(1, 2))
                    feature = torch.relu(self._model.module.model.conv1(feature))
                    features.append(feature.cpu().numpy())
                else:
                    if evalType == 'test':
                        repr = self._model(inputs.transpose(1, 2), mode='repr')
                        reprDist = torch.sum(repr, dim=1).data.cpu().numpy().reshape((repr.shape[0], 1))
                        self.testReprNeg = np.append(reprDist[np.where(batchData['targets'] == 0)],
                                                     self.testReprNeg)
                        self.testReprPos = np.append(reprDist[np.where(batchData['targets'] == 1)],
                                                     self.testReprPos)

                        # compute distance to center
                        distSqr = torch.sum((repr - center) ** 2, dim=1)
                        distSqr = distSqr.data.cpu().numpy()
                        distSqr = distSqr.reshape((distSqr.shape[0], 1))
                        self.testDistNeg = np.append(np.sqrt(distSqr[np.where(
                            batchData['targets'] == 0)]), self.testDistNeg)
                        self.testDistPos = np.append(np.sqrt(distSqr[np.where(
                            batchData['targets'] == 1)]), self.testDistPos)

                    predictions = self._model(inputs.transpose(1, 2))
                    _, sumOfLoss, nEffTerms =\
                        self._lossCalculator(prediction = predictions, target = targets, weight = weights)
                    allPreds.append(
                        predictions.data.cpu().numpy())
                    batchLosses.add(sumOfLoss, nEffTerms)

        if layer:
            features = np.vstack(features)
            return features
            #return batchLosses.getAveLoss(), allPreds, features
        allPreds = np.vstack(allPreds)
        return batchLosses.getAveLoss(), allPreds


    def predict(self, dataInBatches):
        """
        Apply the model to make prediction for a batch of data

        Parameters
        ----------
        batchData : []
            A list of dictionaries that hold data in batches for the validating

        Returns
        -------
        nArray :
            The prediction
        """
        self._model.eval()
        
        allPreds = []
        for batchData in dataInBatches:
            inputs = torch.Tensor(batchData['sequence'])

            if self._useCuda:
                inputs = inputs.cuda()
            with torch.no_grad():
                predictions = self._model(inputs.transpose(1, 2))
                allPreds.append(predictions.data.cpu().numpy())
        allPreds = np.vstack(allPreds)
        
        return allPreds

    
    def init(self, stateDict = None):
        """
        Initialize the model before training or making prediction
        """
        if stateDict is not None:
            self._model = loadModel(stateDict, self._model)
    
    def initFromFile(self, filepath):
        '''
        Initialize the model by a previously trained model saved 
        to a file
        '''
        loadModelFromFile(filepath, self._model)
    
    def save(self, outputDir, modelName = 'model'):
        """
        Save the model
        
        Parameters:
        --------------
        outputDir : str
            The path to the directory where to save the model
        """
        outputPath = os.path.join(outputDir, modelName)
        torch.save(self._model.state_dict(), 
                   "{0}.pth.tar".format(outputPath))

    def initCenter(self, dataBatches, eps=0.1, saveCenterTo=None):
        '''
        Computer the center of the hidden representation of input data

        Parameters
        ----------
        dataBatches : A list of mini-batch of data
        saveCenterTo : Str, optional
            Default is None, center will not be saved
            The path to the file to save center
        '''

        self._model.train()  # use train mode, important!!!

        nSamps = 0
        center = torch.zeros(self.getReprDim())
        if self._useCuda:
            center = center.cuda()

        with torch.no_grad():
            for batchData in dataBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
                repr = self._model(inputs.transpose(1, 2), mode='repr')
                nSamps += repr.shape[0]
                center += torch.sum(repr, dim=0)

            center /= nSamps

        # If center_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps

        if self._useCuda:
            centToSave = center.cpu()
        centSer = pd.Series(centToSave, copy = False)
        centSer.to_csv('/scratch/ml-csm/projects/fgenom/ocl/train-data/rmdnase/hg38/training/data/single/outputs/0/center.txt', header = False, index = False)

        return center
