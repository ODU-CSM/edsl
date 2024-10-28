'''
Implementation of semi-SVSD loss

Created on Aug 12, 2021

@author: Javon
'''

import torch
import torch.nn as nn
from torch import Tensor

from .multi_class import weightedBCELoss

class SemiSVSDLoss(nn.Module):
    
    REPR_LOSSES = ['soft-boundary', 'minimal-distance']
    
    '''
    Possible operating modes, determining the loss that is calculated 
    when forward function is called.
    '''
    MODES = ['multi-class', 'one-class']
    
    def __init__(self, gamma = 0.5, center = None, 
                 radius = None, reprLoss = 'soft-boundary'):
        super(SemiSVSDLoss, self).__init__()
        if gamma <= 0 or gamma > 1:
            raise ValueError('The input gamma is {0}, but needs to be a value in '
                             '(0, 1]'.format(gamma))
        self._gamma = gamma
        self._gammaInverse = 1 / gamma if gamma is not None else None
        
        self._center = center
        self._radius = radius
        self._radiusSquare = radius ** 2 if radius is not None else None
        
        if reprLoss not in self.REPR_LOSSES:
            raise ValueError('The input reprLoss is {0}, but needs to be one '
                             'of [{1}]'.format(reprLoss, self.REPR_LOSSES.join(', ')))
        self._reprLoss = reprLoss
    
    def getGamma(self):
        return self._gamma
    
    def setGamma(self, gamma):
        if gamma <= 0 or gamma > 1:
            raise ValueError('The input gamma is {0}, but needs to be a value in '
                             '(0, 1]'.format(gamma))
        self._gamma = gamma
        self._gammaInverse = 1 / gamma
        
    def getCenter(self):
        return self._center
    
    def setCenter(self, center):
        self._center = center
    
    def getRadius(self):
        return self._radius
    
    def setRadius(self, radius):
        self._radius = radius
        self._radiusSquare = radius ** 2
    
    def getReprLoss(self):
        return self._reprLoss
    
    def setReprLoss(self, reprLoss):
        if reprLoss not in self.REPR_LOSSES:
            raise ValueError('The input reprLoss is {0}, but needs to be one '
                             'of [{1}]'.format(reprLoss, self.REPR_LOSSES.join(', ')))
        self._reprLoss = reprLoss
    
    def forward(self, prediction : Tensor = None, repr : Tensor = None,
                target : Tensor = None, weight : Tensor = None,
                mode = 'multi-class'):
        '''
        Calculate the loss based on input parameters
        
        Parameters
        ----------
        mode : str, optional
            Default is 'one-class'. Can be either 'multi-class' or 'one-class'.
            When multi-class, input parameters pred, weight, and target are 
            mandatory, and weighted BCE loss is computed. If one-class, input
            parameter repr is mandatory and representation loss is calculated.
            
        Return
        -------
        Tensor : loss to optimize, i.e., average loss over the batch
        Tensor : sum of the loss of the batch. In the case of one-class, this is 
            the sum of the loss associated with the squared distance part
        Tensor : the input weights is not currently considered. So, simply the
            the number of examples in the batch
        '''
        if mode not in self.MODES:
            raise ValueError('The mode needs to be one of [{0}], but the input is'
                             '{1}'.format(self.MODES.join(', '), mode))
        
        if mode == 'multi-class':
            if prediction is None or target is None:
                raise ValueError('To compute multi-class loss, inputs for parameters: '
                                 'pred and target are required.')
            return weightedBCELoss(prediction = prediction, target = target, weight = weight)
        
        if mode == 'one-class':
            if repr is None:
                raise ValueError('To compute one-class loss, input for parameter: '
                                 'repr is required')
                
            distSquare = torch.sum((repr - self._center) ** 2, dim = 1)
            if self._reprLoss == 'minimal-distance':
                lossToOptim = torch.mean(distSquare)
                sumOfDistLoss = torch.sum(distSquare)
            else:
                loss = distSquare - self._radiusSquare
                loss = torch.maximum(torch.zeros_like(loss), loss)
                sumOfDistLoss = torch.sum(loss)
                loss = torch.mean(loss)
                loss = loss * self._gammaInverse
                lossToOptim = loss + self._radiusSquare
                
            return lossToOptim, sumOfDistLoss.item(), repr.shape[0]

    def calcLossFromSum(self, sumOfDistLoss, nExamps):
        '''
        Calculate the overall one-class loss from the given sum of loss associated with 
        distance, e.g., returned by forward function 
        
        Parameters
        ----------
        nExamps : the number of examples included to obtain the given sumOfDistLoss
        '''
        
        loss = sumOfDistLoss / nExamps
        if self._reprLoss == 'minimal-distance':
            return loss
        
        if self._reprLoss == 'soft-boundary':
            loss *= self._gammaInverse 
            loss += self._radiusSquare
            return loss
        
        
        
        
        
        
