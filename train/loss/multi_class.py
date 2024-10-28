'''
Implementation  of multi-class loss

Created on Aug 12, 2021

@author: javon
'''

import torch
# from torch.tensor import Tensor # no module torch.tensor in latest version (7/28/2021)
from torch import Tensor
import torch.nn as nn


def weightedBCELoss(prediction: Tensor = None, target: Tensor = None, weight: Tensor = None):
    '''
    weighted binary cross entropy loss. Reduction is mean.
    Samples with 0 weight are ignored during the reduction 
    
    Return
    -------
    Tensor : average loss over the batch
    Tensor : sum of the loss of the batch
    Tensor : number of effective terms in the loss, i.e., number of non-zero weights
    '''
    loss = nn.functional.binary_cross_entropy(prediction, target, 
                      weight = weight, reduction = 'none')
    sumOfLoss = torch.sum(loss)
    if weight is not None:
        nEffTerms = torch.count_nonzero(weight)
        nEffTerms = nEffTerms.item()
    else:
        nEffTerms = torch.tensor(torch.numel(target))
    if nEffTerms == 0:
        aveOfLoss = torch.tensor(0)
    else:
        aveOfLoss = torch.div(sumOfLoss, nEffTerms)
    return aveOfLoss, sumOfLoss.item(), nEffTerms
    

    
