'''
Created on May 23, 2021

@author: jsun
'''

class LossTracker:
    '''
    Track loss when batches used in training and validation
    '''
    def __init__(self):
        self._loss = 0  # accumulated loss
        self._nItems = 0 # number of samples the produced that loss in _loss
    
    def add(self, loss, nItems):
        ''' 
        add the loss produced by a batch of nItems samples
        '''
        self._loss += loss
        self._nItems += nItems
    
    def getAveLoss(self):
        '''
        Return the average loss
        '''
        if self._nItems == 0:
            return 'NA'
        return self._loss / self._nItems
    
    def getSumOfLoss(self):
        '''
        Get accumulated loss
        '''
        return self._loss
    
    def getNItems(self):
        '''
        Get the number of Items 
        '''
        return self._nItems
    
    def reset(self):
        '''
        Reset the track
        '''
        self._loss = 0
        self._nItems = 0

