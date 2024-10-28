'''
Implementation of data collators used in the dataloader

Created on Aug 10, 2021

@author: javon
'''

import numpy as np
import torch

from ..utils import getSWeight

class IntervalCollator(object):
    '''collate retrieved data to create a mini-batch
    '''
    def __init__(self, seqLen, featsInH5, cWeights = None, iFeatsPred = None, 
                 unpackbits = False, nSeqBase = 4, valueOfMissing = None,
                 needExampleId = False, needTargets = True, needWeights = True):
        '''
        Constructor of a collator
        
        Parameters
        -----------
        needExampleId : bool, optional
            Default is False. Indicate whether to include the ID of the example in a batch
            If true, it is expected that such ID is included in the input of __call__ 
            function
        '''
        self._seqLen = seqLen
        self._nSeqBase = nSeqBase
        self._featsInH5 = featsInH5
        self._cWeights = cWeights
        self._iFeatsPred = iFeatsPred
        self._unpackbits = unpackbits
        self._valueOfMissing = valueOfMissing
        self._needExampleId = needExampleId
        self._needWeights = needWeights
        self._needTargets = needTargets
    
    def __call__(self, batch):
        if self._unpackbits:
            sequences = np.zeros((len(batch), self._seqLen, self._nSeqBase),
                             dtype = np.float32)
            if self._needTargets:
                targets = np.zeros((len(batch), len(self._featsInH5)))
            if self._needExampleId:
                ids = []
            for iSamp in range(len(batch)):
                samp = batch[iSamp]
                # retrieve sequence
                sequence = np.unpackbits(samp['sequence'], axis=-2)
                nulls = np.sum(sequence, axis=-1) == sequence.shape[-1]
                sequence = sequence.astype(float)
                sequence[nulls, :] = 1.0 / sequence.shape[-1]
                sequences[iSamp] = sequence[:self._seqLen, :]
                
                if self._needTargets:
                    # retrieve targets
                    target = np.unpackbits(samp['targets'], axis = -1).astype(float)
                    targets[iSamp] = target[:len(self._featsInH5)]
                
                if self._needExampleId:
                    ids.append(samp['id'])
        else:
            sequences = np.zeros((len(batch), batch[0]['sequence'].shape[1], self._nSeqBase),
                             dtype = np.float32)
            if self._needTargets:
                targets = np.zeros((batch[0]['targets'].shape[0], len(self._featsInH5)))
            if self._needExampleId:
                ids = []
            for iSamp in range(len(batch)):
                samp = batch[iSamp]
                sequences[iSamp] = samp['sequence'] 
                if self._needTargets:
                    targets[iSamp] = samp['targets']
                if self._needExampleId:
                    ids.append(samp['id'])
        
        # keep only target data of features to predict
        if self._needTargets and self._iFeatsPred is not None:
            targets = targets[:, self._iFeatsPred]
        
        if self._needWeights:
            # compute the sample weights
            if self._iFeatsPred is not None:
                features = [self._featsInH5[i] for i in self._iFeatsPred]
            else:
                features = self._featsInH5
            weights = getSWeight(targets, features = features, 
                     cWeights = self._cWeights, valueOfMissing = self._valueOfMissing)
        
        colBatch = (torch.from_numpy(sequences),)
        if self._needTargets:
            colBatch = colBatch + (torch.from_numpy(targets),)
        if self._needWeights:
            colBatch = colBatch + (torch.from_numpy(weights),)
        if self._needExampleId:
            colBatch = colBatch + (ids,)
            
        return colBatch
