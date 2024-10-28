'''
Created on Apr 30, 2021

@author: jsun
'''

from torch.utils.data.dataset import IterableDataset
from torch.utils.data import get_worker_info
import math
import numpy as np

from .reader import H5Reader

class H5Dataset(IterableDataset):
    '''
    Iterable dataset for loading H5 files
    '''
    
    def __init__(self, h5Files, nameOfData, seed = None, shuffle = True,
                 exampleId = False, nDigitInFile = None):
        '''
        Parameters
        ----------
        h5Files: the list of paths to h5 files 
        exampleId : bool, optional
            Default is False
            Whether to include id of example in the data batch
        nDigitInFile : int, optional
            Default is None. When exampleId is True, this is mandatory. 
            Specify the number of lower digits reserved for indexing examples within a file
            while creating unique IDs for all examples.
        '''
        self._nameOfData = nameOfData
        self._h5Files = h5Files
        self._seed = seed
        self._shuffle = shuffle
        self._exampleId = exampleId
        self._nDigitsInFile = nDigitInFile
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
    def setH5Files(self, h5Files):
        '''
        Set H5 files, to facilitate the implementation of shuffling H5 files in multi-processing 
        setting without generating duplicated examples from workers, because each worker has its
        own copy of dataset instance. 
        '''
        self._h5Files = h5Files
        
    def __iter__(self):
            
        workerInfo = get_worker_info()
        if workerInfo is None:
            # single-process data loading
            return iter(H5Reader(self._h5Files, self._nameOfData, 
                        shuffle = self._shuffle, exampleId = self._exampleId,
                        nDigitInFile = self._nDigitsInFile))
        else: # in a worker process
            nFilesPerWorker = int(math.floor(len(self._h5Files) / 
                     float(workerInfo.num_workers)))
            nFilesLeft = int(len(self._h5Files) % float(workerInfo.num_workers))
            if workerInfo.id >= nFilesLeft:
                iStart = nFilesPerWorker * workerInfo.id + nFilesLeft
                iEnd = iStart + nFilesPerWorker
                if iStart >= len(self._h5Files):
                    # more workers than the number of files
                    return iter([])
            else:
                iStart = (nFilesPerWorker + 1) * workerInfo.id
                iEnd = iStart + nFilesPerWorker + 1
            return iter(H5Reader(self._h5Files[iStart:iEnd], self._nameOfData, 
                        shuffle = self._shuffle, exampleId = self._exampleId,
                        nDigitInFile = self._nDigitsInFile))    
            
        
            
            
            