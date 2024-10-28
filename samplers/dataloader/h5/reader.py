'''
This module provides the base class of H5Reader

Created on Apr 30, 2021

@author: jsun
'''

import h5py as h5
import numpy as np
import six
from six.moves import range

class H5Reader():
    '''
    Base class of H5Reader
    '''
    def __init__(self, h5Files, nameOfData, seed = None, shuffle = True, 
                 exampleId = False, nDigitInFile = None):
        '''
        Parameters
        -------------------------
        h5Files: the list of h5 files from which data are retrieved
        nameOfData: the name of the dataset to retrieve form the h5 files
        exampleId : bool, optional
            Default is False
            Whether to include id of example in the data batch.
            Such an id is created as iFile * 10^nDigitInFile + iInFile, where iFile is the 
            index of h5 file found in the input h5Files; and iInFile is the index of the 
            example within any specific file
        nDigitInFile : int, optional
            Default is None. When exampleId is True, this is mandatory. 
            Specify the number of lower digits reserved for indexing examples within a file
            while creating unique IDs for all examples.
        '''
        
        self._h5Files = h5Files
        self._nameOfData = nameOfData
        self._shuffle = shuffle
        self._seed = seed
        self._exampleId = exampleId
        if exampleId and nDigitInFile is None:
            raise ValueError('Parameter nDigitInFile cannot be none when example ID is needed')
        if nDigitInFile is not None:
            self._iFileMul = 10 ** nDigitInFile
        
        if self._seed is not None:
            np.random.seed(self._seed)
    
    def __iter__(self):
        # permute the ordering of the file
        if self._shuffle:
            np.random.shuffle(self._h5Files)
            
        # iterate through files
        for h5File in self._h5Files:
            h5FileHdl = h5.File(h5File[1], 'r')
            
            datasets = dict()
            for name in self._nameOfData:
                datasets[name] = h5FileHdl[name]
            nSampInFile = len(list(datasets.values())[0])
            
            idx = np.arange(nSampInFile)
            if self._shuffle:
                # Shuffle data within the entire file, which requires reading
                # the entire file into memory
                np.random.shuffle(idx)
                for name, value in six.iteritems(datasets):
                    datasets[name] = value[:len(idx)][idx]
            
            # iterate through samples in the file
            for iSamp in range(nSampInFile):
                dataOfSamp = dict()
                if self._exampleId:
                    dataOfSamp['id'] = self._iFileMul * h5File[0] + idx[iSamp]
                for name in self._nameOfData:
                    dataOfSamp[name] = datasets[name][iSamp]
                yield  dataOfSamp    
            
            h5FileHdl.close()
            

                
                
                
                