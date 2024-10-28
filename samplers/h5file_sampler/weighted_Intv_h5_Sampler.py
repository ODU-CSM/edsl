'''
Created on Aug 10, 2021

@author: javon
'''

import h5py as h5
import numpy as np
import logging

from .h5_sampler import  H5Sampler
from ..dataloader.h5 import H5DataLoader
from .data_collators import IntervalCollator

logger = logging.getLogger("fugep")

class WeightedIntvH5Sampler(H5Sampler):
    '''
    Implementation of weighted interval H5 sampler
    
    TODO: Implement option to only load sequence data for one-class classification
    '''
    
    _STATE = ['inactive', 'weight-computing', 'production']
    
    _NAME_OF_DATA = ['sequence', 'targets'] # names of datasets to load from H5 file
    
    '''
    Class constructor
    '''
    def __init__(self, 
                 h5FileDir,
                 train = None,
                 nTrainToSample = None,   
                 validate = None,
                 nValidToSample = None, 
                 batchSize = 64,
                 features = None,  
                 needTarget = True,
                 needExamWeight = True,
                 test = None,
                 nTestToSample = None, 
                 unpackbits = False,
                 mode = "train",
                 seed = None,
                 nWorkers = 1,
                 save_datasets = [],
                 output_dir = None):
        '''
        Constructor
        
        Parameters:
        ------------
        nTrainToSample - number of training examples to sample
        nValidToSample - number of validation examples to sample
        nTestToSample -  number of test examples to sample
        '''
        super(WeightedIntvH5Sampler, self).__init__(
            h5FileDir = h5FileDir,
            train = train,
            validate = validate,
            bitPacked = unpackbits,
            features = features,
            needTarget = needTarget,
            needExamWeight = needExamWeight,
            test = test,
            mode = mode,
            weightSampByCls = False,
            save_datasets = save_datasets,
            output_dir = output_dir)
        
        self._nExamsToSample = {'train': 0, 'validate': 0, 'test': 0}
        if nTrainToSample is not None:
            self._nExamsToSample['train'] = nTrainToSample
        if nValidToSample is not None:
            self._nExamsToSample['validate'] = nValidToSample
        if nTestToSample is not None:
            self._nExamsToSample['test'] = nTestToSample 
        
        # # set example source, i.e, the index of the file where the examples locate
        # # and the position (starting from 0) of examples in their corresponding h5 files
        # # this is to facilitate data loading after sampling
        # self._iFileOfExams = dict()
        # self._posOfExams = dict()
        # self._iFileOfExams['train'], self._posOfExams['train'] = self._getExamSrc(self._train)
        # if self._validate is not None:
        #     self._iFileOfExams['validate'], self._posOfExams['validate'] = \
        #         self._getExamSrc(self._validate)
        # if self._test is not None and len(self._test) > 0:
        #     self._iFileOfExams['test'], self._posOfExams['test'] = self._getExamSrc(self._test)
        
        # the weights of examples used in the sampling. Examples with larger
        # weights is expected to have higher chance to get selected.
        # The values in each list should sum up to 1
        self._samplingWeights = {'train': [], 'validate': [], 'test': []}
        self._state = 'inactive' # state of the sampler, can be any value in self._STATE
        # holding sampled examples for training, validate, and testing
        self._selectedExams = {'train': None, 'validate': None, 'test': None}
        # Indicating the starting index of examples in _selectedExams for the next call of data retrieval.
        self._startIndex = {'train': None, 'validate': None, 'test': None}
        self._shuffle = True
        self._trainSampOrder = None
        
        # set full dataset loader
        self._dataloaders['train'] = \
            H5DataLoader(self._train,
                 self._NAME_OF_DATA,
                 collateFunc = IntervalCollator(self._seqLen, self._featsInH5,
                                iFeatsPred = self._iFeatsPred, needExampleId = True,
                                unpackbits = unpackbits, nSeqBase = self._N_SEQ_BASE,
                                needTargets = self._needTarget, needWeights = self._needExamWeight),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = True, 
                 exampleId = True,
                 # NOTE: set dropLast to True for safe, small batch size 
                 # may need to problems in batch normalization layer during training
                 # Set shuffle to be True to avoid examples at the end of h5 files 
                 # are never get used
                 dropLast = True, 
                 nDigitInFile = self._nDigitsInFile)
        if validate is not None:
            self._dataloaders['validate'] = \
                H5DataLoader(self._validate,
                     self._NAME_OF_DATA,
                     collateFunc = IntervalCollator(self._seqLen, self._featsInH5,
                                    iFeatsPred = self._iFeatsPred, needExampleId = True,
                                    unpackbits = unpackbits, nSeqBase = self._N_SEQ_BASE,
                                    needTargets = self._needTarget, needWeights = self._needExamWeight),
                     seed = seed,
                     nWorkers = nWorkers,
                     batchSize = batchSize,
                     shuffle = True, 
                     exampleId = True,
                     dropLast = True,
                     nDigitInFile = self._nDigitsInFile)
        if test is not None:
            self._dataloaders['test'] = \
                H5DataLoader(self._test,
                     self._NAME_OF_DATA,
                     collateFunc = IntervalCollator(self._seqLen, self._featsInH5,
                                    iFeatsPred = self._iFeatsPred, needExampleId = True,
                                    unpackbits = unpackbits, nSeqBase = self._N_SEQ_BASE,
                                    needTargets = self._needTarget, needWeights = self._needExamWeight),
                     seed = seed,
                     nWorkers = nWorkers,
                     batchSize = batchSize,
                     shuffle = True, 
                     exampleId = True,
                     dropLast = True,
                     nDigitInFile = self._nDigitsInFile)
        self._iterators = {'train': None, 'validate': None, 'test': None}
    
    def setNExamToSample(self, nExamsDict):
        '''
        Set the number of example to sample
        
        Parameter
        ----------
        nExamsDict : dict
            A dictionary provides the number of example to sample. Three
            possible keys: 'train', 'validate', 'test'
        '''
        for pt in nExamsDict.keys():
            if pt not in ['train', 'validate', 'test']:
                raise ValueError('Key in the input dictionary can only be train, '
                                 'validate, and test, but gets {0}'.format(pt))
            self._nExamsToSample[pt] = nExamsDict[pt]
    
    def getState(self):
        '''
        Get the state of the sampler
        '''
        return self._state
    
    def _getExamSrc(self, h5Files):
        '''
        Get the index of file where examples locate
        
        Not used current
        '''
        iFile = []
        pos = []
        for iH5File in range(len(h5Files)):
            h5FileHdl = h5.File(h5Files[iH5File][1], 'r')
            nExamInFile = len(h5FileHdl[self._NAME_OF_DATA[0]])
            iFile = iFile + [iH5File] * nExamInFile
            pos =  pos + list(range(nExamInFile))
        return (np.array(iFile), np.array(pos)) 
                
        
    def weightUpdatePrep(self):
        '''
        Set the sampler ready for weight update
        '''
        # set iterator, so the full datasets can be loaded for sampling weights
        # calculation
        self._iterators['train'] = iter(self._dataloaders['train'])
        if self._dataloaders['validate'] is not None:
            self._iterators['validate'] = iter(self._dataloaders['validate'])
        if 'test' in self._dataloaders.keys() and self._dataloaders['test'] is not None:
            self._iterators['test'] = iter(self._dataloaders['test'])
        
        self._state = 'weight-computing'
    
    def _examIdToLoc(self, ids):
        '''
        Computer examples' location from their id, including 
        information about file and the position in the file
        '''
        idArr = np.array(ids)
        iFileMul = 10 ** self._nDigitsInFile
        iFile = idArr // iFileMul 
        pos = idArr % iFileMul
        
        return iFile, pos
        
    def setSamplingWeight(self, weights):
        '''
        Set sampling weight, carry out the sampling, and load the sampled data
        to get the sampler ready
        
        Parameters:
        -----------
        weights- dictionary, keys can be 'train', 'validate', and 'test'. 'train' is mandatory,
            the other two are optional
            Values provides example weights for sampling. If no validation or testing data, set
            the corresponding value to None. Each value is just a list of probabilities corresponding 
            to examples returned by sample() function following the call of weightUpdatePrep() function.
            Because no identifier is used to identify the examples, the provided weights 
            should be arranged in the list in the exact same order in the list of examples returned by a 
            sequence calling of sample() function.
        '''
        
        if self._nExamsToSample['train'] == 0:
            raise ValueError('The number of training examples needs to be set first')
        if 'validate' in weights.keys() and self._nExamsToSample['validate'] == 0:
            raise ValueError('The number of validation examples needs to be set first')
        if 'test' in weights.keys() and self._nExamsToSample['test'] == 0:
            raise ValueError('The number of test examples needs to be set first')
        
        if 'train' not in weights.keys():
            raise ValueError('Weights for training examples are mandatory but cannot be found in '
                             'given input dictionary')
        
        # sampling and load data
        for partition in weights.keys():
            self._samplingWeights[partition] = weights[partition]
            
            if partition == 'train':
                h5Files = self._train
            elif partition == 'validate':
                h5Files = self._validate
            else:
                h5Files = self._test
            
            ids = self._samplingWeights[partition][0]
            wts = self._samplingWeights[partition][1]   
            # sampling
            iSel = np.random.choice(np.arange(len(ids)), 
                        size = self._nExamsToSample[partition], replace = False,
                        p = wts)
            # allocate space for data
            sequences = np.zeros((self._nExamsToSample[partition], self._seqLen, 
                         self._N_SEQ_BASE), dtype = np.float32)
            if self._needTarget:
                if self._iFeatsPred is not None:
                    targets = np.zeros((self._nExamsToSample[partition], len(self._iFeatsPred)))
                else:
                    targets = np.zeros((self._nExamsToSample[partition], len(self._featsInH5)))
            # load data
            iSel.sort()
            iFile, allPos = self._examIdToLoc(ids[iSel])
            iCur = 0
            for iF in np.unique(iFile):
                pos = allPos[iFile == iF]
                pos.sort()
                assert iF == h5Files[iF][0]
                h5FileHdl = h5.File(h5Files[iF][1], 'r')
                # load sequence
                seq = h5FileHdl['sequence'][pos]
                if not self._bitPacked:
                    sequences[iCur:(iCur + len(pos))] = seq
                else:
                    # unpack bits
                    seq = np.unpackbits(seq, axis = -2)
                    nulls = np.sum(seq, axis = -1) == seq.shape[-1]
                    seq = seq.astype(float)
                    seq[nulls, :] = 1.0 / seq.shape[-1]
                    sequences[iCur:(iCur + len(pos))] = seq[:, :self._seqLen, :]
                
                # load target
                if self._needTarget:
                    trt = h5FileHdl['targets'][pos]
                    if self._bitPacked:
                        # unpack bits
                        trt = np.unpackbits(trt, axis = -1).astype(float)
                        trt = trt[:, :len(self._featsInH5)]
                    if self._iFeatsPred is not None:
                        # retain only needed features
                        trt = trt[:, self._iFeatsPred]
                    targets[iCur:(iCur + len(pos))] = trt
                    
                iCur = iCur + len(pos)
                h5FileHdl.close()
            self._selectedExams[partition] = {'sequence': sequences}
            if self._needTarget:
                self._selectedExams[partition]['targets'] = targets
            
            self._startIndex[partition] = 0
        
        # set up the training data serving ordering            
        if self._shuffle:
            self._trainSampOrder = np.random.permutation(self._selectedExams['train']['sequence'].shape[0])
        else:
            self._trainSampOrder = np.range(self._selectedExams['train']['sequence'].shape[0])
        
        self._state = 'production'
        
    def sample(self, batchSize = 64, mode = None, restart = False, 
               fullBatch = False, autoReset = True):
        """
        Fetches a mini-batch of the data from the sampler. Depending on the
        state of the sampler, a mini-batch of the full data (weight-computing state) or 
        that of sampled data is returned. 

        Parameters
        ----------
        batchSize : int, optional
            Default is 1. The size of the batch to retrieve.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        restart : bool, optional
            Default is False. Indicate whether to start from beginning after exhausting 
            selected samples. If False, None is returned when the end is reached, while
            a mini-batch obtained from the front is returned if True. If this parameter
            is set to be True, autoReset is ignored
        fullBatch : bool, optional
            Default is False. Indicate whether the returned mini-batch needs to be full, 
            containing the "batchSize" number of examples. If True, in the case that there 
            is not enough for a full mini-batch at the end of the pool, the function 
            call returns either none or a mini-batch starting from the beginning when restart is set. 
        autoReset: bool, optional
            Default is True. Indicate whether to reset the sampler when the end is reached, so
            the next call of the function will return mini-batch of data from the beginning. The 
            current call is returned with None, which can be used as a signal for the end.
        """
        
        if not (self._state in ['weight-computing', 'production']):
            raise ValueError(('The sampler is in {0} state. Sampling only works when ' + 
                      'the sampler is in weight-computing or production state').format(self._state))
        
        dataColl = dict()
        mode = mode if mode else self.mode
        if self._state == 'weight-computing':
            self.setBatchSize(batchSize, mode = mode)
            try:
                # seq, targets, weights = next(self._iterators[mode])
                dataBatch = next(self._iterators[mode])
                dataColl['sequence'] = dataBatch[0].numpy()
                idxInTuple = 1
                if self._needTarget:
                    dataColl['targets'] = dataBatch[idxInTuple].numpy()
                    idxInTuple += 1
                if self._needExamWeight:
                    dataColl['weights'] = dataBatch[idxInTuple].numpy()
                    idxInTuple += 1
                dataColl['id'] = dataBatch[idxInTuple] 
            except StopIteration:
                return None

        else:
            # production
            iStart = self._startIndex[mode]
            nTotal = self._selectedExams[mode]['sequence'].shape[0]
            if (iStart == nTotal) or (nTotal - iStart < batchSize and fullBatch):
                # end of the pool reached
                if not restart:
                    if autoReset:
                        self._startIndex[mode] = 0
                        if mode == 'train' and self._shuffle:
                            # NOTE: shuffling is only done for training data
                            self._trainSampOrder = np.random.permutation(nTotal)   
                    return None
                else:
                    if mode == 'train' and self._shuffle:
                        # NOTE: shuffling is only done for training data
                        self._trainSampOrder = np.random.permutation(nTotal)
                    self._startIndex[mode] = 0
                    iStart = 0
                    if fullBatch and nTotal < batchSize:
                        # batchSize is bigger than the total number of examples in the pool
                        logger.info('The batchSize {0} is bigger than the '
                                    'total number of examples {1} in the pool'.format(
                                        batchSize, nTotal))
                        return None
            iEnd = np.min([nTotal, iStart + batchSize])
            
            if mode == 'train':
                # training data need to be specially handled because shuffling
                idxs = self._trainSampOrder[range(iStart, iEnd)]
            else:
                idxs = range(iStart, iEnd)
            dataColl['sequence'] = self._selectedExams[mode]['sequence'][idxs]
            if self._needTarget:
                dataColl['targets'] = self._selectedExams[mode]['targets'][idxs]
            if self._needExamWeight:
                dataColl['weights'] = np.full(dataColl['sequence'].shape[0], 1) # the weights are not effectively implemented
                
            self._startIndex[mode] = iEnd
                
        return dataColl   

    def getDataAndTargets(self, batchSize, nSamps = None, mode = None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batchSize : int
            The size of the batches to divide the data into.
        nSamps : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `nSamps` is None, if a FileSampler is specified for the 
            mode, the number of samplers returned is defined by the FileSample, 
            or if a Dataloader is specified, will set `nSamps` to 32000 
            if the mode is `validate`, or 640000 if the mode is `test`. 
            If the mode is `train` you must have specified a value for 
            `nSamps`.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        """
        
        if self._state != 'production':
            raise ValueError(('The sampler is in {0} state, but needs' + 
                              ' to be in production state').format(self._state))
        
        mode = mode if mode is not None else self.mode
        dataBatches = []
        if self._needTarget:
            trgtsMat = []
        if nSamps is None:
            nSamps = self._selectedExams[mode]['sequence'].shape[0]
        elif nSamps > self._selectedExams[mode]['sequence'].shape[0]:
            logger.warning('The required number of examples ({0}) is larger than '
                           'the total ({1}) in the sampler. The total in sampler '
                           'is returned.'.format(nSamps, self._selectedExams[mode]['sequence'].shape[0]))
            nSamps = self._selectedExams[mode]['sequence'].shape[0]
        
        iStart = 0
        iEnd = iStart + batchSize
        while True:
            if mode == 'train':
                idxs = self._trainSampOrder[range(iStart, iEnd)]
            else:
                idxs = range(iStart, iEnd)
                
            batchData = {'sequence': self._selectedExams[mode]['sequence'][idxs]}
            if self._needTarget:
                batchData['targets'] = self._selectedExams[mode]['targets'][idxs]
                trgtsMat.append(self._selectedExams[mode]['targets'][idxs])
            if self._needExamWeight:
                batchData['weights'] = np.full(batchData['sequence'].shape[0], 1)
            dataBatches.append(batchData) 
            
            if iEnd == nSamps:
                break
            iStart = iEnd
            iEnd = np.min([nSamps, iStart + batchSize]) 
        
        if self._needTarget:
            trgtsMat = np.vstack(trgtsMat)
            return dataBatches, trgtsMat
        else:
            return dataBatches
            
    def getValidationSet(self, batchSize, nSamps = None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batchSize : int
            The size of the batches to divide the data into.
        nSamps : int, optional
            Default is None. The total number of validation examples to
            retrieve. If `nSamps` is None,
            then if a FileSampler is specified for the 'validate' mode, the
            number of samplers returned is defined by the FileSample,
            or if a Dataloader is specified, will set `nSamps` to
            32000.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(dict()), numpy.ndarray)
            Tuple containing the list of sequence-target-weight dicts, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets` sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batchSize`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S = nSamps`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        return self.getDataAndTargets(
            batchSize, nSamps, mode = "validate")

    def getTestSet(self, batchSize, nSamps = None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batchSize : int
            The size of the batches to divide the data into.
        nSamps : int or None, optional
            Default is None. The total number of test examples to
            retrieve. If `nSamps` is None,
            then if a FileSampler is specified for the 'test' mode, the
            number of samplers returned is defined by the FileSample,
            or if a Dataloader is specified, will set `nSamps` to
            640000.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(dict()), numpy.ndarray)
            Tuple containing the list of sequence-target-weight dicts, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets` sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batchSize`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S = nSamps`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        return self.getDataAndTargets(
            batchSize, nSamps, mode = "test")

    def saveDatasetToFile(self, mode, close_filehandle = False):
        """
        We implement this function in this class only because the
        TrainModel class calls this method. In the future, we will
        likely remove this method or implement a different way
        of "saving the data" for file samplers. For example, we
        may only output the row numbers sampled so that users may
        reproduce exactly what order the data was sampled.

        Parameters
        ----------
        mode : str
            Must be one of the modes specified in `save_datasets` during
            sampler initialization.
        close_filehandle : bool, optional
            Default is False. `close_filehandle=True` assumes that all
            data corresponding to the input `mode` has been saved to
            file and `saveDatasetToFile` will not be called with
            `mode` again.
        """
        return None    
        