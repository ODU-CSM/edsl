'''
Created on Apr 30, 2021

@author: jsun
'''

import numpy as np

from .h5_sampler import H5Sampler
from ..dataloader.h5 import H5DataLoader
from .data_collators import IntervalCollator

class IntervalH5Sampler(H5Sampler):
    '''
    Class for sampling from H5 files of genomic intervals
    '''
    _NAME_OF_DATA = ['sequence', 'targets'] # names of datasets to load from H5 file

    def __init__(self,
                 h5FileDir,
                 train = None,
                 validate = None,
                 batchSize = 64,
                 features = None,  # if none, all features in h5 will be included
                 test = None,
                 unpackbits = False,
                 mode="train",
                 weightSampByCls = True,
                 seed = None,
                 nWorkers = 1,
                 save_datasets = [],
                 output_dir = None):
        """
        Constructs a new `IntervalH5Sampler` object.
        """
        super(IntervalH5Sampler, self).__init__(
            h5FileDir = h5FileDir,
            train = train,
            validate = validate,
            bitPacked = unpackbits,
            features = features,
            test = test,
            mode = mode,
            weightSampByCls = weightSampByCls,
            save_datasets = save_datasets,
            output_dir = output_dir)

        self._dataloaders['train'] = \
            H5DataLoader(self._train,
                 self._NAME_OF_DATA,
                 collateFunc = IntervalCollator(self._seqLen, self._featsInH5,
                                cWeights = self._cWeights, iFeatsPred = self._iFeatsPred,
                                unpackbits = unpackbits, nSeqBase = self._N_SEQ_BASE),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = True)
        self._iterators['train'] = iter(self._dataloaders['train'])
        
        self._dataloaders['validate'] = \
            H5DataLoader(self._validate,
                 self._NAME_OF_DATA,
                 collateFunc = IntervalCollator(self._seqLen, self._featsInH5,
                                cWeights = self._cWeights, iFeatsPred = self._iFeatsPred,
                                unpackbits = unpackbits, nSeqBase = self._N_SEQ_BASE),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = False)
        self._iterators['validate'] = iter(self._dataloaders['validate'])
        
        if test is not None:
            self._dataloaders['test'] = \
            H5DataLoader(self._test,
                 self._NAME_OF_DATA,
                 collateFunc = IntervalCollator(self._seqLen, self._featsInH5,
                                cWeights = self._cWeights, iFeatsPred = self._iFeatsPred,
                                unpackbits = unpackbits, nSeqBase = self._N_SEQ_BASE),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = False)
        self._iterators['test'] = iter(self._dataloaders['test'])

    def sample(self, batchSize = 64, mode = None):
        """
        Fetches a mini-batch of the data from the sampler.

        Parameters
        ----------
        batchSize : int, optional
            Default is 1. The size of the batch to retrieve.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        """
        mode = mode if mode else self.mode
        self.setBatchSize(batchSize, mode = mode)
        try:
            seq, targets, weights = next(self._iterators[mode])
        except StopIteration:
            #If DataLoader iterator reaches its length, reinitialize
            if mode == 'train':
                self._iterators[mode] = iter(self._dataloaders[mode])
                seq, targets, weights = next(self._iterators[mode])
            else:
                self._iterators[mode] = iter(self._dataloaders[mode])
                return None
        
        return {'sequence': seq.numpy(),
                'targets': targets.numpy(),
                'weights': weights.numpy()}    

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
        
        mode = mode if mode is not None else self.mode
        self.setBatchSize(batchSize, mode = mode)
        
        dataAndTargets = []
        trgtsMat = []
        
        if nSamps is None:
            # retrieve all samples managed by the corresponding sampler
            # reset the iterator
            self._iterators[mode] = iter(self._dataloaders[mode])
            while True:
                try:
                    seq, targets, weights = next(self._iterators[mode])
                    dataBatch = {'sequence': seq.numpy(),
                                'targets': targets.numpy(),
                                'weights': weights.numpy()}
                    dataAndTargets.append(dataBatch)
                    trgtsMat.append(dataBatch['targets'])
                except StopIteration:
                    break
        else:
            count = batchSize
            while count < nSamps:
                dataBatch = self.sample(batchSize = batchSize, mode = mode)
                dataAndTargets.append(dataBatch)
                trgtsMat.append(dataBatch['targets'])
                count += batchSize
            remainder = batchSize - (count - nSamps)
            dataBatch = self.sample(batchSize = remainder, mode = mode)
            dataAndTargets.append(dataBatch)
            trgtsMat.append(dataBatch['targets'])
        
        trgtsMat = np.vstack(trgtsMat)
        return dataAndTargets, trgtsMat
            
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
