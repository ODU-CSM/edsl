'''
Created on May 5, 2021

@author: jsun
'''
"""
This module provides the `MultiSampler` class, which accepts
either an online sampler or a file sampler for each mode of
sampling (train, test, validation).
MultiSampler is a subclass of Sampler.
"""
from abc import ABCMeta
import os
import glob  
import h5py as h5  
import numpy as np

from ..sampler import Sampler
from ..utils import calcCWeightByH5File
from .utils import countExamsInH5s

class H5Sampler(Sampler, metaclass = ABCMeta):
    """
    This sampler draws samples from individual file samplers or data loaders
    that corresponds to training, validation, and testing (optional) modes.
    MultiSampler calls on the correct file sampler or data loader to draw
    samples for a given mode. Example file samplers are under
    `fugep.samplers.file_samplers` and example data loaders are under
    `fugep.samplers.dataloaders`.

    MultiSampler can use either file samplers or data loaders for
    different modes. Using data loaders for some modes while using file samplers
    for other modes are also allowed. The file samplers parse data files
    (e.g. bed, mat, or hdf5). The data loaders provide multi-worker iterators
    that draw samples from online samplers (i.e. on-the-fly sampling). As data
    loaders support parallel sampling, they are generally recommended for
    sampling speed.

    Parameters
    ----------
    h5FileDir: the directory that holds all the input h5 files
    train : chromosomes used for training
    validate : chromosomes used for validation
    test : chromosomes used for testing
    mode : str, optional
        Default is "train". Must be one of `{train, validate, test}`. The
        starting mode in which to run the sampler.
    save_datasets : list(str) or None, optional
        Default is None. Currently, we are only including this parameter
        so that `MultiSampler` is consistent with the `Sampler` interface.
        The save dataset functionality for MultiSampler has not been
        defined yet.
    output_dir : str or None, optional
        Default is None. Only used if the sampler has any data or
        logging statements to save to file. Currently not used in
        `MultiSampler`.

    Attributes
    ----------
    modes : list(str)
        A list of the modes that the object may operate in.
    mode : str or None
        Default is `None`. The current mode that the object is operating in.

    """
    
    _N_SEQ_BASE = 4
    
    def __init__(self,
                 h5FileDir,
                 train = None,
                 validate = None,
                 bitPacked = False,
                 features = None,  # if none, all features in h5 will be included
                 needTarget = True,
                 needExamWeight = True,
                 test = None,
                 mode="train",
                 weightSampByCls = True,
                 clsWeights = None,
                 valOfMisInTarget = None, # value representing missing in target
                 save_datasets = [],
                 output_dir = None):
        """
        Constructs a new `H5Sampler` object.
        
        Parameters
        ----------
        needTarget : bool, optional
            The default is True. Indicate whether to load target
        """
        # identify all h5 files under the given h5FileDir
        h5Files = glob.glob(os.path.join(h5FileDir, '*.h5'))
        # obtain the list of files for training, validation and testing
        self._train = []
        self._validate = []
        self._test = []
        for h5File in h5Files:
            chrom = os.path.basename(h5File).split('_')[0]
            # files in each list are indexed to facilitate the identification of examples
            if train is not None and chrom in train:
                self._train.append((len(self._train), h5File))
            elif validate is not None and chrom in validate:
                self._validate.append((len(self._validate), h5File))
            elif test is not None and chrom in test:
                self._test.append((len(self._test), h5File))
        
        if mode == 'train':
            if len(self._train) == 0:
                raise ValueError('No chromosomes are specified for training')
            if len(self._validate) == 0:
                raise ValueError('No chromosomes are specified for validation')
            sampH5File = self._train[0][1]
        elif mode == 'validate':
            if len(self._validate) == 0:
                raise ValueError('No chromosomes are specified for validation')
            sampH5File = self._validate[0][1]
        elif mode == 'test':
            if len(self._test) == 0:
                raise ValueError('No chromosomes are specified for testing')
            sampH5File = self._test[0][1]

        # retrieve features from a H5 file
        with h5.File(sampH5File, 'r') as fh:
            self._featsInH5 = \
                [feat.decode() for feat in fh['features'][()]]
            self._seqLen = fh['sequence_length'][()]
        
        # initialize members in the super class    
        super(H5Sampler, self).__init__(
            self._featsInH5,
            save_datasets = save_datasets,
            output_dir = output_dir,
            valOfMisInTarget = valOfMisInTarget,
            clsWeights = clsWeights)
        
        self._bitPacked = bitPacked
        # add test operation mode
        if len(self._test) > 0:
            self.modes.append('test')
        
        # set the features to predict for
        if features is not None: 
            for feat in features:
                if feat not in self._featsInH5:
                    raise ValueError('Feature: {0} can not be found in H5 file'.format(feat))
                
            self._features = features
            self._iFeatsPred = [i for i, feat in enumerate(self._featsInH5) if feat in features]
        else:
            self._features = self._featsInH5
            self._iFeatsPred = None
        
        self._indexToFeature = {i: f for (i, f) in enumerate(self._features)}
        self._needTarget = needTarget
        self._needExamWeight = needExamWeight
        
        self._dataloaders = {} # to be initialized in subclass
        self._iterators = {} # to be initialized in subclass

        self.mode = mode
        
        # set class weights
        if weightSampByCls and self._cWeights is None \
            and len(self._train) > 0:
            # compute class weights from training data
            self._cWeights = calcCWeightByH5File(self._train, features = features, 
                bitPacked = bitPacked, valueOfMissing = self._valOfMisInTarget)
        
        # count total training, validation, testing examples
        cnts = countExamsInH5s(self._train)
        allCnts = cnts
        self.nTtlTrain = np.sum(cnts)
        # validation set
        if len(self._validate) > 0:
            cnts = countExamsInH5s(self._validate)
            self.nTtlValidate = np.sum(cnts)
            allCnts += cnts
        # test set
        if len(self._test) > 0:
            cnts = countExamsInH5s(self._test)
            self.nTtlTest = np.sum(cnts)
            allCnts += cnts
        
        # set the number of digits needed to index examples within any file
        # to facilitate the creation of unique ID for examples
        self._nDigitsInFile = len(str(np.max(allCnts)))
        
    def setMode(self, mode):
        """
        Sets the sampling mode.

        Parameters
        ----------
        mode : str
            The name of the mode to use. It must be one of
            `Sampler.BASE_MODES` ("train", "validate") or "test" if
            the test data is supplied.

        Raises
        ------
        ValueError
            If `mode` is not a valid mode.

        """
        if mode not in self.modes:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.modes))
        self.mode = mode

    def setBatchSize(self, batchSize, mode = None):
        """
        Sets the batch size for DataLoader for the specified mode,
        if the specified batchSize does not equal to the current batchSize.
        Parameters
        ----------
        batchSize : int
            The batch size for the mode.
        mode : str, optional
            Default is None. The  mode to set batchSize
            If None, will use the current mode `self.mode`.
        """
        if mode is None:
            mode = self.mode

        if self._dataloaders[mode] is not None:
            batch_size_matched = True
            if self._dataloaders[mode].batch_sampler is not None:
                if self._dataloaders[mode].batch_sampler.batch_size != batchSize:
                    self._dataloaders[mode].batch_sampler.batch_size = batchSize
                    batch_size_matched = False
            else:
                if self._dataloaders[mode].batch_size != batchSize:
                    self._dataloaders[mode].batch_size = batchSize
                    batch_size_matched = False

            if not batch_size_matched:
                print("Reset data loader for mode {0} to use the new batch "
                      "size: {1}.".format(mode, batchSize))
                self._iterators[mode] = iter(self._dataloaders[mode])

    def getFeatureByIndex(self, index):
        """
        Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        index : int
            The index of the feature to retrieve the name for.

        Returns
        -------
        str
            The name of the feature occurring at the specified index.

        """
        return self._indexToFeature[index]
    
    def getSequenceLength(self):
        '''
        This method return the sequence length
        '''
        return self._seqLen
        
    
    def saveDatasetToFile(self, mode, close_filehandle=False):
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
