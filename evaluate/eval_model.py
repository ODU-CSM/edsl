"""
This module implements the ModelEvaluator class.
"""
import logging
import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..utils import initialize_logger
from .metrics import PerformanceMetrics
from .motif import *


logger = logging.getLogger("fugep")


class ModelEvaluator(object):
    """
    Evaluate model on a test set of sequences with known targets.

    Parameters
    ----------
    model : torch.nn.Module
        The model architecture.
    lossCalculator : torch.nn._Loss
        The loss function that was optimized during training.
    dataSampler : fugep.samplers.Sampler
        Used to retrieve samples from the test set for evaluation.
    features : list(str)
        List of distinct features the model predicts.
    trainedModelPath : str
        Path to the trained model file, saved using `torch.save`.
    outputDir : str
        The output directory in which to save model evaluation and logs.
    batchSize : int, optional
        Default is 64. Specify the batch size to process examples.
        Should be a power of 2.
    nTestSamples : int or None, optional
        Default is `None`. Use `nTestSamples` if you want to limit the
        number of samples on which you evaluate your model. If you are
        using a sampler of type `fugep.samplers.OnlineSampler`,
        by default it will draw 640000 samples if `nTestSamples` is `None`.
    nMinMinorsReport : int, optional
        Default is 10. In the final test set, each class/feature must have
        more than `nMinMinorsReport` positive samples in order to
        be considered in the test performance computation. The output file that
        states each class' performance will report 'NA' for classes that do
        not have enough positive samples.
    useCuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    dataParallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    useFeaturesOrd : list(str) or None, optional
        Default is None. Specify an ordered list of features for which to
        run the evaluation. The features in this list must be identical to or
        a subset of `features`, and in the order you want the resulting
        `test-targets.npz` and `testpredictions.npz` to be saved.
        TODO: Need a fix. This function seems not being currently implemented correctly, Javon, 05/26/2021

    Attributes
    ----------
    model : PredMWrapper
        The trained model.
    lossCalculator : torch.nn._Loss
        The model was trained using this loss function.
    sampler : fugep.samplers.Sampler
        The example generator.
    batchSize : int
        The batch size to process examples. Should be a power of 2.
    useCuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    dataParallel : bool
        Whether to use multiple GPUs or not.

    """

    def __init__(self,
                 model,
                 dataSampler,
                 trainedModelPath,
                 outputDir,
                 lossCalculator = None, # if None, lossCalculator is set to model directly
                 batchSize = 64,
                 nTestSamples = None,
                 nMinMinorsReport = 10,
                 useCuda=False,
                 dataParallel=False,
                 useFeaturesOrd=None,
                 valOfMisInTarget = None,
                 loggingVerbosity = 2,
                 imbalanced = False,
                 nValidationSamples = None,
                 motif = False):
        self._imbalanced = imbalanced
        self._nValidationSamples = nValidationSamples
        self.lossCalculator = lossCalculator
        self.motif = motif
        self.model = model
        self.useCuda = useCuda
        if self.useCuda:
            self.model.toUseCuda()
            
        trainedModel = torch.load(
            trainedModelPath, map_location = lambda storage, location: storage)
        if "state_dict" in trainedModel:
            self.model.init(trainedModel["state_dict"])
        else:
            self.model.init(trainedModel)

        self.sampler = dataSampler
        if 'cWeights' in trainedModel:
            self.sampler.setClassWeights(trainedModel['cWeights'])

        self.outputDir = outputDir
        os.makedirs(outputDir, exist_ok = True)

        self.features = dataSampler.getFeatures()
        self._useIxs = list(range(len(self.features)))
        if useFeaturesOrd is not None:
            featureIxs = {f: ix for (ix, f) in enumerate(self.features)}
            self._useIxs = []
            self.features = []

            for f in useFeaturesOrd:
                if f in featureIxs:
                    self._useIxs.append(featureIxs[f])
                    self.features.append(f)
                else:
                    warnings.warn(("Feature {0} in `useFeaturesOrd` "
                                   "does not match any features in the list "
                                   "`features` and will be skipped.").format(f))
            self._saveFeaturesOrdered()

        initialize_logger(
            os.path.join(self.outputDir, "fugep.log"),
            verbosity = loggingVerbosity)

        self.dataParallel = dataParallel
        if self.dataParallel:
            self.model.toDataParallel()
            logger.debug("Wrapped model in DataParallel")

        self.batchSize = batchSize

        self._valOfMisInTarget = valOfMisInTarget
        self._metrics = PerformanceMetrics(
            self._getFeatureByIndex,
            nMinMinorsReport = nMinMinorsReport,
            valOfMisInTarget = self._valOfMisInTarget)

        self._testData, self._allTestTargets = \
            self.sampler.getDataAndTargets(self.batchSize, nTestSamples)
        # TODO: we should be able to do this on the sampler end instead of
        # here. the current workaround is problematic, since
        # self._testData still has the full featureset in it, and we
        # select the subset during `evaluate`
        self._allTestTargets = self._allTestTargets[:, self._useIxs]

    def _saveFeaturesOrdered(self):
        """
        Write the feature ordering specified by `useFeaturesOrd`
        after matching it with the `features` list from the class
        initialization parameters.
        """
        fp = os.path.join(self.outputDir, 'use-features-ord.txt')
        with open(fp, 'w+') as fileHdl:
            for f in self.features:
                fileHdl.write('{0}\n'.format(f))

    def _getFeatureByIndex(self, index):
        """
        Gets the feature at an index in the features list.

        Parameters
        ----------
        index : int

        Returns
        -------
        str
            The name of the feature/target at the specified index.

        """
        return self.features[index]

    def evaluate(self):
        """
        Passes all samples retrieved from the sampler to the model in
        batches and returns the predictions. Also reports the model's
        performance on these examples.

        Returns
        -------
        dict
            A dictionary, where keys are the features and the values are
            each a dict of the performance metrics (currently ROC AUC and
            AUPR) reported for each feature the model predicts.

        """

        thresholds = None
        # find optimal thresholds
        if self._imbalanced:
            valData, valTargets = \
                self.sampler.getDataAndTargets(self.batchSize, self._nValidationSamples, mode='validate')
            valTargets = valTargets[:, self._useIxs]
            _, val_predictions = self.model.validate(valData, evalType='validate', sampler=self.sampler)
            thresholds = self._metrics.getThresh(val_predictions, valTargets)
        print(thresholds)

        #self._valData, _ = self.sampler.getDataAndTargets(self.batchSize, mode='validate')
        #self.motif = False
        if self.motif:
            features = self.model.validate(self._testData, evalType='validate', sampler=self.sampler,
                                                              layer=1)
            # get 1st convolution layer filters
            W = activation_pwm(np.transpose(features, (0, 2, 1)),
                               X=np.concatenate([x['sequence'] for x in self._testData]), threshold=0.5, window=25)
            # get 1st convolution layer filters
            fig, logo = plot_filters(W, num_cols=6, figsize=(30,10))
            outfile = os.path.join(self.outputDir, 'filters25.pdf')
            fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
            # clip filters about motif to reduce false-positive Tomtom matches
            #W_clipped = clip_filters(W, threshold=0.5, pad=3)
            output_file = os.path.join(self.outputDir, 'sfem25' + '.meme')
            meme_generate(W, output_file)
            #loss, predictions = self.model.validate(self._testData, evalType='validate', sampler=self.sampler)
        else:
            loss, predictions = self.model.validate(self._testData, evalType='validate', sampler=self.sampler)

        #trainDist = self.model.getTrainDist(self.sampler)
        #import matplotlib
        #matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        #plt.hist(trainDist, density=True, bins=100, label='train')
        #plt.hist(self.model.valDist, density=True, bins=100, label='validation')
        plt.hist(self.model.testDistNeg, density=True, bins=100, label='test negative')
        plt.hist(self.model.testDistPos, density=True, bins=40, label='test positive')
        plt.legend()
        plt.savefig(f'{self.outputDir}/density_histogram.png')
        plt.close()
        import seaborn as sn
        #sn.kdeplot(self.model.valDist, label='validation')
        sn.kdeplot(self.model.testDistNeg, label='test negative')
        sn.kdeplot(self.model.testDistPos, label='test positive')
        plt.legend()
        plt.savefig(f'{self.outputDir}/density_curve.png')
        average_scores = self._metrics.update(predictions, self._allTestTargets, thresholds)

        self._metrics.visualize(predictions, self._allTestTargets, self.outputDir)

        np.savez_compressed(
            os.path.join(self.outputDir, "test-predictions.npz"),
            data = predictions)

        np.savez_compressed(
            os.path.join(self.outputDir, "test-targets.npz"),
            data=self._allTestTargets)

        logger.info("test loss: {0}".format(loss))
        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(
            self.outputDir, "test-performance.txt")
        feature_scores_dict = self._metrics.write_feature_scores_to_file(
            test_performance)

        return feature_scores_dict
