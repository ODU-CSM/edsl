"""
This module provides the `TrainModel` class and supporting methods.
"""
import logging
import math
from time import time

import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from .sgd_trainer import SGDTrainer

logger = logging.getLogger("fugep")


class StandardSGDTrainer(SGDTrainer):
    """
    This class ties together the various objects and methods needed to
    train and validate a model.

    TrainModel saves a checkpoint model (overwriting it after
    `save_checkpoint_every_n_steps`) as well as a best-performing model
    (overwriting it after `report_stats_every_n_steps` if the latest
    validation performance is better than the previous best-performing
    model) to `output_dir`.

    TrainModel also outputs 2 files that can be used to monitor training
    as Selene runs: `fugep.train_model.train.txt` (training loss) and
    `fugep.train_model.validation.txt` (validation loss & average
    ROC AUC). The columns in these files can be used to quickly visualize
    training history (e.g. you can use `matplotlib`, `plt.plot(auc_list)`)
    and see, for example, whether the model is still improving, if there are
    signs of overfitting, etc.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_sampler : fugep.samplers.Sampler
        The example generator.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    optimizer_kwargs : dict
        The dictionary of keyword arguments to pass to the optimizer's
        constructor.
    batch_size : int
        Specify the batch size to process examples. Should be a power of 2.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    report_stats_every_n_steps : int
        The frequency with which to report summary statistics. You can
        set this value to be equivalent to a training epoch
        (`n_steps * batch_size`) being the total number of samples
        seen by the model so far. Selene evaluates the model on the validation
        dataset every `report_stats_every_n_steps` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `output_dir`.
    output_dir : str
        The output directory to save model checkpoints and logs in.
    save_checkpoint_every_n_steps : int or None, optional
        Default is 1000. If None, set to the same value as
        `report_stats_every_n_steps`
    save_new_checkpoints_after_n_steps : int or None, optional
        Default is None. The number of steps after which Selene will
        continually save new checkpoint model weights files
        (`checkpoint-<TIMESTAMP>.pth.tar`) every
        `save_checkpoint_every_n_steps`. Before this point,
        the file `checkpoint.pth.tar` is overwritten every
        `save_checkpoint_every_n_steps` to limit the memory requirements.
    n_validation_samples : int or None, optional
        Default is `None`. Specify the number of validation samples in the
        validation set. If `n_validation_samples` is `None` and the data sampler
        used is the `fugep.samplers.IntervalsSampler` or
        `fugep.samplers.RandomSampler`, we will retrieve 32000
        validation samples. If `None` and using
        `fugep.samplers.MultiSampler`, we will use all
        available validation samples from the appropriate data file.
    n_test_samples : int or None, optional
        Default is `None`. Specify the number of test samples in the test set.
        If `n_test_samples` is `None` and

            - the sampler you specified has no test partition, you should not
              specify `evaluate` as one of the operations in the `ops` list.
              That is, Selene will not automatically evaluate your trained
              model on a test dataset, because the sampler you are using does
              not have any test data.
            - the sampler you use is of type `fugep.samplers.OnlineSampler`
              (and the test partition exists), we will retrieve 640000 test
              samples.
            - the sampler you use is of type
              `fugep.samplers.MultiSampler` (and the test partition
              exists), we will use all the test samples available in the
              appropriate data file.

    cpu_n_threads : int, optional
        Default is 1. Sets the number of OpenMP threads used for parallelizing
        CPU operations.
    use_cuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    logging_verbosity : {0, 1, 2}, optional
        Default is 2. Set the logging verbosity level.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    checkpointResume : str or None, optional
        Default is `None`. If `checkpointResume` is not None, it should be the
        path to a model file generated by `torch.save` that can now be read
        using `torch.load`.
    use_scheduler : bool, optional
        Default is `True`. If `True`, learning rate scheduler is used to
        reduce learning rate on plateau. PyTorch ReduceLROnPlateau scheduler 
        with patience=16 and factor=0.8 is used.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    sampler : fugep.samplers.Sampler
        The example generator.
    criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer : torch.optim.Optimizer
        The optimizer to minimize loss with.
    batch_size : int
        The size of the mini-batch to use during training.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    nth_step_report_stats : int
        The frequency with which to report summary statistics.
    nth_step_save_checkpoint : int
        The frequency with which to save a model checkpoint.
    use_cuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    data_parallel : bool
        Whether to use multiple GPUs or not.
    output_dir : str
        The directory to save model checkpoints and logs.

    """

    def __init__(self,
                 model,
                 dataSampler,
                 outputDir,
                 maxNSteps, # TODO: default None, use early stopping
                 lossCalculator = None,
                 optimizerClass = None,
                 optimizerKwargs = None,
                 gradMethod = None,
                 batchSize = 64,
                 nStepsStatReport = 100,
                 nStepsCheckpoint = 1000,
                 nStepsStartCheckpoint = None,
                 nMinMinorsReport = 10,
                 nValidationSamples=None,
                 nTestSamples=None,
                 nCpuThreads = 1,
                 useCuda = False,
                 dataParallel = False,
                 loggingVerbosity=2,
                 preloadValData=False,
                 preloadTestData=False,
                 checkpointResume = None,
                 metrics=dict(roc_auc=roc_auc_score, avg_precision=average_precision_score,
                              f1_negPos=f1_score, accuracy=accuracy_score, recall=recall_score,
                              confusion_TnFpFnTp=confusion_matrix),
                 useScheduler=True,
                 deterministic=False,
                 valOfMisInTarget = None,
                 imbalanced = False,
                 patience = None):
        """
        Constructs a new `StandardSGDTrainer` object.
        """
        super(StandardSGDTrainer, self).__init__(
            model = model,
            dataSampler = dataSampler,
            lossCalculator = lossCalculator,
            optimizerClass = optimizerClass,
            optimizerKwargs = optimizerKwargs,
            gradMethod = gradMethod,
            outputDir = outputDir,
            maxNSteps = maxNSteps, 
            batchSize = batchSize,
            nStepsStatReport = nStepsStatReport,
            nStepsCheckpoint = nStepsCheckpoint,
            nStepsStartCheckpoint = nStepsStartCheckpoint,
            nMinMinorsReport = nMinMinorsReport,
            nValidationSamples = nValidationSamples,
            nTestSamples = nTestSamples,
            nCpuThreads = nCpuThreads,
            useCuda = useCuda,
            dataParallel = dataParallel,
            loggingVerbosity = loggingVerbosity,
            preloadValData = preloadValData,
            preloadTestData = preloadTestData,
            metrics = metrics,
            useScheduler = useScheduler,
            deterministic = deterministic,
            valOfMisInTarget = valOfMisInTarget,
            imbalanced = imbalanced)

        if patience:
            self._patience = patience
        else:
            self._patience = np.inf
        if checkpointResume is not None:
            self._loadCheckpoint(checkpointResume)


    def trainAndValidate(self):
        """
        Trains the model and measures validation performance.

        """
        self._counter = 0
        validationPerformance = os.path.join(
            self.outputDir, "validation-performance.txt")
        for step in range(self._startStep, self.maxNSteps):
            self.step = step
            self._train()

            if (step + 1) % self.nStepsCheckpoint == 0:
                self._checkpoint()
            if self.step and (self.step + 1) % self.nStepsStatReport == 0:
                self._validate()
                if self._counter == self._patience:
                    break

        self._validationMetrics.writeValidationFeatureScores(
            validationPerformance)

        self.sampler.saveDatasetToFile("train", close_filehandle = True)


    def _train(self):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.
        """
        tStart = time()
        
        self.sampler.setMode("train")
        batchData = self._getBatchData(self.sampler.mode)
        sumOfLoss, nEffTerms = \
            self.model.fit(batchData)
        # track the loss
        self._trainLoss.add(sumOfLoss, nEffTerms)

        tFinish = time()

        self._timePerStep.append(tFinish - tStart)
        if self.step and (self.step + 1) % self.nStepsStatReport == 0:
            logger.info(("[STEP {0}] average number "
                         "of steps per second: {1:.1f}").format(
                self.step, 1. / np.average(self._timePerStep)))
            self._trainLogger.info(self._trainLoss.getAveLoss())
            logger.info("training loss: {0}".format(
                self._trainLoss.getAveLoss()))
            self._timePerStep = []
            self._trainLoss.reset()

    def _validate(self):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        self.sampler.setMode("validate")
        if self.preloadValData == True:
            loss, predictions = self.model.validate(self._validationData)
            validScores = self._validationMetrics.update(
                predictions, self._allValidationTargets, step=self.step)
        else:
            losses = []
            predictions = []
            allTargets = []

            batchData = self._getBatchData(self.sampler.mode)
            while batchData != None:
                batchLoss, batchPreds = self.model.validate([batchData])
                predictions.append(batchPreds)
                allTargets.append(batchData['targets'])
                losses.append(batchLoss)
                batchData = self._getBatchData(self.sampler.mode)
    
            predictions = np.vstack(predictions)
            allTargets = np.vstack(allTargets)
            loss = np.mean(losses)
            validScores = self._validationMetrics.update(
                predictions, allTargets, step=self.step)
        for name, score in validScores.items():
            logger.info("validation {0}: {1}".format(name, score))

        validScores["loss"] = loss

        to_log = [str(loss)]
        for k in sorted(self._validationMetrics.metrics.keys()):
            if k in validScores and validScores[k].any():
                to_log.append(str(validScores[k]))
            else:
                to_log.append("NA")
        self._validationLogger.info("\t".join(to_log))

        # scheduler update
        if self._useScheduler:
            self.scheduler.step(
                math.ceil(loss * 1000.0) / 1000.0)
        print('Current learning rate: ', self.model._optimizer.param_groups[0]['lr'])

        # early stopping
        if self._counter < self._patience and loss < self._minLoss:
            self._counter = 0
            # save best_model
            self.best_model = self.model
            self._minLoss = loss
            self._saveCheckpoint({
                "step": self.step,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.getStateDict(),
                "min_loss": self._minLoss,
                "optimizer": self.model.getOptimizer().state_dict()}, True)
            logger.debug("Updating `best_model.pth.tar`")
        else:
            self._counter += 1
            logger.debug(f'early stopping counter: {self._counter} out of {self._patience}')

        logger.info("validation loss: {0}".format(loss))
        
    


