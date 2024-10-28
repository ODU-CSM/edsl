"""
This module provides the `TrainModel` class and supporting methods.
"""
import logging
import math
import os
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from .sgd_trainer import SGDTrainer
from ..utils import LossTracker

logger = logging.getLogger("fugep")


class SemiSVSDTrainer(SGDTrainer):
    """
    This class implements functionalities for training a classifier via
    one-class learning with side information utilized in the framework of
    semi-supervised learning.

    Attributes
    ----------
    
    """

    def __init__(self,
                 model,
                 dataSampler,
                 outputDir,
                 maxNSteps, # TODO: default None, use early stopping
                 lossCalculator = None,
                 optimizerClass = None,
                 optimizerKwargs = None,
                 pretrainedNetwork = None, 
                 centerEps = 0.1,
                 reprLoss = 'soft-boundary', 
                 saveDist = True,
                 solver = 'minimize_scalar', 
                 scalarMethod = 'brent', 
                 lpObj = 'primal', 
                 tol = 0.001,
                 batchSize = 64,
                 nStepsStatReport = 100,
                 nStepsCheckpoint = 1000,
                 nStepsStartCheckpoint = None,
                 nMinMinorsReport = 10,
                 nValidationSamples = None,
                 nTestSamples = None,
                 nCpuThreads = 1,
                 enableMCL = True,
                 enableCL = False,
                 classify=False,
                 negRate = 1,
                 bgDataSampler = None,
                 saveSampWeights = True,
                 nCyclesResample = 2,
                 nStepsWarmup = 200, 
                 nStepsWPToIncr = 100,
                 nStepsUpdateR = 200,  
                 nMClSteps = 200, 
                 nOCLSteps = 200,
                 useCuda = False,
                 dataParallel = False,
                 loggingVerbosity = 2,
                 checkpointResume = None,
                 metrics=dict(roc_auc = roc_auc_score, avg_precision = average_precision_score,
                              f1_negPos = f1_score, accuracy = accuracy_score,
                              recall = recall_score, confusion_TnFpFnTp = confusion_matrix),
                 useScheduler = True,
                 deterministic=False,
                 modeInValidate = 'eval'):
        """
        Constructs a new `SemiSVSDTrainer` object.
        
        Parameters
        ----------
        pretrainedNetwork : str, optional
            Path to saved pretrained network.
            This is mandatory when training from scratch, i.e., not starting from a 
            saved checkpoint. 
        nStepsWarmup : int, optional
            Number of warm up steps before updating R or starting MCL training
            Default is 200.
        nStepsWPToIncr: int, optional
            Number of steps to add to warm up phase if updating R failed
            Default is 100
        nStepsUpdateR: int, optional
            Per every number of steps, R is updated
            Default is 200
        enableMCL : bool, optional 
            Whether to include multi-class learning phase
            Default is True.
        bgDataSampler : fugep.samplers.WeightedIntvH5Sampler, optional
            Sampler for sampling background data to use in 
            multi-class learning (MCL) steps.
            Default is none. This is mandatory when enableMCL is set to True
        saveSampWeights : bool, optional
            Indicates whether to save calculated weights for sampling negative examples
            Default is True
        nCyclesResample : every number of cycles (one round of OCL + MCL) followed 
            warm up to resample background examples for multi-class training.
            The resampling is done right after the completion of the OCL in each 
            corresponding cycle
        nMCLSteps : number of multi-class learning (MCL) steps followed 
            every one-class learning (OCL) phase, when OCL and MCL are 
            scheduled on in an alternating fashion, i.e., with enableMCL 
            setting to True.
            Default is 200
        nOCLSteps : number of OCL steps after warm up before starting MCL 
            steps when enableMCL is set to True
            Default is 200
        centerEps : float, optional
            Default is 0.1
            The lower bound of obsolute values in the center during its initialization
            This is to avoid trivial solution
        reprLoss : str, optional
            The representation loss to use, can be either 'soft-boundary' or 'minimal-distance'
            Default is 'soft-boundary'
        saveDist : bool, optional
            Indicate whether to save the distance of training examples to center
            Default is True
        solver : str, optional
            The solver for solving R with center and network weights fixed. 
            Possible values are 'minimize_scalar' and 'lp'
            Default is 'minimize_scalar'. 
        scalarMethod : str, optional
            The method to use when the scalar solver is used.
            Possible values are 'brent', 'bounded', and 'golden'
            Default is 'brent'
        lpobj : str, optional
            The objective to use when LP solver is used
            Possible values are 'primal' and 'dual'
            Default is 'primal'
        tol : float, optional
            Default is 0.001
            The tolerance used in line search in LP dual
        modeInValidate : str, optional
            The mode of the model in validation, can be either 'train' or 'eval'
            There can be major difference between these two when batch size is small.
            Default is 'eval'.
        """
        
        if enableMCL and bgDataSampler is None:
            raise ValueError('A sampler needs to be provided via parameter bgDataSampler '
                             'to include multi-class learning steps in the training')

        self._enableMCL = enableMCL
        self._enableCL = enableCL

        super(SemiSVSDTrainer, self).__init__(
            model = model,
            dataSampler = dataSampler,
            lossCalculator = lossCalculator,
            optimizerClass = optimizerClass,
            optimizerKwargs = optimizerKwargs,
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
            preloadValData = True,
            preloadTestData = True,
            metrics = metrics,
            useScheduler = useScheduler,
            deterministic = deterministic,
            valOfMisInTarget = None,
            classify = classify)

        self.classify = classify
        
        self._reprLoss = reprLoss
        self.model.setReprLoss(self._reprLoss)
        self._outputDir = outputDir

        if self._reprLoss == 'semiCon':
            self._bgBatchSize = int(self.batchSize // negRate)
        else:
            self._bgBatchSize = self.batchSize
            
        if self._enableMCL or self._enableCL or self._reprLoss == 'soft-boundary':
            # warm up phase needed when either MCL is enabled or using soft-boundary loss
            self._nStepsWarmup = nStepsWarmup
            
        if self._reprLoss == 'soft-boundary':
            # set up when using soft-boundary loss
            # initialization for computing R
            self.model.initForBndry(solver = solver, scalarMethod = scalarMethod, 
                              lpObj = lpObj, tol = tol) 
            # training hyper-parameters when using soft-boundary loss
            self._nStepsWPToIncr = nStepsWPToIncr
            self._nStepsUpdateR = nStepsUpdateR
            # for saving the distance of training examples to center 
            self._saveDist = saveDist
            if self._saveDist:
                self._distDir = outputDir + '/distance'
                os.makedirs(self._distDir, exist_ok = True)

        if self._enableMCL:
            # set up for MCL 
            self.model.setTrainAlg('semi-ocl') 
            # set up background data sampler 
            self._bgDataSampler = bgDataSampler
            nBgToSample = dict()
            nBgToSample['train'] = self.sampler.nTtlTrain
            nBgToSample['validate'] = nValidationSamples \
                if nValidationSamples is not None else len(self._validationData)*self.batchSize
                #if nValidationSamples is not None else self.sampler.nTtlValidate
            self._bgDataSampler.setNExamToSample(nBgToSample)
            
            # for saving sampling weights of background examples 
            self._saveSampWeights = saveSampWeights
            if self._saveSampWeights:
                self._sampWeightsDir = outputDir + '/samp-weight'
                os.makedirs(self._sampWeightsDir, exist_ok = True)
            
            # training hyper-parameters
            self._nCyclesResample = nCyclesResample
            self._nOCLSteps = nOCLSteps
            self._nMCLSteps = nMClSteps

        if self._enableCL or self.classify:
            # set up background data sampler
            self._bgDataSampler = bgDataSampler
            optimizerClass = torch.optim.AdamW
            #optimizerKwargs = {"lr": 0.0001, "weight_decay": 1e-6, "momentum": 0.9}
            optimizerKwargs = {"lr": 0.0001}
            if self.classify:
                self.model.setOptimizer_classify(optimizerClass, optimizerKwargs)
            else:
                self.model.setOptimizer(optimizerClass, optimizerKwargs)

        self._modeInValidate = modeInValidate
        
        if checkpointResume is not None:
            self._loadCheckpoint(checkpointResume)
        elif not self._enableCL or self.classify:
            # training from scratch
            if pretrainedNetwork is None:
                raise ValueError('A path to a saved pretrained network needs to provided '
                                 'via parameter "pretrainedNetwork" to train from scratch')
            if self.classify:
                self.model._model.load_state_dict(torch.load(pretrainedNetwork)['state_dict']['networkState'])
                for par in model._model.module.model.parameters():
                    par.requires_grad = False
                self.model._model.module.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(in_features=256, out_features=1, bias=True),
                    torch.nn.Sigmoid()).cuda()
            elif not self._enableCL:
                # load pretrained model
                self.model.initNetworkFromFile(pretrainedNetwork)
                # initialize center with both train and validation data
                trainData, _ = self.sampler.getDataAndTargets(self.batchSize, mode = 'train')
                allBatches = trainData + self._validationData
                self.model.initCenter(allBatches, eps = centerEps,
                                      saveCenterTo = '{0}/center.txt'.format(outputDir))
            else:
                # load pretrained model
                self.model.initNetworkFromFile(pretrainedNetwork)
                for par in model._model.module.model.classifier.parameters():
                    par.requires_grad = False
         
        if self._reprLoss == 'soft-boundary' and self._startStep < self._nStepsWarmup:
            # train model using 'minimal-distance' loss during beginning warm up steps
            self.model.setReprLoss('minimal-distance')
            
    def _initTrain(self):
        '''
        Initialization for training
        Overwrite the implementation in the base class
        '''
        header = False if self._enableMCL else True
        super(SemiSVSDTrainer, self)._initTrain(header = header)
        
        if self._enableMCL:
            self._timePerStepMcl = []
            self._trainLossMcl = LossTracker()
            self._trainLogger.info('\t'.join(['ocl_loss', 'mcl_loss']))
            
    def _initValidate(self, needMetric = True):
        '''
        Initialization for validation
        Overwrite the implementation in the base class
        '''
        if not self._enableMCL and not self._enableCL and not self.classify:
            needMetric = False
            header = True
        else:
            header = False
        super(SemiSVSDTrainer, self)._initValidate(
            needMetric = needMetric, header = header)
        if self._enableMCL:
            colNames = ['ocl_loss', 'mcl_loss']
            if needMetric:
                colNames += sorted([x for x in self._validationMetrics.metrics.keys()])
            self._validationLogger.info('\t'.join(colNames))
        
            # set the target of all validation example to positive facilitate 
            # validation by classification
            for batch in self._validationData:
                batch['targets'] = np.full(batch['targets'].shape, 1)
            self._allValidationTargets = np.full(self._allValidationTargets.shape, 1)
        elif self.classify:
            colNames = ['loss']
            if needMetric:
                colNames += sorted([x for x in self._validationMetrics.metrics.keys()])
            self._validationLogger.info('\t'.join(colNames))

            # set the target of all validation example to positive facilitate
            # validation by classification
            for batch in self._validationData:
                batch['targets'] = np.full(batch['targets'].shape, 1)
            self._allValidationTargets = np.full(self._allValidationTargets.shape, 1)
    
    def _prepareBgData(self):
        '''
        Prepare background data sampler and data for validation
        '''
        logger.info('[STEP {0}] Prepare background data sampler ...'.format(self.step))
        tStart = time()
        
        # Calculate the weights
        distFilePrefix = None
        if self._saveSampWeights:
            distFilePrefix = self._sampWeightsDir + '/step-{0}'.format(self.step)
        
        self._bgDataSampler.weightUpdatePrep()
        sampWeights = self.model.calcNegExamWeight(
            self._bgDataSampler, partitions = ['train', 'validate'], 
            batchSize = self.batchSize, saveDistTo = distFilePrefix)
        self._bgDataSampler.setSamplingWeight(sampWeights)
        
        tFinish = time()
        logger.info('[STEP {0}] Updating sampling weight has been completed, '
                    'time used: {1}s.'.format(self.step, tFinish - tStart))
        
        tStart = time()
        # load data for validation
        self._validationBgData = self._bgDataSampler.getValidationSet(self.batchSize)
        # set the target of all background examples to negative to facilitate validation
        for batch in self._validationBgData:
            batch['targets'] = np.full((batch['sequence'].shape[0], 1), 0)  
        self._allValidationBgTargets = np.full(self._allValidationTargets.shape, 0)
        
        tFinish = time()
        logger.info('[STEP {0}] Preloading background examples has been completed, '
                    'time used: {1}'.format(self.step, tFinish - tStart))
        
    def _updateRadius(self):
        '''
        Update radius
        '''
        assert(self._reprLoss == 'soft-boundary')
        
        logger.info('[STEP {0}] Update radius ...'.format(self.step))
        
        fileToSaveDist = None
        if self._saveDist:
            fileToSaveDist = '{0}/step-{1}.txt'.format(self._distDir, self.step)
            
        dataBatches, _ = self.sampler.getDataAndTargets(self.batchSize, mode = 'train') 
        try:
            self.model.updateRadius(dataBatches, saveDistTo = fileToSaveDist)
            logger.info('[STEP {0}] Updating radius has been completed'.format(self.step))
        except ValueError:
            self._nStepsWarmup += self._nStepsWPToIncr
            logger.info('[STEP {0}] Failed to update radius, increase '
                        'the number of warm up steps to {1}'.format(
                            self.step, self._nStepsWarmup))
    
    def trainAndValidate(self):
        """
        Trains the model and measures validation performance.
        """
        self.trainPreds = []
        self.trainTargets = []
        postWarmup = False
        if self._enableMCL:
            # precalculate the number of steps in one cycle to facilitate 
            # subsequent computation
            nStepsPerCycle = self._nOCLSteps + self._nMCLSteps
        self.step = 0
        self._validate(incClass=False)
        for step in range(self._startStep, self.maxNSteps):
            self.step = step
            
            if (self._enableMCL or self._reprLoss == 'soft-boundary') and \
                not postWarmup and step >= self._nStepsWarmup:
                
                # entering post warm up training
                if self._reprLoss == 'soft-boundary':
                    if step == self._nStepsWarmup:
                        # this is the first step after warm up, initialize R
                        self._updateRadius()
                    # the post warm up training may be delayed (if failure to update radius)
                    # so, check again, if still ready for post warm up training
                    if step == self._nStepsWarmup:
                        self.model.setReprLoss('soft-boundary')
                
                if self.step >= self._nStepsWarmup: 
                    postWarmup = True
                    if step == self._nStepsWarmup: 
                        logger.info('[STEP {0}] Entering post warm up training'.format(self.step))
                    else:
                        logger.info('[STEP {0}] Continue post warm up training from loaded ' 
                                    'checkpoint'.format(self.step))
            
            if self._reprLoss == 'soft-boundary' and self.step > self._nStepsWarmup and \
                (step - self._nStepsWarmup) % self._nStepsUpdateR == 0:
                # update R
                self._updateRadius()
            
            if self._enableMCL and step >= self._nStepsWarmup + self._nOCLSteps:
                iStepsInCycle = (self.step - self._nStepsWarmup) % nStepsPerCycle 
                iCycle = (self.step - self._nStepsWarmup) // nStepsPerCycle
                # the first condition in below is to consider resume from checkpoint or
                # the beginning of the first MCL where the bgSampler is not initialized yet
                # the second, resample in every self._nCyclesResample cycles following 
                # the warm up right after the completion of the OCL training phase
                if (iStepsInCycle >= self._nOCLSteps and 
                    self._bgDataSampler.getState() is not 'production') or \
                    (iStepsInCycle == self._nOCLSteps and iCycle % self._nCyclesResample == 0):
                    
                    # Prepare background data sampler before MCL starts
                    self._prepareBgData()
            
            if self._enableMCL and self.step >= self._nStepsWarmup:
                if (self.step - self._nStepsWarmup) % nStepsPerCycle == 0:
                    logger.info('[STEP {0}] Entering one-class training phase'.format(self.step))
                elif (self.step - self._nStepsWarmup) % nStepsPerCycle == self._nOCLSteps:
                    logger.info('[STEP {0}] Entering multi-class training phase'.format(self.step))

            if self._enableCL:
                self._train('contrastive')
            elif self.classify:
                self._train('classify')
            elif not self._enableMCL or self.step < self._nStepsWarmup or \
                (self.step - self._nStepsWarmup) % nStepsPerCycle < self._nOCLSteps:
                self._train('one-class')
            else:
                self._train('multi-class')
            
            # save check point    
            if self.step and (self.step % self.nStepsCheckpoint == 0 or
                      self.step == self.maxNSteps - 1): # save the last model
                self._checkpoint()
            
            # validate the model    
            if self.step and (self.step % self.nStepsStatReport == 0 or 
                      self.step == self.maxNSteps - 1): # validate the model from the last step
                self.trainPreds = []
                self.trainTargets = []
                if not self._enableMCL or not self._enableCL or\
                    self.step < self._nStepsWarmup + nStepsPerCycle:
                    self._validate(incClass = False) # validate without classification
                else:
                    self._validate(incClass = True) # validate with classification

        
        if self._validationMetrics is not None:
            # output validation metrics
            validationPerformance = os.path.join(
                self.outputDir, "validation-performance.txt")
            self._validationMetrics.writeValidationFeatureScores(
                validationPerformance)

        self.sampler.saveDatasetToFile("train", close_filehandle = True)

    
    def _mixBatch(self, posBatch, bgBatch):
        '''
        Create two data batches, in each which half examples from posBatch
        labeled as positive and the other half from bgBatch, labeled as
        negative
        '''
        posSeq = posBatch['sequence']
        bgSeq = bgBatch['sequence']
        nPos = posSeq.shape[0]
        nBg = bgSeq.shape[0]
        
        batches = []
        if np.min([nPos, nBg]) < 2 or np.max([nPos, nBg]) < self.batchSize / 2:
            # simply merge the two input batches
            batch = {}
            batch['sequence'] = np.vstack(posSeq, bgSeq)
            batch['targets'] = np.vstack(np.full((nPos, 1), 1, dtype = np.int8), 
                                         np.full((nBg, 1), 0, dtype = np.int8))
            batches.append(batch)
            return batches
        
        # break the two batches and cross link to create another two batches
        posHalf = nPos // 2
        bgHalf = nBg // 2
        
        # first batch
        batch = {}
        batch['sequence'] = np.vstack([posSeq[:posHalf], bgSeq[:bgHalf]])
        batch['targets'] = np.vstack((np.full((posHalf, 1), 1, dtype = np.int8), 
                                     np.full((bgHalf, 1), 0, dtype = np.int8)))
        batches.append(batch)
        
        # second batch
        batch = {}
        batch['sequence'] = np.vstack([posSeq[posHalf:], bgSeq[bgHalf:]])
        batch['targets'] = np.vstack((np.full((nPos - posHalf, 1), 1, dtype = np.int8), 
                                     np.full((nBg - bgHalf, 1), 0, dtype = np.int8)))
        batches.append(batch)
        
        return batches
    
    def _train(self, mode):
        """
        Trains the model on a batch of data.
        
        Parameters
        ----------
        mode : str
            Indicate whether the training is one-class or multi-class step.
            Can be either 'one-class' or 'multi-class'
        """
        
        assert mode in ['one-class', 'multi-class', 'contrastive', 'classify', 'warmup']
        
        tStart = time()
        
        batchData = self._getBatchData('train')
        if mode == 'one-class':
            # one-class step
            sumOfLoss, nEffTerms =  self.model.fit(batchData, mode = 'one-class')
            # track the loss
            self._trainLoss.add(sumOfLoss, nEffTerms)
            tFinish = time()
            self._timePerStep.append(tFinish - tStart)

        elif mode in ['classify', 'contrastive']:
            bgBatch = self._bgDataSampler.sample(batchSize = self._bgBatchSize, mode ='train')
            batch = {'sequence': np.vstack([batchData['sequence'], bgBatch['sequence']]),
                     'targets': np.vstack([np.full((batchData['sequence'].shape[0], 1), 1, dtype=np.int8),
                                         np.full((bgBatch['sequence'].shape[0], 1), 0, dtype=np.int8)])}
            if mode == 'contrastive':
                loss, nEffTerms =  self.model.fit(batch, mode = mode, reprLoss = self._reprLoss, n = self.batchSize)
            else:
                loss, nEffTerms, predictions = self.model.fit(batch, mode=mode, reprLoss=self._reprLoss, n=self.batchSize)
                self.trainPreds.extend(predictions)
                self.trainTargets.extend(torch.Tensor(batch['targets']))
            self._trainLoss.add(loss, nEffTerms)

        else:
            # multi-class step
            # obtain a batch of background data for using as negative examples
            bgBatch = self._bgDataSampler.sample(batchSize = self.batchSize,
                                                     mode = 'train', restart=True)

            sumOfLoss, nEffTerms = 0, 0
            for batch in self._mixBatch(batchData, bgBatch):
                sumOfLoss, nEffTerms = self.model.fit(batch, mode = 'multi-class')
                self._trainLossMcl.add(sumOfLoss, nEffTerms)
            tFinish = time()
            self._timePerStepMcl.append(tFinish - tStart)
        
        # report training status
        if self.step and (self.step % self.nStepsStatReport == 0 or 
                  self.step == self.maxNSteps - 1): # also report for the last step
            if not self._enableMCL:
                logger.info(("[STEP {0}] average number "
                             "of steps per second: {1:.1f}").format(
                    self.step, 1. / np.average(self._timePerStep)))
                logger.info("training loss: {0}".format(self._trainLoss.getAveLoss()))
                if not self.classify:
                    self._trainLogger.info(self._trainLoss.getAveLoss())
                else:
                    loss = self._trainLoss.getAveLoss()
                    trainScores = dict()
                    trainScores["loss"] = loss
                    to_log = [str(loss)]
                    trainScores.update(self._trainMetrics.update(
                        torch.Tensor(self.trainPreds).unsqueeze(1),
                        torch.Tensor(self.trainTargets).unsqueeze(1), step=self.step))

                    for name, score in trainScores.items():
                        logger.info("Train {0}: {1}".format(name, score))

                    for k in sorted(self._trainMetrics.metrics.keys()):
                        if k in trainScores:
                            to_log.append(str(trainScores[k]))
                        else:
                            to_log.append("NA")
                    self._trainLogger.info("\t".join(to_log))
            else:
                aveOcl = np.average(self._timePerStep)
                nOclStepPS = 'NA' if aveOcl == 0 else '{0: .3f}'.format(1. / aveOcl)
                aveMcl = np.average(self._timePerStepMcl)
                nMclStepPS = 'NA' if aveMcl == 0 else '{0: .3f}'.format(1. / aveMcl)
                logger.info(("[STEP {0}] average number "
                             "of OCL steps per second: {1}, that of MCL: {2}").format(
                    self.step, nOclStepPS, nMclStepPS))
                
                self._trainLogger.info('\t'.join([str(self._trainLoss.getAveLoss()), 
                                                  str(self._trainLossMcl.getAveLoss())]))
                logger.info("training OCL loss: {0}, and MCL loss: {1}".format(
                    self._trainLoss.getAveLoss(), self._trainLossMcl.getAveLoss()))
            
            # clearing    
            self._timePerStep = []
            self._trainLoss.reset()
            if self._enableMCL:
                self._timePerStepMcl = []
                self._trainLossMcl.reset()
                
        
    def _validate(self, incClass):
        """
        Measures model validation performance.
        
        Parameter
        ---------
        incClass : bool
            Whether to include classification performance in the validation
        """
        self.sampler.setMode("validate")
        
        validScores = dict()

        if self._enableCL:
            self._validationBgData = self._bgDataSampler.getValidationSet(self.batchSize)
            oclLoss, predictions, targets = self.model.validate((self._validationData, self._validationBgData[0]),
                                                                mode='contrastive', reprLoss=self._reprLoss,
                                                                n=self.batchSize, networkMode=self._modeInValidate,
                                                                step=self.step, outDir=self.outputDir)
            validScores["loss"] = oclLoss
            to_log = [str(oclLoss)]

        elif self.classify:
            self._validationBgData = self._bgDataSampler.getValidationSet(self.batchSize)
            oclLoss, predictions, targets = self.model.validate((self._validationData, self._validationBgData[0]),
                                                                mode='classify', reprLoss=self._reprLoss,
                                                                n=self.batchSize, networkMode=self._modeInValidate)
            validScores["loss"] = oclLoss
            to_log = [str(oclLoss)]
            validScores.update(self._validationMetrics.update(
                predictions, targets, step=self.step))

            for name, score in validScores.items():
                logger.info("validation {0}: {1}".format(name, score))

            for k in sorted(self._validationMetrics.metrics.keys()):
                if k in validScores:
                    to_log.append(str(validScores[k]))
                else:
                    to_log.append("NA")

        else:
            # validation based on representation loss
            oclLoss, _ = self.model.validate(self._validationData, mode = 'one-class',
                                             networkMode = self._modeInValidate)
            validScores["OCL loss"] = oclLoss
            to_log = [str(oclLoss)]
        
        if incClass:
            mclLoss, predictions = self.model.validate(
                self._validationData + self._validationBgData,
                mode = 'multi-class', networkMode = self._modeInValidate) 
            validScores["MCL loss"] = mclLoss
            allTargets = np.vstack([self._allValidationTargets, self._allValidationBgTargets])
            validScores.update(self._validationMetrics.update(
                predictions, allTargets, step=self.step))
            
            for name, score in validScores.items():
                logger.info("validation {0}: {1}".format(name, score))
        
        if self._enableMCL:
            mclLoss = str(mclLoss) if incClass else 'NA'           
            to_log.append(mclLoss)
            for k in sorted(self._validationMetrics.metrics.keys()):
                if k in validScores and validScores[k].any():
                    to_log.append(str(validScores[k]))
                else:
                    to_log.append("NA")
                        
        self._validationLogger.info("\t".join(to_log))

        # scheduler update
        if self._useScheduler:
            self.scheduler.step(
                math.ceil(oclLoss * 1000.0) / 1000.0)

        # save best_model
        if oclLoss < self._minLoss:
            self._minLoss = oclLoss
            self.best_model = self.model
            self._saveCheckpoint({
                "step": self.step,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.getStateDict(),
                "min_loss": self._minLoss,
                "optimizer": self.model.getOptimizer().state_dict()}, True)
            logger.debug("Updating `best_model.pth.tar`")

        if not self._enableCL and not self.classify:
            distDir = self._outputDir + '/distance'
            if not os.path.isdir(distDir): os.mkdir(distDir)
            np.savetxt(f'{distDir}/dist_{self.step}.txt', self.model.valDist)

            reprDir = self._outputDir + '/repr'
            if not os.path.isdir(reprDir): os.mkdir(reprDir)
            np.savetxt(f'{reprDir}/repr_{self.step}.txt', self.model.valRepr)
