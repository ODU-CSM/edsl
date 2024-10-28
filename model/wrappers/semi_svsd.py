'''
Implementation of model for one-class classification with the 
capability of training via semi-supervised learning

Created on Aug 12, 2021

@author: Javon
'''
import torch
from matplotlib import pyplot as plt
from torch import Tensor
import numpy as np
import pandas as pd
import logging
import os
from time import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from .pred import PredMWrapper
from ...train import LossTracker
from ..utils import loadModel


logger = logging.getLogger("fugep")

class SemiSVSDMWrapper(PredMWrapper):
    '''
    Implementation of model for one-class classification with the 
    capability of training via semi-supervised learning
    '''
    
    SOLVERS = ['minimize_scalar', 'lp']
    SCALAR_METHODS = ['brent', 'bounded', 'golden']
    LP_OBJS = ['primal', 'dual']
    TRAIN_ALGS = ['ocl', 'semi-ocl'] 
    
    def __init__(self, model, center : Tensor = None, radius = None, 
             mode = 'train', trainAlg = 'ocl', evalMode = 'one-class',
             evalType = 'validate', netModeForVal = 'eval',
             lossCalculator = None, radiusForPred = None, 
             useCuda = False, optimizerClass = None, optimizerKwargs = None,
             reprLoss = 'soft-boundary', solver = 'minimize_scalar',
             scalarMethod = 'brent', lpObj = 'primal', tol = 0.001):
        '''
        Constructor
        
        Parameters
        ----------
        center : Tensor, optional
            Center to use in the calculation of distance between data points and the center
            If not provided here, the center needs to be set by other means before training 
            or prediction can proceed. 
        radius : float, optional
            The initial radius used in training with soft-boundary representation loss, or 
            the radius used as default in making predictions  
        trainAlg : str, optional
            Training mode, can be either 'ocl' or 'semi-ocl'
            Default is 'ocl'
        evalMode : str, optional
            Evaluate mode, can be either 'one-class' or 'multi-class'. Note that models trained
            using 'ocl' cannot be evaluated in 'multi-class' mode. Models trained using 'semi-ocl' 
            mode have no problem to be evaluated in 'one-class' mode.
            Default is 'one-class'
        evalType : str, optional
            Can be either 'validate' or 'test'. If 'test', prediction (based on distance 
            and radius) is also made when evaluate mode is 'one-class'
        netModeForVal : str, optional
            Network mode to use in validation, can be either 'train' or 'eval'
            Default is 'eval'
        radiusForPred : float, optional
            The radius for making prediction one-class mode. If not provided, radius will 
            be used. Examples with distances from the center within the radius are predicted
            as positive, otherwise negative
        reprLoss : str, optional
            The representation loss to use, can be either 'soft-boundary' or 'minimal-distance'
            Default is 'soft-boundary'
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
        '''
        
        super(SemiSVSDMWrapper, self).__init__(model, 
            mode = mode, lossCalculator = lossCalculator,
            useCuda = useCuda, optimizerClass = optimizerClass,
            optimizerKwargs = optimizerKwargs)

        self._center = center
        self._reprLoss = reprLoss
        
        if self._center is not None:
            self._lossCalculator.setCenter(self._center)
        self._lossCalculator.setReprLoss(reprLoss)
        if reprLoss == 'soft-boundary':
            self.initForBndry(radius = radius, solver = solver, scalarMethod = scalarMethod, 
                              lpObj = lpObj, tol = tol)
        
        self._trainAlg = self.setTrainAlg(trainAlg)
        self._evalMode = evalMode
        self._evalType = evalType
        self._netModeForVal = netModeForVal

        self.radiusForPred = radiusForPred
        self._reprLoss = reprLoss
        self.valDist = None
        self.valRepr = None
        self._minLoss = torch.inf

    def initForBndry(self, radius = None, solver = 'minimize_scalar',  
             scalarMethod = 'brent', lpObj = 'primal', tol = 0.001):
        '''
        Initialize the class members for training
        
        Parameters
        -----------
        radius : float, optional
            The initial radius used in training with soft-boundary representation loss, or 
            the radius used as default in making predictions  
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
        '''
        
        self._radius = radius
        if radius is not None:
            self._lossCalculator.setRadius(radius)
        
        # set solver settings for radius
        if solver not in self.SOLVERS:
            raise ValueError('Parameter solver should be one of [{0}], '
                             'but get a {1}'.format(self.SOLVERS.join(', '), solver))
        self._solver = solver
        if solver == 'minimize_scalar':
            # set scalar method
            if scalarMethod not in self.SCALAR_METHODS:
                raise ValueError('Parameter scalarMethod should be one of [{0}], '
                                 'but get a {1}'.format(
                                     self.SCALAR_METHODS.join(', '), scalarMethod))
            self._scalarMethod = scalarMethod 
        elif solver == 'lp':
            # set objective and tolerance
            if lpObj not in self.LP_OBJS:
                raise ValueError('Parameter lpObj should be one of [{0}], '
                                 'but get a {1}'.format(
                                     self.LP_OBJS.join(', '), lpObj))
            self._lpObj = lpObj 
            self._tol = tol
    
    def setReprLoss(self, reprLoss):
        '''
        Set representation loss to use for training
        '''
        self._lossCalculator.setReprLoss(reprLoss)
    
    def setTrainAlg(self, trainAlg):
        '''
        Set training algorithm
        '''
        if trainAlg not in self.TRAIN_ALGS:
            raise ValueError('The training algorithm needs to be one of [{0}], but the input is '
                             '{1}'.format(self.TRAIN_ALGS.join(', '), trainAlg))
        self._trainAlg = trainAlg
        
        
    def initCenter(self, dataBatches, eps = 0.1, saveCenterTo = None):
        '''
        Computer the center of the hidden representation of input data
        
        Parameters
        ----------
        dataBatches : A list of mini-batch of data
        saveCenterTo : Str, optional
            Default is None, center will not be saved
            The path to the file to save center
        '''
        
        logger.info('Initializing center ...')
        tStart = time()
        
        self._model.train() # use train mode, important!!!
        
        nSamps = 0
        center = torch.zeros(self.getReprDim())
        if self._useCuda:
            center = center.cuda()

        with torch.no_grad():
            for batchData in dataBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
                repr = self._model(inputs.transpose(1, 2), mode = 'repr')
                nSamps += repr.shape[0]
                center += torch.sum(repr, dim = 0)

            center /= nSamps

        # If center_i is too close to 0, set to +-eps. 
        # Reason: a zero unit can be trivially matched with zero weights.
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps
        
        # install center
        self._center = center
        if self._lossCalculator is not None:
            self._lossCalculator.setCenter(center)
            
        if saveCenterTo is not None:
            # save center
            centToSave = self._center
            if self._useCuda:
                centToSave = centToSave.cpu()
            centSer = pd.Series(centToSave, copy = False)
            centSer.to_csv(saveCenterTo, header = False, index = False)
        
        tFinish = time()
        logger.info('{0}s to initialize center.'.format(tFinish - tStart))
        
    def _findRByMS(self, distSqr):
        '''
        Update R by scipy.optimize.minimize_scalar
        '''
        from scipy.optimize import minimize_scalar
    
        n = len(distSqr)
        gamma = self._lossCalculator.getGamma()
        # define deep SVDD objective function in R
        def f(x):
            return (x**2 + (1 / (gamma * n)) *
                    np.sum(np.max(np.column_stack((np.zeros(n), distSqr - x**2)), axis = 1), 
                           dtype = np.float32))

        # get lower and upper bounds around the (1-nu)-th quantile of distances
        bracket = None
        bounds = None

        upper_idx = int(np.max((np.floor(n * gamma * 0.1), 1)))
        lower_idx = int(np.min((np.floor(n * gamma * 1.1), n)))
        sort_idx = distSqr.argsort()
        upper = distSqr[sort_idx][-upper_idx]
        lower = distSqr[sort_idx][-lower_idx]

        if self._scalarMethod in ("brent", "golden"):
            bracket = (lower, upper)
        elif self._scalarMethod == "bounded":
            bounds = (lower, upper)

        # solve for R
        res = minimize_scalar(f, bracket = bracket, bounds = bounds, 
                              method = self._scalarMethod)

        # Get new R, mimimize_scalar may return negative solution, its positive counterpart is equally good
        return np.absolute(res.x)
    
    def _findRByLP(self, distSqr):
        from cvxopt import matrix
        from cvxopt.solvers import lp

        n = len(distSqr)
        gamma = self._lossCalculator.getGamma()
        
        # Solve either primal or dual objective
        if self._lpObj == "primal":
            # Define LP
            c = matrix(np.append(np.ones(1), (1 / (gamma * n)) * np.ones(n), axis = 0).astype(np.double))
            G = matrix(- np.concatenate((np.concatenate((np.ones(n).reshape(n, 1), np.eye(n)), axis = 1),
                                         np.concatenate((np.zeros(n).reshape(n, 1), np.eye(n)), axis = 1)),
                                        axis = 0).astype(np.double))
            h = matrix(np.append(-distSqr, np.zeros(n), axis = 0).astype(np.double))

            # Solve LP
            sol = lp(c, G, h)['x']
            if (sol is None):
                # no feasible solution can be found
                return None
            
            # Get new R
            rSqr = np.array(sol).reshape(n + 1).astype(np.float32)[0]
        elif self._lpObj == "dual":
            # Define LP
            c = matrix(distSqr.astype(np.double))
            G = matrix((np.concatenate((np.eye(n), -np.eye(n)), axis=0)).astype(np.double))
            h = matrix((np.concatenate(((1/(gamma * n)) * np.ones(n), np.zeros(n)), axis=0)).astype(np.double))
            A = matrix((np.ones(n)).astype(np.double)).trans()
            b = matrix(1, tc='d')

            # Solve LP
            sol = lp(c, G, h, A, b)['x']
            if (sol is None):
                # no feasible solution can be found
                return None
            a = np.array(sol).reshape(n)

            # Recover R (using the specified numeric tolerance on the range)
            n_svs = 0  # number of support vectors
            tol = self._tol
            while n_svs == 0:
                lower = tol * (1/(gamma * n))
                upper = (1 - tol) * (1/(gamma * n))
                idx_svs = (a > lower) & (a < upper)
                n_svs = np.sum(idx_svs)
                tol /= 10  # decrease tolerance if there are still no support vectors found

            rSqr = np.median(np.array(c).reshape(n)[idx_svs]).astype(np.float32) 
            
        return np.sqrt(rSqr)   
        
    def updateRadius(self, dataBatches, saveDistTo = None):
        """
        Function to update R while leaving the network parameters and center
        fixed in a block coordinate optimization.
        Using scipy.optimize.minimize_scalar or linear programming of cvxopt.
        
        Adapted from function update_R() from DeepSVDD, ICML, 2018
        
        Parameters
        -----------
        saveDistTo : str, optional
            Path to file for saving distance of training examples to center
            Default is None
        """
        self._model.train() # use train mode, important!!!
        
        # obtain hidden representation of input data
        reprInBatch = []
        with torch.no_grad():
            for batchData in dataBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
                repr = self._model(inputs.transpose(1, 2), mode = 'repr')
                if self._useCuda:
                    repr = repr.cpu()
                reprInBatch.append(repr.numpy())
        repr = np.vstack(reprInBatch)
        
        # compute the square of distance
        center = self._center
        if self._useCuda:
            center = center.cpu()
        distSqr = np.sum((repr - center.numpy()) ** 2, axis = 1, dtype = np.float32)
        if saveDistTo is not None:
            # save calculated distance
            distSer = pd.Series(np.sqrt(distSqr), copy = False)
            distSer.to_csv(saveDistTo, header = False, index = False)
            
        # find optimal R
        if self._solver == "minimize_scalar":
            radius = self._findRByMS(distSqr)
        elif self._solver == "lp":
            radius = self._findRByLP(distSqr)
        
        if radius is None:
            if self._radius is not None:
                # keep the old radius
                logger.info('No feasible solution can be found. The old radius is retained '
                            'and will be used in subsequent optimization.')
                return
            else:
                raise ValueError('No feasible solution can be found')
        # update R
        logger.info('The found optimal radius is {0}'.format(radius))
        self._radius = radius
        if self._lossCalculator is not None:
            self._lossCalculator.setRadius(radius)
    
    def calcNegExamWeight(self, sampler, partitions = ['train', 'validate', 'test'], 
                          batchSize = 64, saveDistTo = None):
        '''
        Compute sampling weights for negative examples for multi-class phase training. 
        Considering the number of negative examples can be in very large amount, 
        this function takes a sampler to load data on the fly. 
        It is assumed the batch data returned by the sampler contains example id
        
        Parameters
        ----------
        saveDistTo : the path prefix to save calculated distance for diagnosis purpose
        
        Return
        ----------
        A dictionary of two-element tuples (i.e., (ids, weights)). 
        The first element is ID of the examples, and the second
        is the calculated weights. A tuple is created for each element in the input 
        parameter partitions. 
        '''
        weights = {}
        self._model.train() # use train mode, important!!!
        for pt in partitions:
            # obtain hidden representation and calculate the distance
            idsInBatch = []
            distInBatch = []
            batchData = sampler.sample(batchSize = batchSize, mode = pt)
            while batchData is not None:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda() 
                with torch.no_grad():
                    repr = self._model(inputs.transpose(1, 2), mode = 'repr')
                    distSqr = torch.sum((repr - self._center) ** 2, axis = 1, dtype = torch.float32)
                    if self._useCuda:
                        distSqr = distSqr.cpu()
                    distInBatch.append(np.sqrt(distSqr))
                idsInBatch.append(batchData['id'])
                batchData = sampler.sample(batchSize = batchSize, mode = pt)    
            dist = np.concatenate(distInBatch)
            ids = np.concatenate(idsInBatch)
            
            # calculate the weight based on distance
            wt = dist / np.sum(dist)
            weights[pt] = (ids, wt) # return a list of tuple
            
            if saveDistTo is not None:
                # save calculated distance
                distSer = pd.DataFrame({'id': ids, 'distance': dist, 'weight': wt}, copy = False)
                distSer.to_csv('{0}.{1}.txt'.format(saveDistTo, pt), header = True, index = False)
        return weights
    
    def _validateMode(self, mode):
        '''
        Validate training or validation mode
        '''
        if mode not in self._lossCalculator.MODES:
            raise ValueError('The mode needs to be one of [{0}], but the input is '
                             '{1}'.format(self._lossCalculator.MODES.join(', '), mode))
    
    def fit(self, batchData, mode = 'one-class', reprLoss = None, n = None):
        """
        Fit the model with a batch of data

        Parameters
        ----------
        batchData : dict
            A dictionary that holds the data for training
        mode : str
            Can either be 'one-class' or 'multi-class' to indicate whether a
            one-class or multi-class training step.
            Default is 'one-class' 

        Returns
        -------
        float : sum
            The sum of the loss over the batch of the data
        int : nTerms
            The number of terms involved in calculated loss. 
            
        Note
        ------
        The current implementation of this function is one step of gradient update.
        Future implementation can include a nStep argument to support multiple
        steps update or even train to the end (until no improvement over 
        the input batch of data)
        """
        if mode not in ['classify', 'contrastive', 'multi-class', 'one-class', 'warmup']:
            self._validateMode(mode)
        if self._trainAlg == 'ocl' and not (mode == 'one-class'):
            raise ValueError('Only "one-class" training step is allowed when using ocl '
                             'algorithm, but {0} is encountered'.format(mode))
        
        self._model.train()
        inputs = torch.Tensor(batchData['sequence'])
        if self._useCuda:
            inputs = inputs.cuda()
        
        # forward propagation and compute the loss
        if mode == 'multi-class':
            # a multi-class step
            targets = torch.Tensor(batchData['targets'])
            if self._useCuda:
                targets = targets.cuda()
    
            predictions = self._model(inputs.transpose(1, 2), mode = 'pred')
            lossToOptim, sumOfLoss, nEffTerms = \
                self._lossCalculator(prediction = predictions, target = targets,
                                     mode = 'multi-class')
        elif mode == 'one-class':
            # a one-class step
            repr = self._model(inputs.transpose(1, 2), mode = 'repr')
            lossToOptim, sumOfLoss, nEffTerms = \
                self._lossCalculator(repr = repr, mode = 'one-class')
        else:
            targets = torch.Tensor(batchData['targets'] )
            cl_targets = torch.Tensor(batchData['targets'][n//2:3*n//2])
            if self._useCuda:
                targets = targets.cuda()
                cl_targets = cl_targets.cuda()
            if mode == 'classify':
                predictions = self._model(inputs.transpose(1, 2), mode = 'pred')
                loss = self._lossCalculator(predictions, targets, mode='class')
                print(loss)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                return loss * torch.numel(targets), torch.numel(targets), predictions
            else:
                repr = self._model(inputs.transpose(1, 2), mode='repr')
                if reprLoss == 'semiCon':
                    poss = repr[:n]
                    negs = repr[n:]
                    nearNegs = []
                    distSqr = torch.zeros(negs.shape[0]).cuda()
                    for pos in poss:
                        distSqr += torch.sum((negs - pos) ** 2, dim=1)
                    idx = torch.topk(distSqr, negs.shape[0] - poss.shape[0]).indices
                    mask = torch.ones(negs.size(0), dtype=torch.bool)
                    mask[idx] = False
                    realNegs = negs[mask]
                    #unif = torch.ones(realNegs.shape[0])
                    #idx = unif.multinomial(num_samples=n, replacement=False)
                    #chosenNegs = realNegs[idx]
                    view1 = torch.cat([poss[:n//2], realNegs[:n//2]])
                    view2 = torch.cat([poss[n//2:], realNegs[n//2:]])
                    features = torch.cat((view1.unsqueeze(1), view2.unsqueeze(1)), dim=1)
                else:
                    view1 = torch.cat([repr[:n//2], repr[n:3*n//2]])
                    view2 = torch.cat([repr[n//2:n], repr[3*n//2:]])
                    features = torch.cat((view1.unsqueeze(1), view2.unsqueeze(1)), dim=1)
                loss = self._lossCalculator(features, cl_targets, mode='supCon2')
            print(loss)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            return loss * torch.numel(targets), torch.numel(targets)

        # backprop to update the network weights    
        self._optimizer.zero_grad()
        lossToOptim.backward()
        self._optimizer.step()
        return (sumOfLoss, nEffTerms, preds)
    
    def validate(self, dataInBatches, mode = None, reprLoss=None, n=None, networkMode = None, evalType='validate',
                 sampler=None, step=None, outDir=None):
        """
        Validate the model with a batch of data

        Parameters
        ----------
        dataInBatches : []
            A list of dictionaries that hold data in batches for the validating
        
        Returns
        -------
        float : 
            The average loss over the batch of the data
        nArray :
            The prediction, None when the mode is 'one-class'
        """
        
        if mode is None:
            mode = self._evalMode
        
        if networkMode is None:
            networkMode = self._netModeForVal

        if mode not in ['classify', 'contrastive', 'one-class']:
            self._validateMode(mode)
        assert(networkMode in ['train', 'eval'])
        
        if networkMode == 'eval':
            self._model.eval()
        else:
            self._model.train()

        batchLosses = LossTracker()
        if mode == 'multi-class':
            allPreds = []
            for batchData in dataInBatches:
                inputs = torch.Tensor(batchData['sequence'])
                targets = torch.Tensor(batchData['targets'])
                if self._useCuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
    
                with torch.no_grad():
                    predictions = self._model(inputs.transpose(1, 2), mode = 'pred')
                    _, sumOfLoss, nEffTerms =\
                        self._lossCalculator(prediction = predictions, target = targets,
                                             mode = 'multi-class')
    
                    allPreds.append(predictions.data.cpu().numpy())
                    batchLosses.add(sumOfLoss, nEffTerms)
            loss = batchLosses.getAveLoss()
            allPreds = np.vstack(allPreds)
        elif mode == 'one-class':
            if self._evalType == 'test':
                # prediction of class labels will be made based on distance and radius
                allPreds = []
                if self.radiusForPred is None:
                    self._setRadiusForPred()
                radiusSqr = self.radiusForPred ** 2
            else:
                allPreds = None

            self.testDistNeg = np.array([])
            self.testDistPos = np.array([])
            self.testReprNeg = np.array([])
            self.testReprPos = np.array([])
            reprInBatch = []
            for batchData in dataInBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
    
                with torch.no_grad():
                    repr = self._model(inputs.transpose(1, 2), mode = 'repr')
                    reprDist = torch.sum(repr, dim=1).data.cpu().numpy().reshape((repr.shape[0], 1))
                    self.testReprNeg = np.append(reprDist[np.where(batchData['targets'] == 0)],
                                                 self.testReprNeg)
                    self.testReprPos = np.append(reprDist[np.where(batchData['targets'] == 1)],
                                                 self.testReprPos)
                    _, sumOfLoss, nEffTerms =\
                        self._lossCalculator(repr = repr, mode = 'one-class')
                    batchLosses.add(sumOfLoss, nEffTerms)
                    
                    if self._evalType == 'test':
                        # compute distance to center
                        distSqr = torch.sum((repr - self._center)** 2, dim=1)
                        distSqr = distSqr.data.cpu().numpy()
                        distSqr = distSqr.reshape((distSqr.shape[0], 1))
                        self.testDistNeg = np.append(np.sqrt(distSqr[np.where(
                            batchData['targets'] == 0)]), self.testDistNeg)
                        self.testDistPos = np.append(np.sqrt(distSqr[np.where(
                            batchData['targets'] == 1)]), self.testDistPos)
                        # make prediction
                        predictions = np.full(distSqr.shape, 1)
                        predictions[distSqr > radiusSqr] = 0
                        allPreds.append(predictions)

                    if self._useCuda:
                        repr = repr.cpu()
                    reprInBatch.append(repr.numpy())
            repr = np.vstack(reprInBatch)

            center = self._center
            if self._useCuda:
                center = center.cpu()
            distSqr = np.sum((repr - center.numpy()) ** 2, axis = 1, dtype = np.float32)
            if not self._evalType == 'test':
                # calculated distance
                self.valDist = np.sqrt(distSqr)
                self.valRepr = repr

            loss = self._lossCalculator.calcLossFromSum(batchLosses.getSumOfLoss(),
                                                        batchLosses.getNItems())
            if self._evalType == 'test':
                allPreds = np.vstack(allPreds)

        else:
            pdist = torch.nn.PairwiseDistance(p=2)
            allPreds = []
            allTargets = []
            if mode == 'classify' and evalType == 'test':
                for batchData in dataInBatches:
                    inputs = torch.Tensor(batchData['sequence'])
                    targets = torch.Tensor(batchData['targets'])
                    if self._useCuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()

                    with torch.no_grad():
                        predictions = self._model(inputs.transpose(1, 2), mode='pred')
                    loss = self._lossCalculator(predictions, targets, mode=reprLoss)
                    allPreds.append(predictions.cpu().detach().numpy())
                    print(loss)
                    batchLosses.add(loss * torch.numel(targets), torch.numel(targets))
                    allTargets.append(targets.cpu())
            else:
                for batchData, bgBatch in zip(*dataInBatches):
                    batch = {'sequence': np.vstack([batchData['sequence'], bgBatch['sequence']]),
                             'targets': np.vstack([np.full((batchData['sequence'].shape[0], 1), 1, dtype=np.int8),
                                                   np.full((bgBatch['sequence'].shape[0], 1), 0, dtype=np.int8)])}

                    inputs = torch.Tensor(batch['sequence'])
                    targets = torch.Tensor(batch['targets'])
                    cl_targets = torch.Tensor(batch['targets'][n // 2:3 * n // 2])
                    if self._useCuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        cl_targets = cl_targets.cuda()
                    if mode == 'classify':
                        with torch.no_grad():
                            predictions = self._model(inputs.transpose(1, 2), mode='pred')
                        loss = self._lossCalculator(predictions, targets, mode=reprLoss)
                        allPreds.append(predictions.cpu().detach().numpy())
                    else:
                        with torch.no_grad():
                            repr = self._model(inputs.transpose(1, 2), mode='repr')

                        view1 = torch.cat([repr[:n // 2], repr[n:3 * n // 2]])
                        view2 = torch.cat([repr[n // 2:n], repr[3 * n // 2:]])
                        features = torch.cat((view1.unsqueeze(1), view2.unsqueeze(1)), dim=1)
                        print('Mean pairwise between positives: ', torch.norm(repr[:n].unsqueeze(1) - repr[:n], p=2, dim=2).sum()/(2*n*(2*n-1)))
                        print('Mean pairwise between negatives: ', torch.norm(repr[n:].unsqueeze(1) - repr[n:], p=2, dim=2).sum()/(2*n*(2*n-1)))
                        print('Mean pairwise between positives and negatives: ', torch.norm(repr[:n].unsqueeze(1) - repr[n:], p=2, dim=2).mean())
                        loss = self._lossCalculator(features, cl_targets, mode='supCon2')
                        allPreds.append(np.concatenate([features[:, 0, :].cpu(), features[:, 1, :].cpu()]))
                    print(loss)
                    batchLosses.add(loss * torch.numel(targets), torch.numel(targets))
                    allTargets.append(targets.cpu())

            loss = batchLosses.getAveLoss()
            allPreds = np.vstack(allPreds)
            allTargets = np.vstack(allTargets)

            if mode is 'contrastive':
                labels = np.array(allTargets.squeeze())
                features = allPreds
                indices_0 = np.where(labels == 0)[0]
                indices_1 = np.where(labels == 1)[0]
                tsne = TSNE(2, verbose=1)
                ump = umap.UMAP(n_components=2)
                pca = PCA(n_components=2)
                drs = {'umap': ump, 'pca': pca}
                fontsize = 20
                n = features.shape[0] / 2
                for dr_name, dr in drs.items():
                    proj = dr.fit_transform(features)
                    proj_0 = np.take(proj, indices_0, axis=0)
                    proj_1 = np.take(proj, indices_1, axis=0)
                    print(f'Mean pairwise between positives {dr_name}: ',
                          torch.norm(torch.asarray(proj_1).unsqueeze(1) - torch.asarray(proj_1), p=2, dim=2).sum() / (2 * n * (2 * n - 1)))
                    print(f'Mean pairwise between negatives {dr_name}: ',
                          torch.norm(torch.asarray(proj_0).unsqueeze(1) - torch.asarray(proj_0), p=2, dim=2).sum() / (2 * n * (2 * n - 1)))
                    print(f'Mean pairwise between positives and negatives {dr_name}: ',
                          torch.norm(torch.asarray(proj_1).unsqueeze(1) - torch.asarray(proj_0), p=2, dim=2).mean())
                    plt.figure(figsize=(32, 8))
                    plt.subplot(1, 3, 1)
                    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, alpha=0.4)
                    plt.legend(handles=scatter.legend_elements()[0], labels=['Negative', 'Positive'], loc='upper right',
                               fontsize=fontsize)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.xticks(fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    plt.subplot(1, 3, 2)
                    plt.scatter(proj_0[:, 0], proj_0[:, 1], c='#440154ff', alpha=0.4)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().spines['left'].set_visible(False)
                    plt.xticks(fontsize=fontsize)
                    plt.yticks([])
                    plt.subplot(1, 3, 3)
                    plt.scatter(proj_1[:, 0], proj_1[:, 1], c='#fde725FF', alpha=0.4)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().spines['left'].set_visible(False)
                    plt.xticks(fontsize=fontsize)
                    plt.yticks([])
                    visDir = outDir + '/vis_' + dr_name
                    if not os.path.isdir(visDir): os.mkdir(visDir)
                    plt.savefig(f'{visDir}/{step}.png')
                
        return loss, allPreds, allTargets

    def predict(self, dataInBatches, mode = None):
        """
        Apply the model to make prediction for a batch of data

        Parameters
        ----------
        batchData : []
            A list of dictionaries that hold data in batches for the validating

        Returns
        -------
        nArray :
            The prediction
        """
        if mode is None:
            mode = self._evalMode
        
        self._validateMode(mode)
        
        self._model.eval()
        
        allPreds = []
        if mode == 'multi-class':
            for batchData in dataInBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    predictions = self._model(inputs.transpose(1, 2), mode = 'pred')
                    allPreds.append(predictions.data.cpu().numpy())
        elif mode == 'one-class':
            if self.radiusForPred is None:
                self._setRadiusForPred()
            radiusSqr = self.radiusForPred ** 2
                    
            for batchData in dataInBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    repr = self._model(inputs.transpose(1, 2), mode = 'repr')
                    # compute distance to center
                    distSqr = torch.sum((repr - self._center) ** 2, dim = 1)
                    distSqr = distSqr.data.cpu().numpy()
                    distSqr = distSqr.reshape((distSqr.shape[0], 1))
                    predictions = np.full(distSqr.shape, 1)
                    predictions[distSqr > radiusSqr] = 0
                    allPreds.append(predictions)
        
        allPreds = np.vstack(allPreds)
        return allPreds
    
    def getStateDict(self):
        '''
        Return model state dictionary
        '''
        stateDict = {}
        stateDict['trainAlg'] = self._trainAlg
        if self._radius is not None:
            stateDict['radius'] = self._radius
        if self.radiusForPred is not None:
            stateDict['radiusForPred'] = self.radiusForPred
        if self._center is not None:
            stateDict['center'] = self._center
        if self.valDist is not None:
            stateDict['valDist'] = self.valDist
        stateDict['networkState'] = self._model.state_dict()    
        return stateDict
    
    def init(self, stateDict = None):
        """
        Initialize the model before training or making prediction
        """
        if stateDict is not None:
            if 'trainAlg' in stateDict.keys():
                self._trainAlg = stateDict['trainAlg']
                if self._trainAlg == 'ocl' and self._evalMode == 'multi-class':
                    raise ValueError('OCL trained model cannot be evaluated using'
                                     ' "multi-class" mode')
            if 'radius' in stateDict.keys():
                self._radius = stateDict['radius']
                # install the radius to loss function
                if self._lossCalculator is not None:
                    self._lossCalculator.setRadius(self._radius)
            if self.radiusForPred is None and \
                'radiusForPred' in stateDict.keys():
                self.radiusForPred = stateDict['radiusForPred']
            # center of training examples    
            if 'center' in stateDict.keys():
                self._center = stateDict['center']
                if self._useCuda:
                    self._center = self._center.cuda()
                # set center to loss calculator
                if self._lossCalculator is not None:
                    self._lossCalculator.setCenter(self._center)
            if 'valDist' in stateDict.keys():
                self.valDist = stateDict['valDist']

            #self._model = loadModel(stateDict['networkState'], self._model)
    
    def initFromFile(self, filepath):
        '''
        Initialize the model by a previously trained model saved 
        to a file
        '''
        stateDict = torch.load(filepath,
            map_location=lambda storage, location: storage)
        self.init(stateDict)
    
    def initNetworkFromFile(self, filepath):
        '''
        Initialize network using a checkpoint that includes the state (stat_dict)
        of a pretrained model 
        '''
        pretrainedState = torch.load(filepath,
            map_location=lambda storage, location: storage)
        pretrainedState = pretrainedState['state_dict']
        currState = self._model.state_dict()
        
        # Note that if the current network has the same size classification net 
        # as the pretrained network, the classification weights are also initialized
        # by pretrained. If this is not desired, following code needs to be modified
        pretrainedState = { k:v for k,v in pretrainedState.items() \
                            if k in currState and v.size() == currState[k].size() }
        currState.update(pretrainedState)
        self._model = loadModel(currState, self._model)
        
    def save(self, outputDir, modelName = 'model'):
        """
        Save the model
        
        Parameters:
        --------------
        outputDir : str
            The path to the directory where to save the model
        """
        outputPath = os.path.join(outputDir, modelName)
        torch.save(self.getStateDict(), 
                   "{0}.pth.tar".format(outputPath))

    def _setRadiusForPred(self):
        if self._reprLoss == 'soft-boundary':
            if self._radius is not None:
                logger.info('The radius for prediction has not been set, radius: {0} for '
                            'loss calculation will be used'.format(self._radius))
                self.radiusForPred = self._radius
            else:
                raise ValueError('The radius for prediction needs to be set before prediction can be made')
        else:
            self.radiusForPred = self._calcRadius(self.valDist)
            logger.info('The radius for prediction has not been set, radius: {0} obtained '
                        'from validation of best model will be used'.format(self.radiusForPred))

    def _calcRadius(self, dist):
        std = np.std(dist)
        cutOff = std * 2
        mean = np.mean(dist)
        upper = mean + cutOff
        dist_new = dist[np.where(dist < upper)]
        radius = max(dist_new)
        '''
        import matplotlib
        matplotlib.use('TkAgg')
        plt.hist(dist, density=False, bins=50)
        plt.hist(dist_new, density=False, bins=50)
        plt.savefig('hist.png')
        '''
        return radius

