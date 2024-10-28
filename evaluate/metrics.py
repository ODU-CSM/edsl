"""
This module provides the `PerformanceMetrics` class and supporting
functionality for tracking and computing model performance.
"""
from collections import defaultdict, namedtuple
import logging
import os

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

logger = logging.getLogger("fugep")


Metric = namedtuple("Metric", ["fn", "data"])
"""
A tuple containing a metric function and the results from applying that
metric to some values.

Parameters
----------
fn : types.FunctionType
    A metric.
data : list(float)
    A list holding the results from applying the metric.

Attributes
----------
fn : types.FunctionType
    A metric.
data : list(float)
    A list holding the results from applying the metric.

"""


def visualize_roc_curves(prediction,
                         target,
                         output_dir,
                         nMinMinors=50,
                         style="seaborn-colorblind",
                         fig_title="Feature ROC curves",
                         dpi=500,
                         valOfMisInTarget = None):
    """
    Output the ROC curves for each feature predicted by a model
    as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    nMinMinors : int, optional
        Default is 50. Do not visualize an ROC curve for a feature with
        less than 50 examples of the minor class in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature ROC curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.
    valOfMisInTarget : the value representing the missing in target

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        
        if valOfMisInTarget is not None:
            feature_preds = feature_preds[feature_targets != valOfMisInTarget]
            feature_targets = feature_targets[feature_targets != valOfMisInTarget]
        
        if len(feature_targets) == 0:
            continue
        
        nPos = np.count_nonzero(feature_targets)
        nMinors = min([nPos, len(feature_targets) - nPos])
        if nMinors >= nMinMinors:
            fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
            plt.plot(fpr, tpr, 'r-', alpha=0.3, lw=1, color='gray')
 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_title = False
    if fig_title:
        plt.title(fig_title)
    plt.savefig(os.path.join(output_dir, "roc_curves.pdf"),
                format="pdf",
                dpi=dpi)


def visualize_precision_recall_curves(
        prediction,
        target,
        output_dir,
        nMinMinors = 50,
        style="seaborn-colorblind",
        fig_title="Feature precision-recall curves",
        dpi=500,
        valOfMisInTarget = None):
    """
    Output the precision-recall (PR) curves for each feature predicted by
    a model as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    nMinMinors : int, optional
        Default is 50. Do not visualize an PR curve for a feature with
        less than 50 examples of the minor class in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature precision-recall curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.
    valOfMisInTarget : the value representing the missing in target

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """
    os.makedirs(output_dir, exist_ok=True)

    # TODO: fix this
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        
        if valOfMisInTarget is not None:
            feature_preds = feature_preds[feature_targets != valOfMisInTarget]
            feature_targets = feature_targets[feature_targets != valOfMisInTarget]
        
        if len(feature_targets) == 0:
            continue
        
        nPos = np.count_nonzero(feature_targets)
        nMinors = min([nPos, len(feature_targets) - nPos])
        if nMinors >= nMinMinors:
            precision, recall, _ = precision_recall_curve(
                feature_targets, feature_preds)
            plt.step(
                recall, precision, 'r-',
                alpha=0.3, lw=1, where="post", color='gray')
            
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_title = False
    if fig_title:
        plt.title(fig_title)
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.pdf"),
                format="pdf",
                dpi=dpi)


def compute_score(prediction, target, metric_fn, thresholds,
                  nMinMinors = 10,
                  valOfMisInTarget = None):
    """
    Using a user-specified metric, computes the distance between
    two tensors.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    metric_fn : types.FunctionType
        A metric that can measure the distance between the prediction
        and target variables.
    nMinMinors : int, optional
        Default is 10. The minimum number of examples of minor class for a
        feature in order to compute the score for it.

    Returns
    -------
    average_score, feature_scores : tuple(float, numpy.ndarray)
        A tuple containing the average of all feature scores, and a
        vector containing the scores for each feature. If there were
        no features meeting our filtering thresholds, will return
        `(None, [])`.
    """
    if not thresholds:
        thresholds = [0.5] * prediction.shape[1]

    feature_scores = [np.nan] * target.shape[1]
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        binaryFeaturePreds = np.where(feature_preds >= thresholds[index], 1, 0)
        if valOfMisInTarget is not None:
            binaryFeaturePreds = binaryFeaturePreds[feature_targets != valOfMisInTarget]
            feature_targets = feature_targets[feature_targets != valOfMisInTarget]
        
        if len(feature_targets) == 0:
            continue

        if metric_fn == confusion_matrix:
            feature_scores[index] = confusion_matrix(feature_targets, binaryFeaturePreds, labels=[0, 1]).ravel()
            continue
        
        nPos = np.count_nonzero(feature_targets)
        nMinors = min([nPos, len(feature_targets) - nPos])
        if nMinors >= nMinMinors:
            try:
                if metric_fn in [recall_score, accuracy_score]:
                    feature_scores[index] = metric_fn(
                        feature_targets, binaryFeaturePreds)
                elif metric_fn == f1_score:
                    feature_scores[index] = metric_fn(
                        feature_targets, binaryFeaturePreds, average=None)
                elif metric_fn == precision_recall_curve:
                    precision, recall, _ = metric_fn(
                        feature_targets, feature_preds)
                    feature_scores[index] = auc(recall, precision)
                else:
                    print(feature_targets)
                    print(feature_preds)
                    feature_scores[index] = metric_fn(
                        feature_targets, feature_preds)
            except ValueError:  
                continue

    valid_feature_scores = [s for s in feature_scores if not np.isnan(s).any()] # Allow 0 or negative values.
    if not valid_feature_scores:
        return None, feature_scores
    average_score = np.average(valid_feature_scores, axis=0)
    return average_score, feature_scores


def get_feature_specific_scores(data, get_feature_from_index_fn):
    """
    Generates a dictionary mapping feature names to feature scores from
    an intermediate representation.

    Parameters
    ----------
    data : list(tuple(int, float))
        A list of tuples, where each tuple contains a feature's index
        and the score for that feature.
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).

    Returns
    -------
    dict
        A dictionary mapping feature names (`str`) to scores (`float`).
        If there was no score for a feature, its score will be set to
        `None`.

    """
    feature_score_dict = {}
    for index, score in enumerate(data):
        feature = get_feature_from_index_fn(index)
        if type(score) is not list or not np.isnan(score):
            feature_score_dict[feature] = score
        else:
            feature_score_dict[feature] = None
    return feature_score_dict


def auc_u_test(labels, predictions):
    """
    Outputs the area under the the ROC curve associated with a certain 
    set of labels and the predictions given by the training model.
    Computed from the U statistic.

    Parameters
    ----------
    labels: numpy.ndarray
        Known labels of values predicted by model. Must be one dimensional.
    predictions: numpy.ndarray
        Value predicted by user model. Must be one dimensional, with matching
        dimension to `labels`

    Returns
    -------
    float
        AUC value of given label, prediction pairs  
   
    """
    len_pos = int(np.sum(labels))
    len_neg = len(labels) - len_pos
    rank_sum = np.sum(rankdata(predictions)[labels == 1])
    u_value = rank_sum - (len_pos * (len_pos + 1)) / 2
    auc = u_value / (len_pos * len_neg)
    return auc

class PerformanceMetrics(object):
    """
    Tracks and calculates metrics to evaluate how closely a model's
    predictions match the true values it was designed to predict.

    Parameters
    ----------
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions.
        By default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, and `"average_precision"`,
        which maps to `sklearn.metrics.average_precision_score`.



    Attributes
    ----------
    skip_threshold : int
        The minimum number of positive examples of a feature that must
        be included in an update for a metric score to be
        calculated for it.
    getFeatureByIndex : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    metrics : dict
        A dictionary that maps metric names (`str`) to metric objects
        (`Metric`). By default, this contains `"roc_auc"` and
        `"average_precision"`.

    """

    def __init__(self,
                 get_feature_from_index_fn,
                 nMinMinorsReport=10,
                 metrics=dict(roc_auc=roc_auc_score, pr_auc=precision_recall_curve, avg_precision=average_precision_score,
                              f1_negPos=f1_score, accuracy=accuracy_score, recall=recall_score,
                              confusion_TnFpFnTp=confusion_matrix),

                 valOfMisInTarget = None,
                 isValidate=False):
        """
        Creates a new object of the `PerformanceMetrics` class.
        """
        self.skip_threshold = nMinMinorsReport
        self.getFeatureByIndex = get_feature_from_index_fn
        self.metrics = dict()
        for k, v in metrics.items():
            self.metrics[k] = Metric(fn=v, data=[])
        self._valOfMisInTarget = valOfMisInTarget
        self._isValidate = isValidate
        if self._isValidate:
            self._featureScores = pd.DataFrame()

    def add_metric(self, name, metric_fn):
        """
        Begins tracking of the specified metric.

        Parameters
        ----------
        name : str
            The name of the metric.
        metric_fn : types.FunctionType
            A metric function.

        """
        self.metrics[name] = Metric(fn=metric_fn, data=[])

    def remove_metric(self, name):
        """
        Ends the tracking of the specified metric, and returns the
        previous scores associated with that metric.

        Parameters
        ----------
        name : str
            The name of the metric.

        Returns
        -------
        list(float)
            The list of feature-specific scores obtained by previous
            uses of the specified metric.

        """
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, prediction, target, thresholds=None, step=None):
        """
        Evaluates the tracked metrics on a model prediction and its
        target value, and adds this to the metric histories.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.

        Returns
        -------
        dict
            A dictionary mapping each metric names (`str`) to the
            average score of that metric across all features
            (`float`).

        """
        metric_scores = {}
        for name, metric in self.metrics.items():
            avg_score, feature_scores = compute_score(
                prediction, target, metric.fn, thresholds,
                nMinMinors = self.skip_threshold,
                valOfMisInTarget = self._valOfMisInTarget)
            metric.data.append(feature_scores)
            metric_scores[name] = avg_score

        if self._isValidate:
            for name, metric in self.metrics.items():
                feature_score_dict = get_feature_specific_scores(
                    metric.data[-1], self.getFeatureByIndex)
                addCols = {'step': step, 'metric': name}
                feature_score_dict.update(addCols)
                self._featureScores = self._featureScores.append(feature_score_dict, ignore_index=True)
        return metric_scores

    def visualize(self, prediction, target, output_dir, **kwargs):
        """
        Outputs ROC and PR curves. Does not support other metrics
        currently.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.
        output_dir : str
            The path to the directory to output the figures. Directories that
            do not currently exist will be automatically created.
        **kwargs : dict
            Keyword arguments to pass to each visualization function. Each
            function accepts the following args:

                * style : str - Default is "seaborn-colorblind". Specify a \
                          style available in \
                          `matplotlib.pyplot.style.available` to use.
                * dpi : int - Default is 500. Specify dots per inch \
                              (resolution) of the figure.

        Returns
        -------
        None
            Outputs figures to `output_dir`.

        """
        os.makedirs(output_dir, exist_ok=True)
        if "roc_auc" in self.metrics:
            visualize_roc_curves(
                prediction, target, output_dir,
                nMinMinors = self.skip_threshold,
                valOfMisInTarget = self._valOfMisInTarget,
                **kwargs)
        if "avg_precision" in self.metrics:
            visualize_precision_recall_curves(
                prediction, target, output_dir,
                valOfMisInTarget = self._valOfMisInTarget,
                nMinMinors = self.skip_threshold,
                **kwargs)

    def write_feature_scores_to_file(self, output_path):
        """
        Writes each metric's score for each feature to a specified
        file.

        Parameters
        ----------
        output_path : str
            The path to the output file where performance metrics will
            be written.

        Returns
        -------
        dict
            A dictionary mapping feature names (`str`) to
            sub-dictionaries (`dict`). Each sub-dictionary then maps
            metric names (`str`) to the score for that metric on the
            given feature. If a metric was not evaluated on a given
            feature, the score will be `None`.

        """
        feature_scores = defaultdict(dict)
        for name, metric in self.metrics.items():
            feature_score_dict = get_feature_specific_scores(
                metric.data[-1], self.getFeatureByIndex)
            for feature, score in feature_score_dict.items():
                if score is None:
                    feature_scores[feature] = None
                else:
                    feature_scores[feature][name] = score

        metric_cols = [m for m in self.metrics.keys()]
        cols = '\t'.join(["class"] + metric_cols)
        with open(output_path, 'w+') as file_handle:
            file_handle.write("{0}\n".format(cols))
            for feature, metric_scores in sorted(feature_scores.items()):
                if not metric_scores:
                    file_handle.write("{0}\t{1}\n".format(feature, "\t".join(["NA"] * len(metric_cols))))
                else:
                    metric_score_cols = '\t'.join(
                        ["{0}".format(s) for s in metric_scores.values()])
                    file_handle.write("{0}\t{1}\n".format(feature,
                                                          metric_score_cols))
        return feature_scores

    def writeValidationFeatureScores(self, output_path):
        """
        Writes each metric's score for each feature to a specified
        file.
        Parameters
        ----------
        output_path : str
            The path to the output file where performance metrics will
            be written.
        Returns
        -------
        data frame
        """
        cols = self._featureScores.columns.to_list()
        newCols = cols[-2:] + cols[:-2]
        self._featureScores = self._featureScores[newCols]
        self._featureScores = self._featureScores.replace(np.nan, 'NA', regex=True)
        self._featureScores.to_csv(output_path, index=False)

    def getThresh(self, prediction, target):
        thresholds = []
        feature_scores = [np.nan] * target.shape[1]
        for index, feature_preds in enumerate(prediction.T):
            feature_targets = target[:, index]
            precision, recall, thresh= precision_recall_curve(feature_targets, feature_preds)
            fscore = (2 * precision * recall) / (precision + recall)
            try:
                i = np.nanargmax(fscore[:-1])
            except:
                i = 0
            logger.info(f'Validation fscore: {fscore[i]}')
            th = round(thresh[i], ndigits=4)
            thresholds.append(th)
        return thresholds
