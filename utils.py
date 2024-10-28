"""
This module provides utility functions that are not tied to specific
classes or concepts, but still perform specific and important roles
across many of the packages modules.

"""
import logging
import sys
import os

import numpy as np

def get_indices_and_probabilities(interval_lengths, indices):
    """
    Given a list of different interval lengths and the indices of
    interest in that list, weight the probability that we will sample
    one of the indices in `indices` based on the interval lengths in
    that sublist.

    Parameters
    ----------
    interval_lengths : list(int)
        The list of lengths of intervals that we will draw from. This is
        used to weight the indices proportionally to interval length.
    indices : list(int)
        The list of interval length indices to draw from.

    Returns
    -------
    indices, weights : tuple(list(int), list(float)) \
        Tuple of interval indices to sample from and the corresponding
        weights of those intervals.

    """
    select_interval_lens = np.array(interval_lengths)[indices]
    weights = select_interval_lens / float(np.sum(select_interval_lens))

    keep_indices = []
    for index, weight in enumerate(weights):
        if weight > 1e-10:
            keep_indices.append(indices[index])
    if len(keep_indices) == len(indices):
        return indices, weights.tolist()
    else:
        return get_indices_and_probabilities(
            interval_lengths, keep_indices)




def load_features_list(input_path):
    """
    Reads in a file of distinct feature names line-by-line and returns
    these features as a list. Each feature name in the file must occur
    on a separate line.

    Parameters
    ----------
    input_path : str
        Path to the features file. Each feature in the input file must
        be on its own line.

    Returns
    -------
    list(str) \
        The list of features. The features will appear in the list in
        the same order they appeared in the file (reading from top to
        bottom).

    Examples
    --------
    A file at "input_features.txt", for the feature names :math:`YFP`
    and :math:`YFG` might look like this:
    ::
        YFP
        YFG


    We can load these features from that file as follows:

    >>> load_features_list("input_features.txt")
    ["YFP", "YFG"]

    """
    features = []
    with open(input_path, 'r') as file_handle:
        for line in file_handle:
            features.append(line.strip())
    return features


def initialize_logger(output_path, name = 'fugep', verbosity = 2):
    """
    Initializes the logger for FuGEP.
    This function can only be called successfully once.
    If the logger has already been initialized with handlers,
    the function exits. Otherwise, it proceeds to set the
    logger configurations.

    Parameters
    ----------
    output_path : str
        The path to the output file where logs will be written.
    name : str
        The name of the logger
    verbosity : int, {2, 1, 0}
        Default is 2. The level of logging verbosity to use.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    """
    logger = logging.getLogger(name)
    # check if logger has already been initialized
    if len(logger.handlers):
        return

    if verbosity == 0:
        logger.setLevel(logging.WARN)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(output_path)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)

    stdout_formatter = logging.Formatter(
        "%(asctime)s - %(message)s")

    stdout_handle = logging.StreamHandler(sys.stdout)
    stdout_handle.setFormatter(stdout_formatter)
    stdout_handle.setLevel(logging.INFO)
    logger.addHandler(stdout_handle)


def make_dir(dirname):
    """Create directory `dirname` if non-existing.

    Parameters
    ----------
    dirname: str
        Path of directory to be created.

    Returns
    -------
    bool
        `True`, if directory did not exist and was created.
    """
    if os.path.exists(dirname):
        return False
    else:
        os.makedirs(dirname)
        return True