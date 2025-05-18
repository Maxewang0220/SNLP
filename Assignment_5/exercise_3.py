import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from typing import List


def preprocess(text: List[str]) -> List[str]:
    """ Preprocess the input text by removing extra spaces, characters and lowercasing.
    Args:
        text: The input text to preprocess.

    Returns:
        list: A list of tokens (words) after preprocessing.
    """
    raise NotImplementedError


def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    Args:
        text: input corpus
        test_size: size of the test set, in fractions of the original corpus

    Returns: 
        train and test set
    """
    raise NotImplementedError


def k_validation_folds(text: List[str], k_folds=10):
    """ Splits a corpus into k_folds cross-validation folds
        text: input corpus
        k_folds: number of cross-validation folds

    Returns: 
        the cross-validation folds
    """
    raise NotImplementedError


def plot_pp_vs_alpha(pps: List[float], alphas: List[float], N: int):
    """ Plots n-gram perplexity vs alpha
    Args:
        pps: list of perplexity scores
        alphas: list of alphas
        N: just for plotting
    """
    raise NotImplementedError
