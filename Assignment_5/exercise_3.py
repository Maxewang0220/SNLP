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
    # Remove extra spaces and filter out non-alphabetical characters
    tokens = []
    for token in text:
        filtered = ''.join(ch for ch in token if ch.isalpha())
        if filtered:
            tokens.append(filtered.lower())
    return tokens


def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    Args:
        text: input corpus
        test_size: size of the test set, in fractions of the original corpus

    Returns: 
        train and test set
    """
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(text, test_size=test_size, shuffle=False)
    return train_data, test_data


def k_validation_folds(text: List[str], k_folds=10):
    """ Splits a corpus into k_folds cross-validation folds
        text: input corpus
        k_folds: number of cross-validation folds

    Returns: 
        the cross-validation folds
    """
    # Make each fold the same size
    n = len(text)
    fold_size = n // k_folds
    n_trimmed = fold_size * k_folds
    text= text[:n_trimmed]

    # Create KFolds
    kf = KFold(n_splits=k_folds, shuffle=False)
    folds = []
    indices = list(range(len(text)))
    for train_idx, test_idx in kf.split(indices):
        train_fold = [text[i] for i in train_idx]
        test_fold = [text[i] for i in test_idx]
        folds.append((train_fold, test_fold))
    return folds


def plot_pp_vs_alpha(pps: List[float], alphas: List[float], N: int):
    """ Plots n-gram perplexity vs alpha
    Args:
        pps: list of perplexity scores
        alphas: list of alphas
        N: just for plotting
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, pps, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Perplexity')
    plt.title(f'Perplexity vs Alpha for N={N}')
    plt.grid()
    plt.show()
