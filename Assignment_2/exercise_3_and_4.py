# TODO: Add your necessary imports here
from itertools import count
from typing import Union
from unittest import result

from sympy import per

from arrow import get
import numpy as np

import nltk
from nltk.corpus import brown


def load_corpus():
    """Load `Brown` corpus from NLTK"""

    # Download the Brown corpus if not already downloaded
    nltk.download("brown", quiet=True)

    # Load the Brown corpus
    corpus =  brown.words(categories="news")

    corpus = [word.lower() for word in corpus]

    return corpus


def get_bigram_freqs(text: list):
    """ Get bigram frequencies for the provided text.

    Args:
    text -- A `list` containing the tokenized text to be
            used to calculate the frequencies of bigrams
    """
    # Create a list of bigrams from the text
    bigrams = list(nltk.bigrams(text))

    # Calculate the frequency distribution of the bigrams
    freq_dist = nltk.FreqDist(bigrams)

    # Convert the frequency distribution to a dictionary
    freq_dict = dict(freq_dist)

    return freq_dict


def get_top_n_probabilities(text: list, context_words: Union[str, list], n: int):
    """ Get top `n` following words to `context_words` and their probabilities

    Args:
    text -- A list of tokens to be used as the corpus text
    context_words -- A `str` containing the context word(s) to be considered
    n    -- An `int` that indicates how many tokens to evaluate
    """
    # Get the bigram frequencies from the text
    bigram_freqs = get_bigram_freqs(text)

    # Create a dict to store the probabilities of the next words

    if isinstance(context_words, str):
        context_words = [context_words]

    freq_result = {}

    for context_word in context_words:
        total_count =  sum(count for (word1, word2), count in bigram_freqs.items() if word1 == context_word)
        if total_count == 0:
            continue

        next_word_probs = {}
        for (word1, word2), count in bigram_freqs.items():
            if word1 == context_word:
                next_word_probs[word2] = count / total_count

        top_p = sorted(next_word_probs.items(), key=lambda x: x[1], reverse=True)[:n]
        freq_result[context_word] = top_p
    return freq_result

def get_entropy(top_n_dict: dict):
    """ Get entropy of distribution of top `n` bigrams """

    entropies = {}

    for context_word, top_n in top_n_dict.items():
        entropy = 0.0
        for word, prob in top_n:
            if prob > 0:
                entropy -= prob * np.log2(prob)
        entropies[context_word] = entropy

    return entropies


def plot_top_n(top_n_dict: dict):
    """ Plot top `n` """
    import matplotlib.pyplot as plt
    if not top_n_dict:
        print("No data to plot.")
        return

    for context_word, top_n in top_n_dict.items():
        words, probs = zip(*top_n)
        plt.bar(words, probs)
        plt.title(f"Top {len(top_n)} words following '{context_word}'")
        plt.xlabel("Words")
        plt.ylabel("Probabilities")
        plt.xticks(rotation=45)
        plt.show()


def get_perplexity(bigram_str):
    words = bigram_str.lower().split()
    if len(words) != 2:
        raise ValueError("Input must be a string with exactly two words.")

    w1, w2 = words[0], words[1]
    target_bigram = (w1, w2)

    bigram_freqs = get_bigram_freqs(words)
    context_word_count = sum(count for (word1, word2), count in bigram_freqs.items() if word1 == w1)

    if context_word_count == 0:
        return np.inf

    bigram_count = bigram_freqs.get(target_bigram, 0)

    if bigram_count == 0:
        return np.inf

    prob = bigram_count / context_word_count

    perplexity = 1 / prob if prob > 0 else np.inf

    return perplexity


def get_mean_rank(bigram_str):
    words = bigram_str.lower().split()
    if len(words) != 2:
        raise ValueError("Input must be a string with exactly two words.")

    w1, w2 = words[0], words[1]
    bigram_freqs = get_bigram_freqs(words)
    context_word_count = sum(count for (word1, word2), count in bigram_freqs.items() if word1 == w1)

    if context_word_count == 0:
        return np.inf

    following_word_count = {}

    for (bw1, bw2), count in bigram_freqs.items():
        if bw1 == w1:
            following_word_count[bw2] = count

    if not following_word_count:
        return np.inf

    sorted_words = sorted(following_word_count.items(), key=lambda x: x[1], reverse=True)

    rank = 1
    found = False
    for word, count in sorted_words:
        if word == w2:
            found = True
            break
        rank += 1

    if found:
        return rank
    else:
        return np.inf
