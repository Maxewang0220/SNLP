from copy import deepcopy
import numpy as np
from typing import List
import math 
from collections import defaultdict,Counter
import nltk
import string
nltk.download('treebank')

def load_and_preprocess_data(max_ngram_order=10):
  corpus=list(nltk.corpus.treebank.sents())
  cleaned_corpus=[]
  for sent in corpus:
    #Remove punctuations
    words = [word.translate(str.maketrans(dict.fromkeys(string.punctuation))).lower() for word in sent]
    #Remove empty tokens after cleaning
    words=" ".join(words).split()

    #Removing sentences with less words than our highest ngram order
    #This has to be edited if increasing ngram order higher than 10
    if len(words)>=max_ngram_order:
      cleaned_corpus.append(words)
  return cleaned_corpus

def make_vocab(corpus,top_n):
  '''
  Make the top_n frequent vocabulary from a corpus
  Input: corpus - List[List[str]]
         top_n - int
  Output: Vocabulary - List[str]
  '''
  pass

def restrict_vocab(corpus,vocab):
  '''
  Make the corpus fit inside the vocabulary using <unk>
  Input: corpus - List[List[str]]
         vocab  - List[str]
  Output: Vocabulary_restricted_corpus - List[List[str]]
  '''
  pass

def train_test_split(corpus, split=0.7):
  '''Splits the corpus using a 70:30 ratio. Do not randomize anything here. use the original order
  Input: List[List[str]]
  Output: List[List[str]],List[List[str]]'''
  pass


class Interpolated_Model:
    
    def __init__(self, train_sents: List[List[str]], test_sents: List[List[str]], alpha=0,order=2):
        """ 
        :param train_sents: list of sents from the train section of your corpus
        :param test_sents: list of sents from the test section of your corpus
        :param alpha :  Smoothing factor for laplace smoothing
        :function perplexity() : Calculates perplexity on the test set
        Tips:  Try out the Counter() module from collections to Count ngrams. 
        """
        self.alpha = alpha
        self.order=order
        self.interpolation_weight=1/self.order
        
        #Counting the total number of words in the corpus
        total_words=sum([len(sent) for sent in train_sents])
        self.train_counts=[Counter() for i in range(self.order+1)]

        for sentence in train_sents:
          for ord_ in range(1,self.order+1):
            self.train_counts[ord_]+=Counter(self._get_n_grams(sentence, ord_))

        assert sum(self.train_counts[1].values())==total_words, 'Not all unigrams accounted'

        #At Oth order, return the total number of words for proper normalization
        self.train_counts[0]={():total_words}

        #Getting vocabulary size for laplace smoothing
        self.vocab_size=len(self.train_counts[1])

        #Getting higest order ngrams from the test set
        self.test_ngrams = [self._get_n_grams(sent, self.order) for sent in test_sents]

    def _get_n_grams(self, tokens: List[str], n: int):
      '''
      gets the ngrams out for an arbitrary n value. 
      input: list of tokens
      '''
      n_grams = []
      if n == 0:
          n_grams = [tuple([t]) for t in tokens]
      else:
          for i in range(len(tokens)-n+1):
              n_gram = tuple(tokens[i:i+n])
              n_grams.append(n_gram)
      return n_grams    

    def laplace_prob(self,ngram):
      '''returns the log proabability of an ngram. Adjust this function for Laplace Smoothing'''
      pass

    def interpolated_logprob(self,ngram):
      '''
      calculates the interpolated log probability of a given n-gram using the Laplace smoothed probabilities.
      '''
      pass

    def perplexity(self):
      """ returns the perplexity of the language model for n-grams with n=n """
      pass
