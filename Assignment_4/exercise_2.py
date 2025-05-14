import math
from collections import Counter
import token
from typing import List, Union

from tokenizers import Tokenizer
from morfessor.baseline import BaselineModel


class TokenizerEntropy:
    def tokenize_bpe(self, tokenizer: Tokenizer, text: str) -> List[str]:
        """
        Takes the BPE tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained BPE tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """

        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        return tokens

    def tokenize_morfessor(self, tokenizer: BaselineModel, text: str) -> List[str]:
        """
        Takes the Morfessor tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained Morfessor tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """

        tokens = []
        for word in text.split():
            if word:
                # get the morfessor tokens
                segments = tokenizer.viterbi_segment(word)
                # get the morfessor tokens
                tokens.extend(segments[0])
        return tokens
    def get_probs(self, tokens: List[str]):
        """
        Takes a list of tokens and compute the probability distribution of the tokens.

        params:
        - tokens: A list of tokens

        returns a dictionary of token probabilities i.e. {token: probability, ...}
        """
        token_counts = Counter(tokens)
        total_count = len(tokens)
        probs = {token: count / total_count for token, count in token_counts.items()}
        return probs

    def compute_entropy(
        self, text: str, tokenizer: Union[Tokenizer, BaselineModel]
    ) -> float:
        """
        Takes the input text and the tokenizer and returns the entropy of the text.

        params:
        - text: The input text
        - tokenizer: The pre-trained tokenizer (BPE or Morfessor)

        returns the entropy of the text
        """
        # tokenize the input text
        if isinstance(tokenizer, Tokenizer):
            tokens = self.tokenize_bpe(tokenizer, text)
        elif isinstance(tokenizer, BaselineModel):
            tokens = self.tokenize_morfessor(tokenizer, text)
        else:
            raise ValueError("Tokenizer not supported.")

        # get the probabilities of the tokens
        probs = self.get_probs(tokens)
        # compute the entropy
        entropy = -sum(p * math.log2(p) for p in probs.values())
        return entropy