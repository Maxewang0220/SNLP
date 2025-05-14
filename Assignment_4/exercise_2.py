import math
from collections import Counter
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
        # ====================================
        # Your code here
        
        # ====================================
        encoding = tokenizer.encode(text)
        return encoding.tokens
    
    def tokenize_morfessor(self, tokenizer: BaselineModel, text: str) -> List[str]:
        """
        Takes the Morfessor tokenizer and a text and returns the list of tokens.

        params:
        - tokenizer: The pre-trained Morfessor tokenizer
        - text: The input text to tokenize

        returns a list of tokens
        """
        # ====================================
        # Your code here

        # ====================================
        words = text.split()
        tokens = []

        for word in words:
            segments = tokenizer.viterbi_segment(word)[0]
            tokens.extend(segments)
            
        return tokens

    def get_probs(self, tokens: List[str]):
        """
        Takes a list of tokens and compute the probability distribution of the tokens.

        params:
        - tokens: A list of tokens

        returns a dictionary of token probabilities i.e. {token: probability, ...}
        """
        # ====================================
        # Your code here

        # ====================================
        token_size = len(tokens)
        token_counts = Counter(tokens)
        
        token_probs = {token: count/token_size for token, count in token_counts.items()}
        return token_probs

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

        # ====================================
        # Your code here

        # get the probabilities of each token
        token_probs = self.get_probs(tokens)
        # Compute the entropy
        entropy = -sum(p * math.log2(p) for p in token_probs.values())

        return entropy
        # ====================================
        
