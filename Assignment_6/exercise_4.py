from collections import Counter, defaultdict
import string

def preprocess_text(text, unk_threshold=1):
    """Preprocesses raw text into tokens suitable for language modeling.
    
    Performs the following steps:
    1. Removes all punctuation and newline characters
    2. Converts text to lowercase
    3. Splits text into word tokens
    4. Replaces rare words (appearing fewer than unk_threshold times) with <UNK>
    
    Args:
        text (str): Raw input text to process
        unk_threshold (int, optional): Minimum count threshold for word retention.
            Words with counts below this will be replaced with <UNK>. Defaults to 1.
    
    Returns:
        list[str]: List of processed tokens with <UNK> replacements

    """
    # Remove punctuation and newlines
    translator = str.maketrans('', '', string.punctuation + '\n')
    text = text.translate(translator)
    # Convert to lowercase
    text = text.lower()
    # Split into tokens
    tokens = text.split()
    # Replace rare words with <UNK>
    word_counts = Counter(tokens)
    processed_tokens = []
    for token in tokens:
        if word_counts[token] < unk_threshold:
            processed_tokens.append('<UNK>')
        else:
            processed_tokens.append(token)
    
    return processed_tokens

class SmoothingCounter:
    """Language model with multiple smoothing techniques for n-gram probability estimation.
    
    Supports three smoothing methods:
    - Good-Turing estimation
    - Kneser-Ney smoothing (with fixed discount)
    - Add-alpha (Laplace) smoothing
    
    Attributes:
        d (float): Discount parameter for Kneser-Ney smoothing
        alpha (float): Smoothing parameter for add-alpha (Laplace) smoothing
        V (int): Vocabulary size (number of unique unigrams)
    """

    def __init__(self, text, alpha=0):
        """Initializes language model with n-gram counts and smoothing parameters.
        
        Args:
            text (list[str]): Preprocessed list of tokens
            alpha (float, optional): Smoothing parameter for add-alpha (Laplace) method. 
                Defaults to 0 (no smoothing).
        """
        self.text = text
        self.alpha = alpha
        self.d = 0.75

        self.unigram_counter = Counter(text)
        self.V = len(self.unigram_counter)

        self.bigram_counter = Counter()
        for i in range(len(text) - 1):
            bigram = (text[i], text[i + 1])
            self.bigram_counter[bigram] += 1
        
        self.trigram_counter = Counter()
        for i in range(len(text) - 2):
            trigram = (text[i], text[i + 1], text[i + 2])
            self.trigram_counter[trigram] += 1

        self.bigram_freq_of_freq = Counter(self.bigram_counter.values())
        self.trigram_freq_of_freq = Counter(self.trigram_counter.values())
        
        self._calculate_continuation_counts()
    
    def _calculate_continuation_counts(self):
        """Calculate continuation counts needed for Kneser-Ney smoothing."""
        # N_1+(•w) = number of unique words that precede w
        self.left_continuation = defaultdict(int)
        for (w1, w2) in self.bigram_counter:
            self.left_continuation[w2] += 1
        
        # N_1+(w1w2•) = number of unique words that follow w1w2
        self.right_continuation_trigram = defaultdict(int)
        for (w1, w2, w3) in self.trigram_counter:
            self.right_continuation_trigram[(w1, w2)] += 1
        
        # N_1+(w1•) = number of unique words that follow w1
        self.right_continuation_bigram = defaultdict(int)
        for (w1, w2) in self.bigram_counter:
            self.right_continuation_bigram[w1] += 1


    def prob_good_turing_bigram(self, bigram):
        """Computes Good-Turing smoothed probability for a bigram.
        
        Args:
            bigram (tuple[str, str]): Bigram to calculate probability for
        
        Returns:
            float: Smoothed probability P(w2|w1)
        """
        w1, w2 = bigram
        c = self.bigram_counter.get(bigram, 0)

        if c == 0:
            # c* = n1 / n
            n1 = self.bigram_freq_of_freq.get(1, 0)
            total_bigrams = sum(self.bigram_counter.values())
            c_star = n1 / total_bigrams if total_bigrams > 0 else 0
        else:
            # c* = (c+1) * N_{c+1} / N_c
            nc = self.bigram_freq_of_freq.get(c, 0)
            nc_plus_1 = self.bigram_freq_of_freq.get(c + 1, 0)
            
            if nc > 0 and nc_plus_1 > 0:
                c_star = (c + 1) * nc_plus_1 / nc
            else:
                c_star = c 
        
        # Normalize by unigram count
        w1_count = self.unigram_counter.get(w1, 0)
        return c_star / w1_count if w1_count > 0 else 0.0


    def prob_good_turing_trigram(self, trigram):
        """Computes Good-Turing smoothed probability for a trigram.
        
        Args:
            trigram (tuple[str, str, str]): Trigram to calculate probability for
        
        Returns:
            float: Smoothed probability P(w3|w1,w2)
        """
        w1, w2, w3 = trigram
        c = self.trigram_counter.get(trigram, 0)
        
        if c == 0:
            n1 = self.trigram_freq_of_freq.get(1, 0)
            total_trigrams = sum(self.trigram_counter.values())
            c_star = n1 / total_trigrams if total_trigrams > 0 else 0
        else:

            nc = self.trigram_freq_of_freq.get(c, 0)
            nc_plus_1 = self.trigram_freq_of_freq.get(c + 1, 0)
            
            if nc > 0 and nc_plus_1 > 0:
                c_star = (c + 1) * nc_plus_1 / nc
            else:
                c_star = c
        
        # Normalize by bigram count
        bigram_count = self.bigram_counter.get((w1, w2), 0)
        return c_star / bigram_count if bigram_count > 0 else 0.0

    def knprob_bigram(self, bigram):
        """Computes Kneser-Ney smoothed probability for a bigram.
        
        Args:
            bigram (tuple[str, str]): Bigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_KN(w3|w2)
        """
        w1, w2 = bigram
        
        # P_KN(w2|w1) = max(C(w1w2) - d, 0) / C(w1) + lambda(w1) * P_continuation(w2)
        
        bigram_count = self.bigram_counter.get(bigram, 0)
        w1_count = self.unigram_counter.get(w1, 0)
        
        if w1_count == 0:
            return 0.0
        
        # discounted probability
        first_term = max(bigram_count - self.d, 0) / w1_count
        
        # interpolation weight
        lambda_w1 = (self.d / w1_count) * self.right_continuation_bigram.get(w1, 0)
        
        # Continuation probability: N_1+(•w2) / total_unique_bigram_types
        continuation_prob = self.left_continuation.get(w2, 0) / len(self.bigram_counter)
        
        return first_term + lambda_w1 * continuation_prob

    def knprob_trigram(self, trigram):
        """Computes Kneser-Ney smoothed probability for a trigram.
        
        Args:
            trigram (tuple[str, str, str]): Trigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_KN(w3|w1,w2)
        """
        w1, w2, w3 = trigram
        
        # P_KN(w3|w1w2) = max(C(w1w2w3) - d, 0) / C(w1w2) + lambda(w1w2) * P_KN(w3|w2)
        
        trigram_count = self.trigram_counter.get(trigram, 0)
        bigram_count = self.bigram_counter.get((w1, w2), 0)
        
        if bigram_count == 0:
            # Back off to bigram
            return self.knprob_bigram((w2, w3))
        
        first_term = max(trigram_count - self.d, 0) / bigram_count
        lambda_w1w2 = (self.d / bigram_count) * self.right_continuation_trigram.get((w1, w2), 0)
        
        # Back off to bigram Kneser-Ney probability
        backoff_prob = self.knprob_bigram((w2, w3))
        
        return first_term + lambda_w1w2 * backoff_prob

    def prob_alpha_bigram(self, bigram):
        """Computes add-alpha (Laplace) smoothed bigram probability.
        
        Args:
            bigram (tuple[str, str]): Bigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_alpha(w2|w1)

        """
        w1, w2 = bigram
        num = self.bigram_counter.get(bigram, 0) + self.alpha
        den = self.unigram_counter.get(w1, 0) + self.alpha * self.V
        return num / den if den else 0.0

    def prob_alpha_trigram(self, trigram):
        """Computes add-alpha (Laplace) smoothed trigram probability.
        
        Args:
            trigram (tuple[str, str, str]): Trigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_alpha(w3|w1,w2)
        """
        w1, w2, w3 = trigram
        num = self.trigram_counter.get(trigram, 0) + self.alpha
        den = self.bigram_counter.get((w1, w2), 0) + self.alpha * self.V
        return num / den if den > 0 else 0.0