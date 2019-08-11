import os
import string

import scipy.sparse as scisparse

from src.utils.embeddings.glove_utils import get_closest_words
from sklearn.feature_extraction import stop_words

class ReplacementDictionary:
    def __init__(self, matrix, word_idx, add = None, minus = None, vocabulary = None, limit = 3, dynamic = True):
        if minus is None:
            minus = []
        self.minus = minus
        if add is None:
            add = []
        self.add = add
        self.limit = limit

        self.replacements = dict()
        self.matrix = matrix
        self.word_idx = word_idx

        if not dynamic:
            if vocabulary is None:
                print("You must provide a vocabulary for front-loaded dictionary initialization")
            else:
                for word in vocabulary:
                    self.get_replacements(word)


    #TODO: Filter out replacements that are closer to the original word than the target vector
    def get_replacements(self, word):
        word = word.translate(str.maketrans('','', string.punctuation))
        if word not in self.replacements.keys():
            if word in stop_words.ENGLISH_STOP_WORDS:
                self.replacements[word] = []
            else:
                try:
                    standard_replacements = [item[1] for item in
                                             get_closest_words(self.matrix, self.word_idx, [word], [], self.limit)]
                    swap_replacements = [item[1] for item in
                                         get_closest_words(self.matrix, self.word_idx, self.add + [word], self.minus, self.limit)]

                    self.replacements[word] = [item for item in swap_replacements if item not in standard_replacements]
                except Exception as e:
                    self.replacements[word] = []
        return self.replacements[word]