import os
import scipy.sparse as scisparse

from src.utils.embeddings.glove_utils import get_closest_words

DTYPE = os.getenv('MAT_DTYPE', 'float16')

class ReplacementDictionary:
    def __init__(self, matrix, dict, add = None, minus = None, vocabulary = None, limit = 3, threshold = None, dynamic = True):
        if minus is None:
            minus = []
        self.minus = minus
        if add is None:
            add = []
        self.add = add

        self.threshold = threshold
        self.limit = limit

        self.replacements = dict()
        self.matrix = matrix
        self.word_idx = dict

        if not dynamic:
            if vocabulary is None:
                print("You must provide a vocabulary for front-loaded dictionary initialization")
            else:
                for word in vocabulary:
                    self.get_replacements(word)

    def get_replacements(self, word):
        if word not in self.replacements.keys():
            self.replacements[word] = get_closest_words(self.matrix, self.word_idx, self.add, self.minus, self.limit, self.threshold)
        return self.replacements[word]