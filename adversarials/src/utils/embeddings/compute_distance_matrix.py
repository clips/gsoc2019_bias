import numpy as np
import os

from src.utils.embeddings import glove_utils

DTYPE = os.getenv('MAT_DTYPE', 'float16')

class DistanceMatrix:
    def compute_distance_matrix(self, embedding_matrix):
        c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix).astype(dtype=DTYPE)
        a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1)).astype(dtype=DTYPE)
        self.dist = (a + a.T + c_).astype(dtype=DTYPE)
        print(self.dist)

    def test_distance_matrix(self, word, dict, inv_dict, add_vector = None):
        print(dict[word])
        src_word = dict[word]
        neighbours, neighbours_dist = glove_utils.pick_most_similar_words(src_word, self.dist)
        print('Closest words to {} are :'.format(word))
        result_words = [inv_dict[x] for x in neighbours]
        print(result_words)
