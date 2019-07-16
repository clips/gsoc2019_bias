import numpy as np
import pickle

from src.utils.embeddings import glove_utils

class DistanceMatrix:
    def compute_distance_matrix(self, embedding_matrix):
        c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix)
        a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
        self.dist = a + a.T + c_

    def test_distance_matrix(self, word, dict, inv_dict):
        src_word = dict[word]
        neighbours, neighbours_dist = glove_utils.pick_most_similar_words(src_word, self.dist)
        print('Closest words to `good` are :')
        result_words = [inv_dict[x] for x in neighbours]
        print(result_words)