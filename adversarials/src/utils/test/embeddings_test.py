import logging
import os

import numpy
import scipy.spatial.distance as dist
import csv

from src.utils.data.data_utils import TwitterDataset
from src.utils.embeddings.compute_distance_matrix import DistanceMatrix
from src.utils.embeddings.glove_utils import *

TWITTER_PATH = "hatespeech-data.csv"
GLOVE_PATH = os.getenv('GLOVE_PATH').replace("\"", "").split(sep=',')
GENDER_PATH = "gendered-word-pairs.csv"

def get_closest_words(matrix, dict, add, minus, limit, threshold = None):
    vector = numpy.zeros(matrix.shape[0])
    for word in add:
        vector += matrix[:,dict[word]]
    for word in minus:
        vector -= matrix[:,dict[word]]

    words_with_distance = [(1 - dist.cosine(vector, matrix[:,dict[w]]), w) for w in dict.keys()]
    return sorted(words_with_distance, key=lambda t: t[0], reverse=True)[:10]

if __name__ == "__main__":
    print("Loading dataset dictionary")
    dict, inv_dict = TwitterDataset(path=TWITTER_PATH, min_instances=5).get_dict()
    for glove_name in GLOVE_PATH:
        print("Loading glove model, name = {}".format(glove_name))
        glove_model, embedding_dim = load_glove_model(glove_name)

        print("Creating embedding matrix, dim = {}".format(embedding_dim))
        matrix, not_found = create_embeddings_matrix(glove_model, dict, d=embedding_dim)
        print("Words not found in embedding file: {}".format(len(not_found)))

        with open(GENDER_PATH, 'r') as f:
            reader = csv.reader(f)
            word_list = list(reader)

        for row in word_list:
            print("Pair: {}:{}".format(row[0], row[1]) )
            if row[0] in dict.keys() and row[1] in dict.keys():
                print("Male to female: " + get_closest_words(matrix, dict, [row[0], 'woman'], ['man'], 10).__str__())
                print("Female to male: " + get_closest_words(matrix, dict, [row[1], 'man'], ['woman'], 10).__str__())