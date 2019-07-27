import numpy
import numpy as np
import pickle
import scipy.spatial.distance as dist

UNK_TOKEN = 'UNK'

def load_glove_model(filename):
    f = open(filename, 'r', encoding='utf-8')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding

    embedding_dim = len(next(iter(model.values())))

    if UNK_TOKEN not in model.keys():
        model[UNK_TOKEN] = np.random.rand(embedding_dim)

    return model, embedding_dim

def create_embeddings_matrix(glove_model, dictionary, d=300):
    vocab_size = len(dictionary)
    embedding_matrix = np.zeros(shape=(d, vocab_size + 1))
    not_found = []

    for w, i in dictionary.items():
        if w not in glove_model:
            embedding_matrix[:, i] = glove_model[UNK_TOKEN]
            not_found.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]
    return embedding_matrix, not_found

def get_closest_words(matrix, dict, add, minus, limit, threshold = None):
    vector = numpy.zeros(matrix.shape[0])
    for word in add:
        vector += matrix[:,dict[word]]
    for word in minus:
        vector -= matrix[:,dict[word]]

    words_with_distance = [(1 - dist.cosine(vector, matrix[:,dict[w]]), w) for w in dict.keys()]
    return sorted(words_with_distance, key=lambda t: t[0], reverse=True)[:limit]

def get_similarity(matrix, dict, add, minus, target):
    vector = numpy.zeros(matrix.shape[0])
    for word in add:
        vector += matrix[:,dict[word]]
    for word in minus:
        vector -= matrix[:,dict[word]]
    return 1 - dist.cosine(vector, matrix[:,dict[target]])