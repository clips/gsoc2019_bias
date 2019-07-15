import numpy as np
import pickle


def load_glove_model(filename):
    f = open(filename, 'r')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    return model


def save_glove_to_pickle(glove_model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(glove_model, f)


def load_glove_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def create_embeddings_matrix(glove_model, dictionary, d=300):
    vocab_size = len(dictionary)
    embedding_matrix = np.zeros(shape=(d, vocab_size + 1))
    not_found = []

    for w, i in dictionary.items():
        if w not in glove_model:
            embedding_matrix[:, i] = glove_model['UNK']
            not_found.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]

    print('Number of not found words = ', len(not_found))
    return embedding_matrix, not_found


def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    dist_order = np.argsort(dist_mat[src_word, :])[1:1 + ret_count]
    dist_list = dist_mat[src_word][dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list
