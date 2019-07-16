import numpy as np
import pickle

import os

from src.utils.data import data_utils
from src.utils.embeddings import glove_utils


class EmbeddingLoader:
    def __init__(self, embedding_path, dataset_path):
        self.path = embedding_path
        self.dict = self._load_twitter_dataset(dataset_path)

    def _load_twitter_dataset(self, dataset_path):
        set = data_utils.TwitterDataset(path=dataset_path)
        return set.dict

    def _load_embeddings(self):
        model = glove_utils.load_glove_model
        self.matrix = glove_utils.create_embeddings_matrix(model, self.dict)

    def get_embedding_matrix(self):
        return self.matrix


if __name__ == "__main__":
    IMDB_PATH = 'aclImdb'
    MAX_VOCAB_SIZE = 50000
    GLOVE_PATH = 'glove.840B.300d.txt'
    if not os.path.exists('aux_files'):
        os.mkdir('aux_files')
    imdb_dataset = data_utils.IMDBDataset(path=IMDB_PATH, max_vocab_size=MAX_VOCAB_SIZE)

    # save the dataset
    with open(('aux_files/dataset_%d.pkl' % MAX_VOCAB_SIZE), 'wb') as f:
        pickle.dump(imdb_dataset, f)

    # create the glove embeddings matrix (used by the classification model)
    glove_model = glove_utils.load_glove_model(GLOVE_PATH)
    glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, imdb_dataset.dict, imdb_dataset.full_dict)
    # save the glove_embeddings matrix
    np.save('aux_files/embeddings_glove_%d.npy' % MAX_VOCAB_SIZE, glove_embeddings)

    # Load the counterfitted-vectors (used by our attack)
    glove2 = glove_utils.load_glove_model('counter-fitted-vectors.txt')
    # create embeddings matrix for our vocabulary
    counter_embeddings, missed = glove_utils.create_embeddings_matrix(glove2, imdb_dataset.dict, imdb_dataset.full_dict)

    # save the embeddings for both words we have found, and words that we missed.
    np.save(('aux_files/embeddings_counter_%d.npy' % MAX_VOCAB_SIZE), counter_embeddings)
    np.save(('aux_files/missed_embeddings_counter_%d.npy' % MAX_VOCAB_SIZE), missed)
    print('All done')
