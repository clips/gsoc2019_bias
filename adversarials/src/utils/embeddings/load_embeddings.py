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