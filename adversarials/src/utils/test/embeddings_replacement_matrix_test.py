import os
import csv

from src.utils.data.data_utils import TwitterDataset
from src.utils.embeddings.glove_utils import *
from src.utils.embeddings.replacement_dictionary import ReplacementDictionary

RESOURCES_PATH = os.getenv('RESOURCE_PATH')
TWITTER_PATH = RESOURCES_PATH + os.getenv('DATA_PATH')
GLOVE_PATH = [RESOURCES_PATH + path for path in os.getenv('GLOVE_PATH').replace("\"", "").split(sep=',')]
GENDER_PATH = RESOURCES_PATH + os.getenv('WORDS_PATH')


if __name__ == "__main__":
    print("Loading dataset dictionary")
    dict, inv_dict = TwitterDataset(path=TWITTER_PATH, min_instances=5).get_dict()
    print(GLOVE_PATH)
    print(GLOVE_PATH[-1])

    glove_name = GLOVE_PATH[-1]
    print("Loading glove model, name = {}".format(glove_name))
    glove_model, embedding_dim = load_glove_model(glove_name)

    print("Creating embedding matrix, dim = {}".format(embedding_dim))
    matrix, not_found = create_embeddings_matrix(glove_model, dict, d=embedding_dim)
    print("Words not found in embedding file: {}".format(len(not_found)))


    replacements = ReplacementDictionary(matrix, dict, add = ['woman'], minus = ['man'], vocabulary = None, limit = 3, threshold = None, dynamic = True)
    for word in dict.keys():
        print(word)
        print(replacements.get_replacements(word))