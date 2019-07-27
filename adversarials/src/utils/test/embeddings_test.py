import os
import csv

from src.utils.data.data_utils import TwitterDataset
from src.utils.embeddings.glove_utils import *

TWITTER_PATH = os.getenv('DATA_PATH')
GLOVE_PATH = os.getenv('GLOVE_PATH').replace("\"", "").split(sep=',')
GENDER_PATH = os.getenv('WORDS_PATH')


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

        top3 = 0
        for row in word_list:
            print("Pair: {}:{}".format(row[0], row[1]) )
            if row[0] in dict.keys() and row[1] in dict.keys():
                closest_m2f = get_closest_words(matrix, dict, [row[0], 'woman'], ['man'], 10)
                closest_f2m = get_closest_words(matrix, dict, [row[1], 'man'], ['woman'], 10)
                similarity_m2f = get_similarity(matrix, dict, [row[0], 'woman'], ['man'], row[1])
                similarity_f2m = get_similarity(matrix, dict, [row[1], 'man'], ['woman'], row[0])

                if(row[1] in [i[1] for i in closest_m2f[:3]]):
                    top3 += 1
                if(row[0] in [i[1] for i in closest_f2m[:3]]):
                    top3 += 1

                print("\tMale to female, Similarity: {} \n\tWords: ".format(similarity_m2f) + get_closest_words(matrix, dict, [row[0], 'woman'], ['man'], 10).__str__())
                print("\tFemale to male, Similarity: {} \n\tWords: ".format(similarity_f2m) + get_closest_words(matrix, dict, [row[1], 'man'], ['woman'], 10).__str__())
        print("Top3 results: {}".format(top3))