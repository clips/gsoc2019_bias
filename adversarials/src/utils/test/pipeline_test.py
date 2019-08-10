import os
import pandas

from src.logic.gender_perturbation import GenderSwitchAttackBaseline
from src.utils.data.data_utils import TwitterDataset
from src.utils.embeddings.glove_utils import load_glove_model, create_embeddings_matrix
from src.utils.embeddings.replacement_dictionary import ReplacementDictionary
from src.utils.model_temp.idf_dataset import IDFDataset
from src.utils.model_temp.svc import SVMWrapper

RESOURCES_PATH = os.getenv('RESOURCE_PATH')
TWITTER_PATH = RESOURCES_PATH + os.getenv('DATA_PATH')
GLOVE_PATH = [RESOURCES_PATH + path for path in os.getenv('GLOVE_PATH').replace("\"", "").split(sep=',')]
GENDER_PATH = RESOURCES_PATH + os.getenv('WORDS_PATH')
SAMPLES_PATH = RESOURCES_PATH + os.getenv('SAMPLES_PATH')

def load_dataset():
    print("Loading dataset")
    frame = pandas.read_csv("hatespeech-data.csv", sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))

    print("Tokenizing data")
    dataset = IDFDataset()
    dataset.load_data(frame, 0.25)
    tokens, labels = dataset.get_train_dataset()

    print("Training classifier")
    svm = SVMWrapper(dataset.tokenizer)
    svm.train(tokens, labels)

    return svm

def load_replacements(inverse = False):
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

    print("Creating replacement matrix")
    if inverse:
        print("woman-man")
        return ReplacementDictionary(matrix, dict, add = ['woman'], minus = ['man'], vocabulary = None, limit = 3, dynamic = True)
    else:
        print("man-woman")
        return ReplacementDictionary(matrix, dict, add = ['woman'], minus = ['man'], vocabulary = None, limit = 3, dynamic = True)

def load_samples():
    print("Loading samples")
    frame = pandas.read_csv("hatespeech-data.csv", sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))
    return frame

if __name__ == "__main__":
    svm = load_dataset()
    matrix = load_replacements()
    samples = load_samples()

    print("Generating attack")
    attack = GenderSwitchAttackBaseline(svm, samples, matrix)
