import logging

from src.utils.data.data_utils import TwitterDataset
from src.utils.embeddings.compute_distance_matrix import DistanceMatrix
from src.utils.embeddings.glove_utils import *

TWITTER_PATH = "hatespeech-data.csv"
GLOVE_PATH = "glove.twitter.27B.100d.txt"

LOG_FORMAT = ('%(levelname) -5s %(asctime) -10s %(name) -5s %(funcName) -5s %(lineno) -10d: %(message)s')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

if __name__ == "__main__":
    print("Loading dataset dictionary")
    dict, inv_dict = TwitterDataset(path=TWITTER_PATH, min_instances=5).get_dict()
    print("Loading glove model")
    glove_model = load_glove_model(GLOVE_PATH)
    print("Creating embedding matrix")
    matrix, not_found = create_embeddings_matrix(glove_model, dict, d=100)
    dist_matrix = DistanceMatrix()
    print("Computing distance matrix")
    dist_matrix.compute_distance_matrix(matrix)
    dist_matrix.test_distance_matrix("good", dict, inv_dict)