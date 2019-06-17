import time
import numpy as np
from sklearn.svm import LinearSVC
from torch.utils.data import random_split


from models.src.data.token_data_loader import get_token_dataset

class SVM_Classifier:
    def __init__(self, C=1, loss='hinge', penalty='l2', max_iter=10000):
        self.classifier = LinearSVC(C=C, loss=loss, penalty=penalty, max_iter=max_iter)

    def train(self, x_train, y_train):
        start = time.perf_counter()
        self.classifier.fit(x_train, y_train)
        self.time = time.perf_counter() - start

    def score(self, x_test, y_test):
        return self.classifier.score(x_test, y_test)

    def predict(self, x):
        return self.classifier.predict(x)

if __name__ == "__main__":
    train_set, test_set = get_token_dataset()

    svm = SVM_Classifier()
    svm.train(train_set.token_texts, train_set.num_labels)
    print(svm.score(test_set.token_texts, test_set.num_labels))
