import time

import joblib
import numpy
import pandas
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from src.model_temp.model import Classifier

from src.utils.model_temp.idf_dataset import IDFDataset


class SVM_Wrapper(Classifier):
    def __init__(self, vectorizer, C=1, loss='hinge', penalty='l2', max_iter=10000):
        svc = LinearSVC(C=C, loss=loss, penalty=penalty, max_iter=max_iter)
        self.vectorizer = vectorizer
        self.classifier = CalibratedClassifierCV(svc, cv=3)
        self.training_time = None

    def train(self, x_train, y_train):
        start = time.perf_counter()
        self.classifier.fit(x_train, y_train)
        self.training_time = time.perf_counter() - start

    def predict(self, x_test, plain=False):
        if plain:
            return self.classifier.predict_proba(self.vectorizer.transform(x_test))
        else:
            return self.classifier.predict_proba(x_test)

    def predict_one(self, x_single, plain=False):
        if plain:
            return self.classifier.predict_proba(self.vectorizer.transform([x_single]))
        else:
            return self.classifier.predict_proba(x_single)

    def save_model(self, filename: str):
        joblib.dump(self.classifier, filename)

    def load_model(self, filename: str):
        self.classifier = joblib.load(filename)

    def get_training_time(self):
        if self.training_time is None:
            raise ValueError()
        else:
            return self.training_time

if __name__ == "__main__":
    print("Loading file")
    frame = pandas.read_csv("hatespeech-data.csv", sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))

    print("Tokenizing data")
    dataset = IDFDataset()
    dataset.load_data(frame, 0.25)
    tokens, labels = dataset.get_train_dataset()
    test_tokens, test_labels = dataset.get_test_dataset()

    print("Training classifier")
    svm = SVM_Wrapper(dataset.tokenizer)
    svm.train(tokens, labels)
    predicted = [numpy.argmax(prob) for prob in svm.predict(test_tokens)]
    print("Accuracy {}".format(svm.classifier.score(test_tokens, test_labels)))
    print("F1 Score {}".format(f1_score(test_labels, predicted, average='weighted')))

    #svm.save_model("SVMModel.pk")