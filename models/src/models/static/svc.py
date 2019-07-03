import time
from sklearn.svm import LinearSVC
import joblib

from models.src.data.token_data_loader import get_token_dataset
from models.src.models.model import classifier

class SVM_Wrapper(classifier):
    def __init__(self, C=1, loss='hinge', penalty='l2', max_iter=10000):
        self.classifier = LinearSVC(C=C, loss=loss, penalty=penalty, max_iter=max_iter)
        self.training_time = None

    def train(self, x_train, y_train):
        start = time.perf_counter()
        self.classifier.fit(x_train, y_train)
        self.training_time = time.perf_counter() - start

    def predict(self, x_test):
        return self.classifier.predict(x_test)

    def predict_one(self, x_single):
        return self.classifier.predict(x_single)

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
    train_set, test_set = get_token_dataset()

    svm = SVM_Wrapper()
    svm.train(train_set.token_texts, train_set.num_labels)
    svm.save_model("SVMModel.pk")
