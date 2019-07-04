from abc import ABC, abstractmethod

class Classifier(ABC):
    @abstractmethod
    def save_model(self, filename : str):
        pass

    @abstractmethod
    def load_model(self, filename : str):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def predict_one(self, x_single):
        pass