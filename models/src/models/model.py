from abc import ABC, abstractmethod

class model(ABC):
    @abstractmethod
    def save_model(self, filename : str):
        pass

    @abstractmethod
    def load_model(self, filename : str):
        pass

    @abstractmethod
    def predict(self, y_input):
        pass

    @abstractmethod
    def train(self, x_input, x_labels):
        pass