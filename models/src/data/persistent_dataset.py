from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class persistent_dataset(ABC, Dataset):
    @abstractmethod
    def load_data(self, filename):
        pass

    @abstractmethod
    def save_train_test(self, train_filename, test_filename):
        pass

    @abstractmethod
    def load_train_test(self, train_filename, test_filename):
        pass