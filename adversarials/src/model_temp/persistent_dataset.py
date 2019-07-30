from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split
import pandas

#Abstract class for a dataset that supports saving and loading data from files
class PersistentDataset(ABC):
    @abstractmethod
    def load_data(self, dataframe, test_percentile):
        pass

    @abstractmethod
    def save_train_test(self, train_filename, test_filename):
        pass

    @abstractmethod
    def load_train_test(self, train_filename, test_filename, preprocessed = True):
        pass

#Base implementation of the persistent dataset, can produce train test splits to be used by both the other dataset classes
class PersistentDatasetImpl(PersistentDataset):
    def __init__(self):
        self.data, self.train, self.test = None, None, None

    def load_data(self, dataframe, test_percentile):
        self.data = dataframe
        self.train, self.test = train_test_split(dataframe, test_size=test_percentile, shuffle=False)

    def save_train_test(self, train_filename, test_filename):
        if self.train is not None and self.test is not None:
            self.train.to_csv(path_or_buf=train_filename + '.csv', sep='\t', header=False, index=False)
            self.test.to_csv(path_or_buf=test_filename + '.csv', sep='\t', header=False, index=False)
        else:
            raise ValueError()

    def load_train_test(self, train_filename, test_filename, preprocessed = True):
        self.train = pandas.read_csv(train_filename + '.csv', sep='\t', header=None, names=["texts", "labels"], usecols=(0, 1))
        self.test = pandas.read_csv(test_filename + '.csv', sep='\t', header=None, names=["texts", "labels"], usecols=(0, 1))

    def get_train_test(self):
        if self.train is not None and self.test is not None:
            return self.train, self.test
        else:
            raise ValueError()