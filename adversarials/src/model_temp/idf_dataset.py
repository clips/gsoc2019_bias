import pandas
import torch

from src.model_temp.persistent_dataset import PersistentDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset

class IDFDataset(PersistentDataset):
    def __init__(self):
        self.data, self.train_tokens, self.test_tokens = None, None, None
        self.train_labels, self.test_labels = None, None

    def load_data(self, dataframe, test_percentile):
        self.data = dataframe
        #TODO: Establish if TFIDF vectorizer maintains the pandas index
        train, test = train_test_split(dataframe, test_size=test_percentile, shuffle=False)
        self._process_data(train, test)

    def save_train_test(self, train_filename, test_filename):
        if self.train_tokens is not None and self.test_tokens is not None:
            save_sparse_csr(train_filename, self.train_tokens)
            save_sparse_csr(test_filename, self.test_tokens)
            self.train_labels.to_csv(path_or_buf=train_filename + '.csv',sep='\t', header=False, index=False)
            self.test_labels.to_csv(path_or_buf=test_filename + '.csv',sep='\t', header=False, index=False)
        else:
            raise ValueError()

    def load_train_test(self, train_filename, test_filename, preprocessed = True):
        if preprocessed is True:
            self.train_tokens = load_sparse_csr(train_filename)
            self.test_tokens = load_sparse_csr(test_filename)
            self.train_labels = pandas.read_csv(train_filename + '.csv', sep='\t', header=None, names=["labels"])
            self.test_labels = pandas.read_csv(test_filename + '.csv', sep='\t', header=None, names=["labels"])
        else:
            train = pandas.read_csv(train_filename + '.csv', sep='\t', header=None, names=["texts", "labels"], usecols=(0, 1))
            test = pandas.read_csv(test_filename + '.csv', sep='\t', header=None, names=["texts", "labels"], usecols=(0, 1))
            self._process_data(train, test)

    def get_train_dataset(self):
        return self.train_tokens, self.train_labels

    def get_test_dataset(self):
        return self.test_tokens, self.test_labels

    def get_train_dataset_torch(self) -> Dataset:
        input_tensor = convert_sparse_matrix_to_tensor(self.train_tokens)
        label_tensor = torch.LongTensor(self.train_labels)
        return TensorDataset(input_tensor, label_tensor)

    def get_test_dataset_torch(self) -> Dataset:
        input_tensor = convert_sparse_matrix_to_tensor(self.train_tokens)
        label_tensor = torch.LongTensor(self.train_labels)
        return TensorDataset(input_tensor, label_tensor)

    def _process_data(self, train, test):
        train_texts, train_labels = train['texts'], train['labels'].astype('category')
        test_texts, test_labels = test['texts'], test['labels'].astype('category')

        self.num_labels = train_labels.nunique()

        self.train_labels = train_labels.cat.codes
        self.test_labels = test_labels.cat.codes

        self.tokenizer = TfidfVectorizer(strip_accents='unicode', min_df=10)
        self.tokenizer.fit(train['texts'])

        self.train_tokens = self.tokenizer.transform(train['texts'])
        self.test_tokens = self.tokenizer.transform(test['texts'])



if __name__ == '__main__':
    print("Loading file")
    frame = pandas.read_csv("hatespeech-data.csv", sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))

    print("Tokenizing data")
    dataset = IDFDataset()
    dataset.load_data(frame, 0.25)