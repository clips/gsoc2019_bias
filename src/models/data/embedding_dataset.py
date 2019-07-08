import pandas
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.vocab import Vectors

from src.models.data.persistent_dataset import PersistentDataset
from src.models.utils.torchtext_dataframe_set import DataFrameDataset


class EmbeddingDataset(PersistentDataset):
    def load_data(self, dataframe, test_percentile):
        self.data = dataframe
        train, test = train_test_split(dataframe, test_size=test_percentile, shuffle=False)
        self._process_data(train, test)

    def save_train_test(self, train_filename, test_filename):
        pass

    def load_train_test(self, train_filename, test_filename, preprocessed=True):
        pass

    def _process_data(self, train, test):
        self.text_field = data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, fix_length=200)
        self.label_field = data.LabelField()
        glove = Vectors(name='glove.twitter.27B.100d.txt')
        dataset_train = DataFrameDataset(train, self.text_field, self.label_field)

        self.text_field.build_vocab(dataset_train, vectors=glove)
        self.label_field.build_vocab(dataset_train)

        word_embeddings = self.text_field.vocab.vectors
        print("Length of Text Vocabulary: " + str(len(self.text_field.vocab)))
        print("Vector size of Text Vocabulary: ", self.text_field.vocab.vectors.size())
        print("Label Length: " + str(len(self.text_field.vocab)))
        # print(self.text_field.vocab.stoi['hello'])
        # print(self.text_field.process('hello'))

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32,
                                                                       sort_key=lambda x: len(x.text), repeat=False,
                                                                       shuffle=True)

        vocab_size = len(self.text_field.vocab)
        return self.text_field, vocab_size, word_embeddings, train_iter, test_iter

if __name__ == '__main__':
    print("Loading file")
    frame = pandas.read_csv("hatespeech-data.csv", sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))

    print("Tokenizing data")
    dataset = EmbeddingDataset()
    dataset.load_data(frame, 0.9)