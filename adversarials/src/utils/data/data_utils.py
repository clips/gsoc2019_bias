import os

import pandas
from keras.preprocessing.text import Tokenizer

import pickle as pickle


class IMDBDataset(object):
    def __init__(self, path='aclImdb', max_vocab_size=None):
        self.path = path
        self.train_path = path + '/train'
        self.test_path = path + '/test'
        self.vocab_path = path + '/imdb.vocab'
        self.max_vocab_size = max_vocab_size
        self._read_vocab()
        train_text, self.train_y = self.read_text(self.train_path)
        test_text, self.test_y = self.read_text(self.test_path)
        self.train_text = train_text
        self.test_text = test_text
        print('tokenizing...')

        # Tokenized text of training data
        self.tokenizer = Tokenizer()

        # nlp = spacy.load('en')
        # train_text = [nltk.word_tokenize(doc) for doc in train_text]
        # test_text = [nltk.word_tokenize(doc) for doc in test_text]
        # train_text = [[w.string.strip() for w in nlp(doc)] for doc in train_text]
        # test_text = [[w.string.strip() for w in nlp(doc)] for doc in test_text]
        self.tokenizer.fit_on_texts(self.train_text)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1
        # sorted_words = sorted([x for x in self.tokenizer.word_counts])
        # self.top_words = sorted_words[:max_vocab_size-1]
        # self.other_words = sorted_words[max_vocab_size-1:]
        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]

        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word
        print('Dataset built !')

    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y), f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y), f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)

    def _read_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab_words = f.read().split('\n')
            self.vocab = dict([(w, i) for i, w in enumerate(vocab_words)])
            self.reverse_vocab = dict([(i, w) for w, i in self.vocab.items()])

    def read_text(self, path):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        pos_list = []
        neg_list = []

        pos_path = path + '/pos'
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_list = [open(x, 'r').read().lower() for x in pos_files]
        neg_list = [open(x, 'r').read().lower() for x in neg_files]
        data_list = pos_list + neg_list
        labels_list = [1] * len(pos_list) + [0] * len(neg_list)
        return data_list, labels_list

class TwitterDataset(object):
    def __init__(self, path, min_instances = 2):
        self.path = path
        self.frame = pandas.read_csv(path, sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.frame["texts"])

        max_vocab_size = len(self.tokenizer.word_counts)

        self.dict = dict()
        self.inv_dict = dict()

        #Create idx and inverse idx dictionaries, filter by min_instances
        skipped = 0
        for word, idx in self.tokenizer.word_index.items():
            if(self.tokenizer.word_counts[word] >= min_instances):
                self.inv_dict[idx] = word
                self.dict[word] = idx - skipped
            else:
                skipped += 1

        vocab_size = len(self.dict)
        self.dict['UNK'] = vocab_size
        self.inv_dict[max_vocab_size] = 'UNK'

    def get_dict(self):
        return self.dict, self.inv_dict

    def save(self):
        with open(self.path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)

if __name__ == "__main__":
    dataset = TwitterDataset("hatespeech-data.csv")
    print(len(dataset.dict))