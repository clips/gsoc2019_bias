import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from models.src.utils.tensor_utils import convert_sparse_matrix_to_sparse_tensor


class Token_Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, min_df=5, sparse=True, tokenizer = None):
        self.texts = frame["texts"]
        self.labels = frame["labels"].astype('category')
        self.num_labels = self.labels.cat.codes

        if tokenizer is None:
            self.tokenizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=min_df)
            self.tokenizer.fit(self.texts)
        else:
            self.tokenizer = tokenizer
        self.token_texts = self.tokenizer.transform(self.texts)
        if not sparse: self.token_texts = self.token_texts.toarray()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_texts[index], self.num_labels[index]

    def make_tensor_set(self):
        self.token_texts = convert_sparse_matrix_to_sparse_tensor(self.token_texts)
        self.num_labels = torch.LongTensor(self.num_labels.tolist())

def get_token_dataset(train_test_percentile = 0.25):
    frame = pd.read_csv("hatespeech-data.csv", sep='\t', header=None, names=["texts", "labels"], usecols=(0, 1))
    train, test = train_test_split(frame, test_size=0.25)

    train = Token_Dataset(train)
    test = Token_Dataset(test, tokenizer=train.tokenizer)

    return train, test

if __name__ == "__main__":
    get_token_dataset()
