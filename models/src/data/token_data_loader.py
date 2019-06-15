from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

class Token_Dataset(Dataset):
    def __init__(self, frame : pd.DataFrame):
        self.texts = frame["texts"]
        self.labels = frame["labels"]

        self.token_texts = None
        self.categoritcal_labels = None

        self._make_tokenizer()
        self._tokenize_texts()
        self._categorize_labels()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_texts[index], self.labels[index]

    def _make_tokenizer(self):
        self.tokenizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_df=0.97, min_df=0.03)

    def _tokenize_texts(self):
        self.token_texts = self.tokenizer.fit_transform(self.texts)

    def _categorize_labels(self):
        pass

def get_token_dataset():
    pass

if __name__ == "__main__":
    ds = Token_Dataset(pd.read_csv("hatespeech-data.csv", sep='\t', header=None, names=["texts", "labels"], usecols=(0,1)))