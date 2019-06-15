from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

class Token_Dataset(Dataset):
    def __init__(self, frame : pd.DataFrame, sparse = True):
        self.texts = frame["texts"]
        self.labels = frame["labels"].astype('category')
        self.num_labels = self.labels.cat.codes


        self.tokenizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_df=0.97, min_df=0.03)
        self.token_texts = self.tokenizer.fit_transform(self.texts)
        if not sparse: self.token_texts = self.token_texts.toarray()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_texts[index], self.labels[index]

def get_token_dataset():
    pass

if __name__ == "__main__":
    ds = Token_Dataset(pd.read_csv("hatespeech-data.csv", sep='\t', header=None, names=["texts", "labels"], usecols=(0,1)))