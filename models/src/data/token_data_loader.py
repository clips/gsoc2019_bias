from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

class Token_Dataset(Dataset):
    def __init__(self, frame : pd.DataFrame, min_df = 5, sparse = True):
        self.texts = frame["texts"]
        self.labels = frame["labels"].astype('category')
        self.num_labels = self.labels.cat.codes


        self.tokenizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=min_df)
        self.token_texts = self.tokenizer.fit_transform(self.texts)
        if not sparse: self.token_texts = self.token_texts.toarray()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_texts[index], self.num_labels[index]

def get_token_dataset():
    pass

if __name__ == "__main__":
    ds = Token_Dataset(pd.read_csv("hatespeech-data.csv", sep='\t', header=None, names=["texts", "labels"], usecols=(0,1)),sparse=False)
    print(ds[4])