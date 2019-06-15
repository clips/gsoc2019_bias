from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class Token_Dataset(Dataset):
    def __init__(self, frame : pd.DataFrame):
        self.texts = frame["texts"]
        self.labels = frame["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def _make_tokenizer(self):
        pass

    def _tokenize_texts(self):
        pass

def get_token_dataset():
    pass

if __name__ == "__main__":
    Token_Dataset(pd.read_csv("hatespeech-data.csv", sep='\t', header=None, names=["texts", "labels"], usecols=(0,1)))
