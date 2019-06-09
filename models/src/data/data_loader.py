from torch.utils.data import Dataset

class Token_Dataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def _make_tokenizer(self):
        pass

    def _tokenize_texts(self):
        pass

class Embedding_Dataset(Dataset):
    pass

def get_token_dataset():
    pass

def get_embedding_dataset():
    pass
