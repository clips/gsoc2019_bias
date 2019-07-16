from torchtext import data

class DataFrameDataset(data.Dataset):
    def __init__(self, df, text_field, label_field, is_test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i, row in df.iterrows():
            label = row.labels if not is_test else None
            text = row.texts
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)