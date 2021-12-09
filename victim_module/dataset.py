import torch
from victim_config import IMDB_Config
from torch.utils.data import Dataset
from tools import logging
from transformers import AutoTokenizer


class IMDB_Dataset(Dataset):
    def __init__(self, train_data=True, debug_mode=False):
        super(IMDB_Dataset, self).__init__()
        if train_data:
            self.path = IMDB_Config.train_data_path
        else:
            self.path = IMDB_Config.test_data_path
        self.sentences, self.labels = self.read_standard_data(self.path, debug_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(IMDB_Config.tokenizer_name)
        self.encodings = self.tokenizer(self.sentences, padding=True, truncation=True, max_length=IMDB_Config.sen_len, return_tensors='pt')
        self.encodings['classification_label'] = torch.tensor(self.labels)

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 250
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels


    def __getitem__(self, item):
        return self.encodings['input_ids'][item], self.encodings['attention_mask'][item], self.encodings['token_type_ids'][item], self.encodings['classification_label'][item]

    def __len__(self):
        return len(self.encodings['input_ids'])