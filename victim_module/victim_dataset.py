import os
import re
import torch
from victim_config import IMDB_Config
from torch.utils.data import Dataset
from tools import logging
from transformers import AutoTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup stopwords list & word (noun, adjective, and verb) lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Function to clean text using RegEx operations, removal of stopwords, and lemmatization."""
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(' ')]
    text = [lemmatizer.lemmatize(token, 'v') for token in text]
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = text.lstrip().rstrip()
    text = re.sub(r'<br />', '', text)
    text = re.sub(' +', ' ', text)
    return text

class IMDB_Dataset(Dataset):
    def __init__(self, train_data=True, debug_mode=False):
        super(IMDB_Dataset, self).__init__()
        if train_data:
            self.path = IMDB_Config.TRAIN_DATA_PATH
        else:
            self.path = IMDB_Config.TEST_DATA_PATH
        self.sentences, self.labels = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(
            IMDB_Config.TOKENIZER_NAME)
        self.encodings = self.tokenizer(
            self.sentences, padding=True, truncation=True, max_length=IMDB_Config.MAX_TOKENIZATION_LENGTH, return_tensors='pt')
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
                    sentence = line[:-1]
                    sentence = re.sub(r'<br />', '', sentence)
                    sentence = sentence.lstrip().rstrip()
                    sentence = re.sub(' +', ' ', sentence)
                    data.append(sentence)
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                sentence = line[:-1]
                sentence = re.sub(r'<br />', '', sentence)
                sentence = sentence.lstrip().rstrip()
                sentence = re.sub(' +', ' ', sentence)
                if IMDB_Config.APPLY_CLEANING:
                    sentence = clean_text(sentence)
                data.append(sentence)
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def __getitem__(self, item):
        return self.encodings['input_ids'][item], self.encodings['attention_mask'][item], self.encodings['token_type_ids'][item], self.encodings['classification_label'][item]

    def __len__(self):
        return len(self.encodings['input_ids'])


class SST2_Dataset(Dataset):
    def __init__(self, train_data=True, debug_mode=False):
        super(SST2_Dataset, self).__init__()


class AGNEWS_Dataset(Dataset):
    def __init__(self, train_data=True, debug_mode=False):
        super(AGNEWS_Dataset, self).__init__()


if __name__ == '__main__':
    path_train_pos = r'data/IMDB/aclImdb/train/pos'
    path_train_neg = r'data/IMDB/aclImdb/train/neg'
    path_test_pos = r'data/IMDB/aclImdb/test/pos'
    path_test_neg = r'data/IMDB/aclImdb/test/neg'

    with open(r'./data/IMDB/aclImdb/train.std', 'w', encoding='utf-8') as write_f:
        for file_name in [path_train_pos + '/' + file for file in os.listdir(path_train_pos)]:
            with open(file_name, 'r', encoding='utf-8') as f:
                line = f.read()
                line.strip()
                write_f.write(line + '1')
                write_f.write('\n')
        for file_name in [path_train_neg + '/' + file for file in os.listdir(path_train_neg)]:
            with open(file_name, 'r', encoding='utf-8') as f:
                line = f.read()
                line.strip()
                write_f.write(line + '0')
                write_f.write('\n')

    with open(r'./data/IMDB/aclImdb/test.std', 'w', encoding='utf-8') as write_f:
        for file_name in [path_test_pos + '/' + file for file in os.listdir(path_test_pos)]:
            with open(file_name, 'r', encoding='utf-8') as f:
                line = f.read()
                line.strip()
                write_f.write(line + '1')
                write_f.write('\n')
        for file_name in [path_test_neg + '/' + file for file in os.listdir(path_test_neg)]:
            with open(file_name, 'r', encoding='utf-8') as f:
                line = f.read()
                line.strip()
                write_f.write(line + '0')
                write_f.write('\n')
