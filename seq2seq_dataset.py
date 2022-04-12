import os
import re
import random
import torch
from victim_module.victim_config import IMDB_Config, BERT_VOCAB_SIZE
from torch.utils.data import Dataset
from tools import logging
from transformers import BertTokenizer
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


class IMDB_MLM_Dataset(Dataset):
    """
    idx = [CLS] + seq
    mask = mask of idx
    type = type of idx
    label_idx = seq + [SEP]
    """

    def __init__(self, train_data=True, debug_mode=False):
        super(IMDB_MLM_Dataset, self).__init__()
        if train_data:
            self.path = IMDB_Config.TRAIN_DATA_PATH
        else:
            self.path = IMDB_Config.TEST_DATA_PATH
        self.sentences, self.labels = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained(
            IMDB_Config.TOKENIZER_NAME)

        self.encoding_idxs = []
        self.encoding_attention_mask = []
        self.encoding_token_type_ids = []
        self.decoding_idxs = []
        self.orgin_idxs = []
        self.sentences_lenth = []

        self.data2idx()

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

    def data2idx(self):
        logging(f'{self.path} in data2idxs')
        for sen in self.sentences:
            data_tokens = self.tokenizer.tokenize(
                sen)[:IMDB_Config.MAX_TOKENIZATION_LENGTH - 2]
            ogrin_idx = [101] + self.tokenizer.convert_tokens_to_ids(data_tokens) + [102]
            encoding_idx, decoding_idx = self.random_word(data_tokens)
            encoding_idx = [101] + encoding_idx + [102]
            decoding_idx = [0] + decoding_idx + [0]
            self.sentences_lenth.append(len(encoding_idx))

            if len(encoding_idx) < IMDB_Config.MAX_TOKENIZATION_LENGTH:
                self.encoding_idxs.append(
                    encoding_idx + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(encoding_idx)))
                self.encoding_attention_mask.append(
                    [1] * len(encoding_idx) + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(encoding_idx)))
                self.decoding_idxs.append(
                    decoding_idx + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(decoding_idx)))
                self.orgin_idxs.append(
                    ogrin_idx + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(ogrin_idx)))
            else:
                self.encoding_idxs.append(encoding_idx)
                self.decoding_idxs.append(decoding_idx)
                self.encoding_attention_mask.append(
                    [1] * IMDB_Config.MAX_TOKENIZATION_LENGTH)
                self.orgin_idxs.append(ogrin_idx)
            self.encoding_token_type_ids.append(
                [0] * IMDB_Config.MAX_TOKENIZATION_LENGTH)

    def __getitem__(self, item):
        return torch.tensor(self.encoding_idxs[item]), \
            torch.tensor(self.encoding_attention_mask[item]), \
                torch.tensor(self.encoding_token_type_ids[item]), \
                    torch.tensor(self.decoding_idxs[item]), \
                        torch.tensor(self.orgin_idxs[item]), \
                            self.sentences_lenth[item]

    def __len__(self):
        return len(self.encoding_idxs)

    def random_word(self, data_tokens):
        output_label = []

        for i, token in enumerate(data_tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    data_tokens[i] = 103

                # 10% randomly change token to random token
                elif prob < 0.9:
                    data_tokens[i] = random.randrange(
                        self.tokenizer.vocab_size)

                # 10% randomly change token to current token
                else:
                    data_tokens[i] = self.tokenizer.convert_tokens_to_ids(
                        token)

                output_label.append(
                    self.tokenizer.convert_tokens_to_ids(token))

            else:
                data_tokens[i] = self.tokenizer.convert_tokens_to_ids(token)
                output_label.append(0)

        return data_tokens, output_label


if __name__ == '__main__':
    dataset = IMDB_MLM_Dataset(train_data=True, debug_mode=True)
    pass
