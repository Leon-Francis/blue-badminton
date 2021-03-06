import os
import re
from victim_module.victim_config import IMDB_Config
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


class IMDB_Seq2Seq_Dataset(Dataset):
    """
    idx = [CLS] + seq
    mask = mask of idx
    type = type of idx
    label_idx = seq + [SEP]
    """
    def __init__(self, train_data=True, debug_mode=False):
        super(IMDB_Seq2Seq_Dataset, self).__init__()
        if train_data:
            self.path = IMDB_Config.TRAIN_DATA_PATH
        else:
            self.path = IMDB_Config.TEST_DATA_PATH
        self.sentences, self.labels = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(
            IMDB_Config.TOKENIZER_NAME)
        
        self.encodings_sentence = []
        self.decodings_sentence = []

        for sen in self.sentences:
            self.encodings_sentence.append('[CLS] ' + sen)
            self.decodings_sentence.append(sen + ' [SEP]')

        self.encodings = self.tokenizer(self.encodings_sentence,
                                        add_special_tokens=False,
                                        padding=True,
                                        truncation=True,
                                        max_length=IMDB_Config.MAX_TOKENIZATION_LENGTH,
                                        return_tensors='pt')

        self.decodings = self.tokenizer(self.decodings_sentence,
                                        add_special_tokens=False,
                                        padding=True,
                                        truncation=True,
                                        max_length=IMDB_Config.MAX_TOKENIZATION_LENGTH,
                                        return_tensors='pt')

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
        return self.encodings['input_ids'][item], self.encodings[
            'attention_mask'][item], self.encodings['token_type_ids'][
                item], self.decodings['input_ids'][item]

    def __len__(self):
        return len(self.encodings['input_ids'])


if __name__ == '__main__':
    dataset = IMDB_Seq2Seq_Dataset(train_data=True, debug_mode=True)
    pass