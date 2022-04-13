import torch
from victim_module.victim_config import dataset_config, Victim_config
BERT_HIDDEN_SIZE = 768
BERT_VOCAB_SIZE = 30522

CONFIG_PATH = 'seq2seq_config.py'
OUTPUT_DIR = 'outputs'


class Seq2Seq_config():
    CUDA_IDX = 5
    TRAIN_DEVICE = torch.device('cuda:' + str(CUDA_IDX))
    DATASET = 'IMDB'
    DEBUG_MODE = False
    FINE_TUNING = True

    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    if FINE_TUNING:
        BATCH_SIZE = 16

    BERT_LEARNING_RATE = 3e-5
    CUSTOM_LEARNING_RATE = 1e-3
    BETAS = (0.9, 0.999)
    BERT_WEIGHT_DECAY = 0.01
    LSTM_WEIGHT_DECAY = 0.1
    EPS = 1e-8

    PRETRAINED_MODEL_NAME = 'bert-base-cased'
    NUM_PRETRAINED_BERT_LAYERS = 12
    MAX_TOKENIZATION_LENGTH = 512
    RECURRENT_HIDDEN_SIZE = 768
    NUM_RECURRENT_LAYERS = 3
    DROPOUT_RATE = 0.2

    STATE_PATH = 'outputs/train_seq2seq_model/2022-04-12_12:45:38/models/IMDB_temp_model_0.85815.pt'