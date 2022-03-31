import torch
BERT_VOCAB_SIZE = 30522
CONFIG_PATH = './victim_module/victim_config.py'
OUTPUT_DIR = './victim_module/outputs'


class Victim_config():
    CUDA_IDX = 4
    TRAIN_DEVICE = torch.device('cuda:' + str(CUDA_IDX))
    DATASET = 'IMDB'
    DEBUG_MODE = False

    NUM_EPOCHS = 10
    BATCH_SIZE = 32

    BERT_LEARNING_RATE = 3e-5
    CUSTOM_LEARNING_RATE = 1e-3
    BETAS = (0.9, 0.999)
    BERT_WEIGHT_DECAY = 0.01
    EPS = 1e-8

    PRETRAINED_MODEL_NAME = 'bert-base-cased'
    NUM_PRETRAINED_BERT_LAYERS = 12
    NUM_CLASSES = 2
    TOP_DOWN = True
    NUM_RECURRENT_LAYERS = 3
    HIDDEN_SIZE = 128
    REINITIALIZE_POOLER_PARAMETERS = False
    USE_BIDIRECTIONAL = True
    DROPOUT_RATE = 0.2
    AGGREGATE_ON_CLS_TOKEN = True
    CONCATENATE_HIDDEN_STATES = False
    FINE_TUNING = False

    STATE_PATH = 'victim_module/outputs/train_victim_model/2022-03-30_07:11:51/models/IMDB_0.91628_03-30-09-19.pt'


class IMDB_Config():
    """
    0 for negative
    1 for positive
    """
    TRAIN_DATA_PATH = r'./data/IMDB/aclImdb/train.std'
    TEST_DATA_PATH = r'./data/IMDB/aclImdb/test.std'
    LABEL_NUM = 2
    TOKENIZER_NAME = 'bert-base-cased'
    MAX_TOKENIZATION_LENGTH = 512
    VOCAB_SIZE = BERT_VOCAB_SIZE
    AGGREGATE_ON_CLS_TOKEN = True
    CONCATENATE_HIDDEN_STATES = False
    FINE_TUNING = False
    APPLY_CLEANING = False


class SST2_Config():
    train_data_path = r'./data/SST2/train.std'
    test_data_path = r'./data/SST2/test.std'
    labels_num = 2
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 20
    vocab_size = BERT_VOCAB_SIZE


class AGNEWS_Config():
    train_data_path = r'./data/AGNEWS/train.std'
    test_data_path = r'./data/AGNEWS/test.std'
    labels_num = 4
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 50
    vocab_size = BERT_VOCAB_SIZE


class SNLI_Config():
    train_data_path = r'./data/SNLI/train.txt'
    test_data_path = r'./data/SNLI/test.txt'
    sentences_data_path = r'./data/SNLI/sentences.txt'
    labels_num = 3
    label_classes = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 15
    vocab_size = BERT_VOCAB_SIZE


dataset_config = {'IMDB': IMDB_Config, 'SST2': SST2_Config,
                  'AGNEWS': AGNEWS_Config, 'SNLI': SNLI_Config}
