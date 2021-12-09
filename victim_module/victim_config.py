import torch
BERT_VOCAB_SIZE = 30522
CONFIG_PATH = './victim_module/victim_config.py'
OUTPUT_DIR = './victim_module/outputs'


class Victim_Train_Config():
    cuda_idx = 1
    train_device = torch.device('cuda:' + str(cuda_idx))
    dataset = 'IMDB'
    victim_model_name = 'Bert'
    batch_size = 16
    epoch = 100
    save_acc_limit = 0.80

    debug_mode = False

    linear_layer_num = 1
    dropout_rate = 0.5
    is_fine_tuning = True

    Bert_lr = 1e-5
    lr = 3e-4
    skip_loss = 0


class IMDB_Config():
    train_data_path = r'./dataset/IMDB/aclImdb/train.std'
    test_data_path = r'./dataset/IMDB/aclImdb/test.std'
    labels_num = 2
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 230
    vocab_size = BERT_VOCAB_SIZE


class SST2_Config():
    train_data_path = r'./dataset/SST2/train.std'
    test_data_path = r'./dataset/SST2/test.std'
    labels_num = 2
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 20
    vocab_size = BERT_VOCAB_SIZE


class AGNEWS_Config():
    train_data_path = r'./dataset/AGNEWS/train.std'
    test_data_path = r'./dataset/AGNEWS/test.std'
    labels_num = 4
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 50
    vocab_size = BERT_VOCAB_SIZE


class SNLI_Config():
    train_data_path = r'./dataset/SNLI/train.txt'
    test_data_path = r'./dataset/SNLI/test.txt'
    sentences_data_path = r'./dataset/SNLI/sentences.txt'
    labels_num = 3
    label_classes = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    tokenizer_name = 'bert-base-cased'
    remove_stop_words = False
    sen_len = 15
    vocab_size = BERT_VOCAB_SIZE


dataset_config = {'IMDB': IMDB_Config, 'SST2': SST2_Config, 'AGNEWS': AGNEWS_Config, 'SNLI': SNLI_Config}
