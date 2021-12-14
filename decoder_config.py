import torch
from victim_module.victim_config import dataset_config, Victim_Train_Config
BERT_HIDDEN_SIZE = 768
BERT_VOCAB_SIZE = 30522

CONFIG_PATH = './decoder_config.py'
OUTPUT_DIR = './outputs'

class Seq2Seq_Train_Config():
    cuda_idx = 0
    train_device = torch.device('cuda:' + str(cuda_idx))
    dataset = 'IMDB'
    decoder_model_name = 'Bert'

    batch_size = 8
    epoch = 5
    save_acc_limit = 0.01

    debug_mode = True

    nhidden = 768
    num_layers = 3
    dropout = 0.3
    fine_tuning = True

    Bert_lr = 5e-6
    lr = 1e-4
    skip_loss = 0