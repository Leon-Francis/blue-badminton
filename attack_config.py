from calendar import EPOCH
import torch
from victim_module.victim_config import dataset_config, Victim_config
CONFIG_PATH = 'attack_config.py'
OUTPUT_DIR = 'attack_outputs'
class Attack_config():
    CUDA_IDX = 3
    TRAIN_DEVICE = torch.device('cuda:' + str(CUDA_IDX))
    DATASET = 'IMDB'
    DEBUG_MODE = False
    ONLY_EVAL = False
    SEQ2SEQ_TRAIN_WITH_ADVERSARY = True
    
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    
    MASK_PROB = 0.3   

    BERT_LEARNING_RATE = 3e-5
    BETAS = (0.9, 0.999)
    BERT_WEIGHT_DECAY = 0.01
    EPS = 1e-8