import torch
from victim_module.victim_config import dataset_config, Victim_config
CONFIG_PATH = 'attack_config.py'
OUTPUT_DIR = 'attack_outputs'
class Attack_config():
    CUDA_IDX = 2
    TRAIN_DEVICE = torch.device('cuda:' + str(CUDA_IDX))
    DATASET = 'IMDB'
    DEBUG_MODE = False
    
    BATCH_SIZE = 32
    
    MASK_PROB = 0.3
    