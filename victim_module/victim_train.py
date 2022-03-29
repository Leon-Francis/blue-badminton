import copy
import os
from shutil import copyfile
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from victim_config import CONFIG_PATH, OUTPUT_DIR, Victim_config, dataset_config
from victim_dataset import IMDB_Dataset, SST2_Dataset, AGNEWS_Dataset
from victim_model import FineTunedBert
from tools import logging, get_time

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def save_config(path):
    copyfile(CONFIG_PATH, path + r'/config.txt')


def build_bert_dataset():
    if Victim_config.DATASET == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          debug_mode=Victim_config.DEBUG_MODE)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         debug_mode=Victim_config.DEBUG_MODE)

    train_data = DataLoader(train_dataset_orig,
                            batch_size=Victim_config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Victim_config.BATCH_SIZE,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


def train(train_data, victim_model, criterion, optimizer):
    victim_model.train()
    loss_mean = 0.0
    for x, masks, types, y in train_data:
        x = x.to(Victim_config.TRAIN_DEVICE)
        masks = masks.to(Victim_config.TRAIN_DEVICE)
        types = types.to(Victim_config.TRAIN_DEVICE)
        y = y.to(Victim_config.TRAIN_DEVICE)
        logits = victim_model(input_ids=x,
                              token_type_ids=types,
                              attention_mask=masks)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate(test_data, victim_model, criterion):

    def binary_accuracy(y_pred, y_true):
        """Function to calculate binary accuracy per batch"""
        y_pred_max = torch.argmax(y_pred, dim=-1)
        correct_pred = (y_pred_max == y_true).float()
        acc = correct_pred.sum() / len(correct_pred)
        return acc

    victim_model.eval()
    loss_mean = 0.0
    acc_mean = 0.0

    for x, masks, types, y in test_data:
        x = x.to(Victim_config.TRAIN_DEVICE)
        masks = masks.to(Victim_config.TRAIN_DEVICE)
        types = types.to(Victim_config.TRAIN_DEVICE)
        y = y.to(Victim_config.TRAIN_DEVICE)
        logits = victim_model(input_ids=x,
                              token_type_ids=types,
                              attention_mask=masks)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        acc = binary_accuracy(logits, y)
        acc_mean += acc.item()

    return loss_mean / len(test_data), acc_mean / len(test_data)


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(Victim_config.CUDA_IDX))
    cur_dir = OUTPUT_DIR + '/train_victim_model/' + \
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cur_models_dir = cur_dir + '/models'
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_models_dir)

    logging('Saving into directory ' + cur_dir)
    save_config(cur_dir)

    logging('preparing data...')
    train_data, test_data = build_bert_dataset()

    logging('init models, optimizer, criterion...')
    model = FineTunedBert(pretrained_model_name=Victim_config.PRETRAINED_MODEL_NAME,
                          num_pretrained_bert_layers=Victim_config.NUM_PRETRAINED_BERT_LAYERS,
                          max_tokenization_length=dataset_config[Victim_config.DATASET].MAX_TOKENIZATION_LENGTH,
                          num_classes=Victim_config.NUM_CLASSES,
                          top_down=Victim_config.TOP_DOWN,
                          num_recurrent_layers=Victim_config.NUM_RECURRENT_LAYERS,
                          use_bidirectional=Victim_config.USE_BIDIRECTIONAL,
                          hidden_size=Victim_config.HIDDEN_SIZE,
                          reinitialize_pooler_parameters=Victim_config.REINITIALIZE_POOLER_PARAMETERS,
                          dropout_rate=Victim_config.DROPOUT_RATE,
                          aggregate_on_cls_token=Victim_config.AGGREGATE_ON_CLS_TOKEN,
                          concatenate_hidden_states=Victim_config.CONCATENATE_HIDDEN_STATES,
                          use_gpu=True if torch.cuda.is_available() else False,
                          device=Victim_config.TRAIN_DEVICE,
                          fine_tuning=Victim_config.FINE_TUNING).to(Victim_config.TRAIN_DEVICE)

    criterion = nn.CrossEntropyLoss().to(Victim_config.TRAIN_DEVICE)

    # Define identifiers & group model parameters accordingly (check README.md for the intuition)
    bert_identifiers = ['embedding', 'encoder', 'pooler']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
    if Victim_config.FINE_TUNING:
        grouped_model_parameters = [
            {'params': [param for name, param in model.named_parameters()
                        if any(identifier in name for identifier in bert_identifiers) and
                        not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
             'lr': Victim_config.BERT_LEARNING_RATE,
             'betas': Victim_config.BETAS,
             'weight_decay': Victim_config.BERT_WEIGHT_DECAY,
             'eps': Victim_config.EPS},
            {'params': [param for name, param in model.named_parameters()
                        if any(identifier in name for identifier in bert_identifiers) and
                        any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
             'lr': Victim_config.BERT_LEARNING_RATE,
             'betas': Victim_config.BETAS,
             'weight_decay': 0.0,
             'eps': Victim_config.EPS},
            {'params': [param for name, param in model.named_parameters()
                        if not any(identifier in name for identifier in bert_identifiers)],
             'lr': Victim_config.CUSTOM_LEARNING_RATE,
             'betas': Victim_config.BETAS,
             'weight_decay': 0.0,
             'eps': Victim_config.EPS}
        ]
    else:
        grouped_model_parameters = [
            {'params': [param for name, param in model.named_parameters()
                        if not any(identifier in name for identifier in bert_identifiers)],
             'lr': Victim_config.CUSTOM_LEARNING_RATE,
             'betas': Victim_config.BETAS,
             'weight_decay': 0.0,
             'eps': Victim_config.EPS}
        ]

    optimizer = optim.AdamW(grouped_model_parameters)

    logging('Start training...')
    best_acc = 0.0
    temp_path = cur_models_dir + \
        f'/{Victim_config.DATASET}_temp_model.pt'
    for ep in range(Victim_config.NUM_EPOCHS):
        logging(f'epoch {ep} start train')
        train_loss = train(train_data, model, criterion, optimizer)
        logging(f'epoch {ep} start evaluate')
        evaluate_loss, acc = evaluate(test_data, model, criterion)
        if acc > best_acc:
            best_acc = acc
            best_path = cur_models_dir + \
                f'/{Victim_config.DATASET}_{acc:.5f}_{get_time()}.pt'
            best_state = copy.deepcopy(model.state_dict())

            if ep > 3 and best_state != None:
                logging(f'saving best model acc {best_acc:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        logging(
            f'epoch {ep} done! train_loss {train_loss:.5f} evaluate_loss {evaluate_loss:.5f} \n'
            f'acc {acc:.5f} now best_acc {best_acc:.5f}')

    if best_state != None:
        logging(f'saving best model acc {best_acc:.5f} in {best_path}')
        torch.save(best_state, best_path)
