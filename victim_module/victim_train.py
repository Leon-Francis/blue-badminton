import copy
import os
from shutil import copyfile
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from victim_config import CONFIG_PATH, OUTPUT_DIR, Victim_Train_Config, dataset_config, IMDB_Config, SST2_Config, AGNEWS_Config
from victim_dataset import IMDB_Dataset, SST2_Dataset, AGNEWS_Dataset
from victim_model import Victim_Bert
from tools import logging, get_time



def save_config(path):
    copyfile(CONFIG_PATH, path + r'/config.txt')


def build_bert_dataset():
    if Victim_Train_Config.dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          debug_mode=Victim_Train_Config.debug_mode)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         debug_mode=Victim_Train_Config.debug_mode)
    elif Victim_Train_Config.dataset == 'SST2':
        train_dataset_orig = SST2_Dataset(train_data=True,
                                          debug_mode=Victim_Train_Config.debug_mode)
        test_dataset_orig = SST2_Dataset(train_data=False,
                                         if_attach_NE=Victim_Train_Config.if_attach_NE,
                                         debug_mode=Victim_Train_Config.debug_mode)
    elif Victim_Train_Config.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(train_data=True,
                                            debug_mode=Victim_Train_Config.debug_mode)
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           debug_mode=Victim_Train_Config.debug_mode)
    train_data = DataLoader(train_dataset_orig,
                            batch_size=Victim_Train_Config.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Victim_Train_Config.batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


def train(train_data, victim_model, criterion, optimizer):
    victim_model.train()
    loss_mean = 0.0
    for x, masks, types, y in train_data:
        x = x.to(Victim_Train_Config.train_device)
        masks = masks.to(Victim_Train_Config.train_device)
        types = types.to(Victim_Train_Config.train_device)
        y = y.to(Victim_Train_Config.train_device)
        logits = victim_model(x, masks, types)
        loss = criterion(logits, y)
        loss_mean += loss.item()
        if loss.item() > Victim_Train_Config.skip_loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate(test_data, victim_model, criterion):
    victim_model.eval()
    loss_mean = 0.0
    correct = 0
    total = 0
    for x, masks, types, y in test_data:
        x = x.to(Victim_Train_Config.train_device)
        masks = masks.to(Victim_Train_Config.train_device)
        types = types.to(Victim_Train_Config.train_device)
        y = y.to(Victim_Train_Config.train_device)
        logits = victim_model(x, masks, types)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        predicts = logits.argmax(dim=-1)
        correct += predicts.eq(y).float().sum().item()
        total += y.size()[0]

    return loss_mean / len(test_data), correct / total


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(Victim_Train_Config.cuda_idx))
    cur_dir = OUTPUT_DIR + '/train_victim_model/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cur_models_dir = cur_dir + '/models'
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_models_dir)

    logging('Saving into directory ' + cur_dir)
    save_config(cur_dir)

    logging('preparing data...')
    if Victim_Train_Config.victim_model_name == 'Bert':
        train_data, test_data = build_bert_dataset()

        logging('init models, optimizer, criterion...')
        victim_model = Victim_Bert(
            label_num=dataset_config[Victim_Train_Config.dataset].labels_num,
            linear_layer_num=Victim_Train_Config.linear_layer_num,
            dropout_rate=Victim_Train_Config.dropout_rate,
            is_fine_tuning=Victim_Train_Config.is_fine_tuning).to(
                Victim_Train_Config.train_device)

        optimizer = optim.AdamW([{
            'params': victim_model.bert_model.parameters(),
            'lr': Victim_Train_Config.Bert_lr
        }, {
            'params': victim_model.fc.parameters()
        }],
            lr=Victim_Train_Config.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.95,
                                                     patience=3,
                                                     verbose=True,
                                                     min_lr=3e-9)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                   lr_lambda=lambda ep: 1e-2
                                                   if ep < 3 else 1.0)

    criterion = nn.CrossEntropyLoss().to(Victim_Train_Config.train_device)

    logging('Start training...')
    best_acc = 0.0
    temp_path = cur_models_dir + \
        f'/{Victim_Train_Config.dataset}_{Victim_Train_Config.victim_model_name}_temp_model.pt'
    for ep in range(Victim_Train_Config.epoch):
        logging(f'epoch {ep} start train')
        train_loss = train(train_data, victim_model, criterion, optimizer)
        logging(f'epoch {ep} start evaluate')
        evaluate_loss, acc = evaluate(test_data, victim_model, criterion)
        if acc > best_acc:
            best_acc = acc
            best_path = cur_models_dir + \
                f'/{Victim_Train_Config.dataset}_{Victim_Train_Config.victim_model_name}_{acc:.5f}_{get_time()}.pt'
            best_state = copy.deepcopy(victim_model.state_dict())

            if ep > 3 and best_acc > Victim_Train_Config.save_acc_limit and best_state != None:
                logging(f'saving best model acc {best_acc:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        if ep < 4:
            warmup_scheduler.step(ep)
        else:
            scheduler.step(evaluate_loss, epoch=ep)

        logging(
            f'epoch {ep} done! train_loss {train_loss:.5f} evaluate_loss {evaluate_loss:.5f} \n'
            f'acc {acc:.5f} now best_acc {best_acc:.5f}')

    if best_acc > Victim_Train_Config.save_acc_limit and best_state != None:
        logging(f'saving best model acc {best_acc:.5f} in {best_path}')
        torch.save(best_state, best_path)
