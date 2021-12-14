import copy
import os
from shutil import copyfile
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from decoder_config import CONFIG_PATH, OUTPUT_DIR, Seq2Seq_Train_Config, Victim_Train_Config, dataset_config
from dataset import IMDB_Seq2Seq_Dataset
from victim_module.victim_model import Victim_Bert
from decoder_model import Decoder_LSTM
from tools import logging, get_time


def save_config(path):
    copyfile(CONFIG_PATH, path + r'/config.txt')


def build_bert_dataset():
    if Seq2Seq_Train_Config.dataset == 'IMDB':
        train_dataset_orig = IMDB_Seq2Seq_Dataset(train_data=True,
                                                  debug_mode=Seq2Seq_Train_Config.debug_mode)
        test_dataset_orig = IMDB_Seq2Seq_Dataset(train_data=False,
                                                 debug_mode=Seq2Seq_Train_Config.debug_mode)
    elif Seq2Seq_Train_Config.dataset == 'SST2':
        pass
    elif Seq2Seq_Train_Config.dataset == 'AGNEWS':
        pass            

    train_data = DataLoader(train_dataset_orig,
                            batch_size=Seq2Seq_Train_Config.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Seq2Seq_Train_Config.batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data, train_dataset_orig.tokenizer


def train(train_data, victim_model, decoder_model, criterion, optimizer):
    victim_model.train()
    loss_mean = 0.0
    for x, masks, types, y in train_data:
        x = x.to(Seq2Seq_Train_Config.train_device)
        masks = masks.to(Seq2Seq_Train_Config.train_device)
        types = types.to(Seq2Seq_Train_Config.train_device)
        y = y.to(Seq2Seq_Train_Config.train_device)
        hidden = victim_model.embedding(x, masks, types)
        logits = decoder_model(x, masks, types, hidden)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = criterion(logits, y)
        loss_mean += loss.item()
        if loss.item() > Seq2Seq_Train_Config.skip_loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate(test_data, victim_model, decoder_model, criterion, path, tokenizer, ep):
    victim_model.eval()
    loss_mean = 0.0
    correct = 0
    total = 0
    for x, masks, types, y in test_data:
        x = x.to(Seq2Seq_Train_Config.train_device)
        masks = masks.to(Seq2Seq_Train_Config.train_device)
        types = types.to(Seq2Seq_Train_Config.train_device)
        y = y.to(Seq2Seq_Train_Config.train_device)
        hidden = victim_model.embedding(x, masks, types)
        logits = decoder_model(x, masks, types, hidden)
        
        # outputs_idx: [batch, sen_len]
        outputs_idx = logits.argmax(dim=-1)
        correct += (outputs_idx == y).float().sum().item()
        total += y.shape[0] * y.shape[1]

        if (ep + 1) % 5 == 0:
            with open(path, 'a') as f:
                for i in range(len(y)):
                    f.write('-------orginal sentence----------\n')
                    f.write(
                        ' '.join(tokenizer.convert_ids_to_tokens(y[i])) +
                        '\n')
                    f.write(
                        '-------sentence -> encoder -> decoder----------\n'
                    )
                    f.write(' '.join(
                        tokenizer.convert_ids_to_tokens(outputs_idx[i])) +
                            '\n' * 2)

        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = criterion(logits, y)
        loss_mean += loss.item()

    return loss_mean / len(test_data), correct / total


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(Seq2Seq_Train_Config.cuda_idx))
    cur_dir = OUTPUT_DIR + '/train_decoder_model/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cur_models_dir = cur_dir + '/models'
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_models_dir)

    logging('Saving into directory ' + cur_dir)
    save_config(cur_dir)

    logging('preparing data...')
    if Seq2Seq_Train_Config.decoder_model_name == 'Bert':
        train_data, test_data, tokenizer = build_bert_dataset()

        logging('init models, optimizer, criterion...')

        victim_model = Victim_Bert(
            label_num=dataset_config[Victim_Train_Config.dataset].labels_num,
            linear_layer_num=Victim_Train_Config.linear_layer_num,
            dropout_rate=Victim_Train_Config.dropout_rate,
            is_fine_tuning=Victim_Train_Config.is_fine_tuning).to(
                Seq2Seq_Train_Config.train_device)

        victim_model.load_state_dict(torch.load('victim_module/outputs/train_victim_model/2021-12-13_17:42:10/models/IMDB_Bert_temp_model.pt', 
                                                map_location=Seq2Seq_Train_Config.train_device))

        decoder_model = Decoder_LSTM(
            nhidden=Seq2Seq_Train_Config.nhidden,
            num_layers=Seq2Seq_Train_Config.num_layers,
            dropout=Seq2Seq_Train_Config.dropout,
            fine_tuning=Seq2Seq_Train_Config.fine_tuning
        ).to(Seq2Seq_Train_Config.train_device)

        optimizer = optim.AdamW([{
            'params': decoder_model.bert_model.parameters(),
            'lr': Seq2Seq_Train_Config.Bert_lr
        }, {
            'params': decoder_model.decoder.parameters()
        }, {
            'params': decoder_model.linear_decoder.parameters()
        }],
            lr=Seq2Seq_Train_Config.lr,
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

    criterion = nn.CrossEntropyLoss().to(Seq2Seq_Train_Config.train_device)

    logging('Start training...')
    best_acc = 0.0
    temp_path = cur_models_dir + \
        f'/{Seq2Seq_Train_Config.dataset}_{Seq2Seq_Train_Config.decoder_model_name}_temp_model.pt'
    for ep in range(Seq2Seq_Train_Config.epoch):
        logging(f'epoch {ep} start train')
        train_loss = train(train_data, victim_model, decoder_model, criterion, optimizer)
        logging(f'epoch {ep} start evaluate')
        indices_path = cur_dir + f'/eval_seq2seq_model_epoch_{ep}.log'
        evaluate_loss, acc = evaluate(test_data, victim_model, decoder_model, criterion, indices_path, tokenizer, ep)
        if acc > best_acc:
            best_acc = acc
            best_path = cur_models_dir + \
                f'/{Seq2Seq_Train_Config.dataset}_{Seq2Seq_Train_Config.decoder_model_name}_{acc:.5f}_{get_time()}.pt'
            best_state = copy.deepcopy(decoder_model.state_dict())

            if ep > 3 and best_acc > Seq2Seq_Train_Config.save_acc_limit and best_state != None:
                logging(f'saving best model acc {best_acc:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        if ep < 4:
            warmup_scheduler.step(ep)
        else:
            scheduler.step(evaluate_loss, epoch=ep)

        logging(
            f'epoch {ep} done! train_loss {train_loss:.5f} evaluate_loss {evaluate_loss:.5f} \n'
            f'acc {acc:.5f} now best_acc {best_acc:.5f}')

    if best_acc > Seq2Seq_Train_Config.save_acc_limit and best_state != None:
        logging(f'saving best model acc {best_acc:.5f} in {best_path}')
        torch.save(best_state, best_path)
