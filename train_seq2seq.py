import copy
import os
from shutil import copyfile
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from seq2seq_config import CONFIG_PATH, OUTPUT_DIR, Seq2Seq_config, Victim_config, dataset_config
from seq2seq_dataset import IMDB_MLM_Dataset
from victim_module.victim_model import FineTunedBert
from seq2seq_model import Seq2Seq_bert
from tools import logging, get_time


def save_config(path):
    copyfile(CONFIG_PATH, path + r'/config.txt')


def build_bert_dataset():
    if Seq2Seq_config.DATASET == 'IMDB':
        train_dataset_orig = IMDB_MLM_Dataset(train_data=True,
                                                  debug_mode=Seq2Seq_config.DEBUG_MODE)
        test_dataset_orig = IMDB_MLM_Dataset(train_data=False,
                                                 debug_mode=Seq2Seq_config.DEBUG_MODE)

    train_data = DataLoader(train_dataset_orig,
                            batch_size=Seq2Seq_config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=4,
                            drop_last=True)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Seq2Seq_config.BATCH_SIZE,
                           shuffle=False,
                           num_workers=4,
                           drop_last=True)
    return train_data, test_data, train_dataset_orig.tokenizer


def train(train_data, seq2seq_model, criterion, optimizer):
    seq2seq_model.train()
    loss_mean = 0.0
    for x, masks, types, y, _, _ in train_data:
        x = x.to(Seq2Seq_config.TRAIN_DEVICE)
        masks = masks.to(Seq2Seq_config.TRAIN_DEVICE)
        types = types.to(Seq2Seq_config.TRAIN_DEVICE)
        y = y.to(Seq2Seq_config.TRAIN_DEVICE)
        y = y.reshape(-1)

        optimizer.zero_grad()

        # Teacher forcing: Feed the target as the next input
        logits = seq2seq_model(
            input_ids=x, token_type_ids=types, attention_mask=masks)

        logits = logits.reshape(-1, logits.shape[-1])

        loss = criterion(logits, y)
        loss_mean += loss.item()

        loss.backward()
        optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate(test_data, seq2seq_model, criterion, path, tokenizer, ep):
    seq2seq_model.eval()
    loss_mean = 0.0
    infer_acc = 0
    for x, masks, types, y, org_x, sentences_lenth in test_data:
        x = x.to(Seq2Seq_config.TRAIN_DEVICE)
        masks = masks.to(Seq2Seq_config.TRAIN_DEVICE)
        types = types.to(Seq2Seq_config.TRAIN_DEVICE)
        y = y.to(Seq2Seq_config.TRAIN_DEVICE)
        org_x = org_x.to(Seq2Seq_config.TRAIN_DEVICE)

        logits = seq2seq_model(
            input_ids=x, token_type_ids=types, attention_mask=masks)
        outputs_idx = logits.argmax(dim=-1)

        for i, idx in enumerate(outputs_idx):
            outputs_idx[i][0] = 101
            outputs_idx[i][sentences_lenth[i]-1] = 102
            outputs_idx[i][sentences_lenth[i]:] = 0

        infer_acc += (outputs_idx == org_x).float().sum().item() / \
            org_x.shape[0] / org_x.shape[1]

        if (ep + 1) % 5 == 0:
            with open(path, 'a') as f:
                for i in range(len(y)):
                    f.write('-------orginal sentence----------\n')
                    f.write(
                        ' '.join(tokenizer.convert_ids_to_tokens(org_x[i])) +
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

    return loss_mean / len(test_data), infer_acc / len(test_data)


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(Seq2Seq_config.CUDA_IDX))
    cur_dir = OUTPUT_DIR + '/train_seq2seq_model/' + \
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cur_models_dir = cur_dir + '/models'
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_models_dir)

    logging('Saving into directory ' + cur_dir)
    save_config(cur_dir)

    logging('preparing data...')
    train_data, test_data, tokenizer = build_bert_dataset()

    logging('init models, optimizer, criterion...')

    victim_model = FineTunedBert(pretrained_model_name=Victim_config.PRETRAINED_MODEL_NAME,
                                 num_pretrained_bert_layers=Victim_config.NUM_PRETRAINED_BERT_LAYERS,
                                 max_tokenization_length=dataset_config[
                                     Victim_config.DATASET].MAX_TOKENIZATION_LENGTH,
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
                                 device=Seq2Seq_config.TRAIN_DEVICE,
                                 fine_tuning=Victim_config.FINE_TUNING).to(Seq2Seq_config.TRAIN_DEVICE)

    victim_model.load_state_dict(torch.load(
        Victim_config.STATE_PATH, map_location=Seq2Seq_config.TRAIN_DEVICE))

    seq2seq_model = Seq2Seq_bert(pretrained_model_name=Seq2Seq_config.PRETRAINED_MODEL_NAME,
                                 num_pretrained_bert_layers=Seq2Seq_config.NUM_PRETRAINED_BERT_LAYERS,
                                 max_tokenization_length=dataset_config[
                                     Victim_config.DATASET].MAX_TOKENIZATION_LENGTH,
                                 recurrent_hidden_size=Seq2Seq_config.RECURRENT_HIDDEN_SIZE,
                                 num_recurrent_layers=Seq2Seq_config.NUM_RECURRENT_LAYERS,
                                 device=Seq2Seq_config.TRAIN_DEVICE,
                                 use_bidirectional=True,
                                 dropout_rate=Seq2Seq_config.DROPOUT_RATE,
                                 fine_tuning=Seq2Seq_config.FINE_TUNING).to(Seq2Seq_config.TRAIN_DEVICE)

    victim_bert_states_dict = victim_model.bert.state_dict()
    seq2seq_model.bert.load_state_dict(victim_bert_states_dict, strict=False)

    bert_identifiers = ['embedding', 'encoder', 'pooler']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']

    if Seq2Seq_config.FINE_TUNING:
        grouped_model_parameters = [
            {'params': [param for name, param in seq2seq_model.named_parameters()
                        if any(identifier in name for identifier in bert_identifiers) and
                        not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
             'lr': Seq2Seq_config.BERT_LEARNING_RATE,
             'betas': Seq2Seq_config.BETAS,
             'weight_decay': Seq2Seq_config.BERT_WEIGHT_DECAY,
             'eps': Seq2Seq_config.EPS},
            {'params': [param for name, param in seq2seq_model.named_parameters()
                        if any(identifier in name for identifier in bert_identifiers) and
                        any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
             'lr': Seq2Seq_config.BERT_LEARNING_RATE,
             'betas': Seq2Seq_config.BETAS,
             'weight_decay': 0.0,
             'eps': Seq2Seq_config.EPS},
            {'params': [param for name, param in seq2seq_model.named_parameters()
                        if not any(identifier in name for identifier in bert_identifiers)],
             'lr': Seq2Seq_config.CUSTOM_LEARNING_RATE,
             'betas': Seq2Seq_config.BETAS,
             'weight_decay': Seq2Seq_config.LSTM_WEIGHT_DECAY,
             'eps': Seq2Seq_config.EPS}
        ]
    else:
        grouped_model_parameters = [
            {'params': [param for name, param in seq2seq_model.named_parameters()
                        if not any(identifier in name for identifier in bert_identifiers)],
             'lr': Seq2Seq_config.CUSTOM_LEARNING_RATE,
             'betas': Seq2Seq_config.BETAS,
             'weight_decay': Seq2Seq_config.LSTM_WEIGHT_DECAY,
             'eps': Seq2Seq_config.EPS}
        ]

    optimizer = optim.AdamW(grouped_model_parameters)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(Seq2Seq_config.TRAIN_DEVICE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=3,
                                                     verbose=True, min_lr=3e-8)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < 4 else 1.0)

    logging('Start training...')
    best_acc = 0.0
    temp_path = cur_models_dir + f'/{Seq2Seq_config.DATASET}_temp_model.pt'
    for ep in range(Seq2Seq_config.NUM_EPOCHS):
        logging(f'epoch {ep+1} start train')
        train_loss = train(train_data, seq2seq_model, criterion, optimizer)
        logging(f'epoch {ep+1} start evaluate')
        indices_path = cur_dir + f'/eval_seq2seq_model_epoch_{ep+1}.log'
        evaluate_loss, infer_acc = evaluate(
            test_data, seq2seq_model, criterion, indices_path, tokenizer, ep)
        if infer_acc > best_acc:
            best_acc = infer_acc
            best_path = cur_models_dir + \
                f'/{Seq2Seq_config.DATASET}_{infer_acc:.5f}_{get_time()}.pt'
            best_state = copy.deepcopy(seq2seq_model.state_dict())

            if ep > 3 and best_state != None:
                logging(
                    f'saving best seq2seq_model acc {best_acc:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        warmup_scheduler.step(ep+1)
        scheduler.step(evaluate_loss, ep+1)

        logging(
            f'epoch {ep+1} done! train_loss {train_loss:.5f} evaluate_loss {evaluate_loss:.5f} \n'
            f'gen_acc {infer_acc:.5f} now best_acc {best_acc:.5f}')

    if best_state != None:
        logging(f'saving best seq2seq_model acc {best_acc:.5f} in {best_path}')
        torch.save(best_state, best_path)
