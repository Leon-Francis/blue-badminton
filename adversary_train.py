import os
import re
import random
import copy
import torch
from datetime import datetime
from shutil import copyfile
from victim_module.victim_config import IMDB_Config
from attack_config import Attack_config, OUTPUT_DIR, CONFIG_PATH, Victim_config, dataset_config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from tools import logging
from transformers import BertTokenizer
from victim_module.victim_model import FineTunedBert
from seq2seq_config import Seq2Seq_config
from seq2seq_model import Seq2Seq_bert
from tools import logging, get_time


class IMDB_MLM_Dataset(Dataset):
    """
    idx = [CLS] + seq
    mask = mask of idx
    type = type of idx
    label_idx = seq + [SEP]
    """

    def __init__(self, train_data=True, debug_mode=False):
        super(IMDB_MLM_Dataset, self).__init__()
        if train_data:
            self.path = IMDB_Config.TRAIN_DATA_PATH
        else:
            self.path = IMDB_Config.TEST_DATA_PATH
        self.sentences, self.labels = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained(
            IMDB_Config.TOKENIZER_NAME)

        self.encoding_idxs = []
        self.encoding_attention_mask = []
        self.encoding_token_type_ids = []
        self.decoding_idxs = []
        self.orgin_idxs = []
        self.sentences_lenth = []

        self.data2idx()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 250
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    sentence = line[:-1]
                    sentence = re.sub(r'<br />', '', sentence)
                    sentence = sentence.lstrip().rstrip()
                    sentence = re.sub(' +', ' ', sentence)
                    data.append(sentence)
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                sentence = line[:-1]
                sentence = re.sub(r'<br />', '', sentence)
                sentence = sentence.lstrip().rstrip()
                sentence = re.sub(' +', ' ', sentence)
                data.append(sentence)
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2idx(self):
        logging(f'{self.path} in data2idxs')
        for sen in self.sentences:
            data_tokens = self.tokenizer.tokenize(
                sen)[:IMDB_Config.MAX_TOKENIZATION_LENGTH - 2]
            ogrin_idx = [101] + \
                self.tokenizer.convert_tokens_to_ids(data_tokens) + [102]
            encoding_idx, decoding_idx = self.random_word(data_tokens)
            encoding_idx = [101] + encoding_idx + [102]
            decoding_idx = [0] + decoding_idx + [0]
            self.sentences_lenth.append(len(encoding_idx))

            if len(encoding_idx) < IMDB_Config.MAX_TOKENIZATION_LENGTH:
                self.encoding_idxs.append(
                    encoding_idx + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(encoding_idx)))
                self.encoding_attention_mask.append(
                    [1] * len(encoding_idx) + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(encoding_idx)))
                self.decoding_idxs.append(
                    decoding_idx + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(decoding_idx)))
                self.orgin_idxs.append(
                    ogrin_idx + [0] * (IMDB_Config.MAX_TOKENIZATION_LENGTH - len(ogrin_idx)))
            else:
                self.encoding_idxs.append(encoding_idx)
                self.decoding_idxs.append(decoding_idx)
                self.encoding_attention_mask.append(
                    [1] * IMDB_Config.MAX_TOKENIZATION_LENGTH)
                self.orgin_idxs.append(ogrin_idx)
            self.encoding_token_type_ids.append(
                [0] * IMDB_Config.MAX_TOKENIZATION_LENGTH)

    def __getitem__(self, item):
        return torch.tensor(self.encoding_idxs[item]), \
            torch.tensor(self.encoding_attention_mask[item]), \
            torch.tensor(self.encoding_token_type_ids[item]), \
            torch.tensor(self.decoding_idxs[item]), \
            torch.tensor(self.orgin_idxs[item]), \
            torch.tensor(self.labels[item]), \
            self.sentences_lenth[item]

    def __len__(self):
        return len(self.encoding_idxs)

    def random_word(self, data_tokens):
        output_label = []

        for i, token in enumerate(data_tokens):
            prob = random.random()
            if prob < Attack_config.MASK_PROB:

                data_tokens[i] = 103

                output_label.append(
                    self.tokenizer.convert_tokens_to_ids(token))

            else:
                data_tokens[i] = self.tokenizer.convert_tokens_to_ids(token)
                output_label.append(0)

        return data_tokens, output_label


def save_config(path):
    copyfile(CONFIG_PATH, path + r'/config.txt')


def build_bert_dataset():
    if Attack_config.DATASET == 'IMDB':
        train_dataset_orig = IMDB_MLM_Dataset(train_data=True,
                                              debug_mode=Attack_config.DEBUG_MODE)
        test_dataset_orig = IMDB_MLM_Dataset(train_data=False,
                                             debug_mode=Attack_config.DEBUG_MODE)

    train_data = DataLoader(train_dataset_orig,
                            batch_size=Attack_config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=4,
                            drop_last=True)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Attack_config.BATCH_SIZE,
                           shuffle=False,
                           num_workers=4,
                           drop_last=True)
    return train_data, test_data, train_dataset_orig.tokenizer


def adversary_train(train_data, seq2seq_model, victim_model, criterion_classification, criterion_seq2seq, optimizer):
    victim_model.train()
    seq2seq_model.train()
    loss_mean = 0.0
    for x, masks, types, y, _, label, _ in test_data:
        x = x.to(Attack_config.TRAIN_DEVICE)
        masks = masks.to(Attack_config.TRAIN_DEVICE)
        types = types.to(Attack_config.TRAIN_DEVICE)
        y = y.to(Attack_config.TRAIN_DEVICE)
        label = label.to(Attack_config.TRAIN_DEVICE)

        optimizer.zero_grad()

        sequence_output, pooled_output = seq2seq_model.encode(input_ids=x,
                                                              token_type_ids=types,
                                                              attention_mask=masks)

        logits = victim_model.forward_from_hidden(
            masks, sequence_output, pooled_output)

        loss = criterion_classification(logits, label)
        loss *= -1

        if Attack_config.SEQ2SEQ_TRAIN_WITH_ADVERSARY:
            y = y.reshape(-1)
            logits = seq2seq_model(
                input_ids=x, token_type_ids=types, attention_mask=masks)

            logits = logits.reshape(-1, logits.shape[-1])

            loss += criterion_seq2seq(logits, y)

        loss.backward()
        loss_mean += loss.item()
        optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate_attack(test_data, seq2seq_model, victim_model, path, tokenizer, ep):
    seq2seq_model.eval()
    victim_model.eval()
    attack_num = 0
    attack_succ = 0
    org_predict_acc = 0
    attacked_predict_acc = 0
    for x, masks, types, _, org_x, label, sentences_lenth in test_data:
        x = x.to(Attack_config.TRAIN_DEVICE)
        masks = masks.to(Attack_config.TRAIN_DEVICE)
        types = types.to(Attack_config.TRAIN_DEVICE)
        org_x = org_x.to(Attack_config.TRAIN_DEVICE)
        label = label.to(Attack_config.TRAIN_DEVICE)

        victim_logits = victim_model(
            input_ids=org_x, token_type_ids=types, attention_mask=masks)
        victim_predicts = victim_logits.argmax(dim=-1)

        skiped = label != victim_predicts
        org_predict_acc += (len(label) -
                            skiped.float().sum().item()) / len(label)

        seq2seq_logits = seq2seq_model(
            input_ids=x, token_type_ids=types, attention_mask=masks)
        outputs_idx = seq2seq_logits.argmax(dim=-1)

        for i, idx in enumerate(outputs_idx):
            outputs_idx[i][0] = 101
            outputs_idx[i][sentences_lenth[i]-1] = 102
            outputs_idx[i][sentences_lenth[i]:] = 0

        attacked_logits = victim_model(
            input_ids=outputs_idx, token_type_ids=types, attention_mask=masks)
        attacked_predict = attacked_logits.argmax(dim=-1)

        successed = label != attacked_predict
        attacked_predict_acc += (len(label) -
                                 successed.float().sum().item()) / len(label)

        for i in range(len(org_x)):
            if skiped[i].item():
                continue

            attack_num += 1

            if successed[i].item():
                attack_succ += 1

        if ep == 0 or (ep+1) % 5 == 0:
            with open(path, 'a') as f:
                for i in range(len(org_x)):
                    if skiped[i].item():
                        continue

                    f.write('-------orginal sentence----------\n')
                    f.write(
                        ' '.join(tokenizer.convert_ids_to_tokens(org_x[i])) +
                        '\n')
                    f.write('-------attacked sentence---------\n')
                    f.write(' '.join(
                        tokenizer.convert_ids_to_tokens(outputs_idx[i])) +
                        '\n')

                    if successed[i].item():
                        f.write(f'attack result: True\n\n')
                    else:
                        f.write(f'attack result: False\n\n')

    return attack_succ / attack_num, org_predict_acc / len(test_data), attacked_predict_acc / len(test_data)


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(Attack_config.CUDA_IDX))
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

    logging('init models')
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
                                 device=Attack_config.TRAIN_DEVICE,
                                 fine_tuning=False).to(Attack_config.TRAIN_DEVICE)

    victim_model.load_state_dict(torch.load(
        Victim_config.STATE_PATH, map_location=Attack_config.TRAIN_DEVICE))

    seq2seq_model = Seq2Seq_bert(pretrained_model_name=Seq2Seq_config.PRETRAINED_MODEL_NAME,
                                 num_pretrained_bert_layers=Seq2Seq_config.NUM_PRETRAINED_BERT_LAYERS,
                                 max_tokenization_length=dataset_config[
                                     Victim_config.DATASET].MAX_TOKENIZATION_LENGTH,
                                 recurrent_hidden_size=Seq2Seq_config.RECURRENT_HIDDEN_SIZE,
                                 num_recurrent_layers=Seq2Seq_config.NUM_RECURRENT_LAYERS,
                                 device=Attack_config.TRAIN_DEVICE,
                                 use_bidirectional=True,
                                 dropout_rate=Seq2Seq_config.DROPOUT_RATE,
                                 fine_tuning=False).to(Attack_config.TRAIN_DEVICE)

    seq2seq_model.load_state_dict(torch.load(
        Seq2Seq_config.STATE_PATH, map_location=Attack_config.TRAIN_DEVICE))

    bert_identifiers = ['embedding', 'encoder', 'pooler']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']

    grouped_model_parameters = [
        {'params': [param for name, param in seq2seq_model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
            'lr': Attack_config.BERT_LEARNING_RATE,
            'betas': Attack_config.BETAS,
            'weight_decay': Attack_config.BERT_WEIGHT_DECAY,
            'eps': Attack_config.EPS},
        {'params': [param for name, param in seq2seq_model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
            'lr': Attack_config.BERT_LEARNING_RATE,
            'betas': Attack_config.BETAS,
            'weight_decay': 0.0,
            'eps': Attack_config.EPS}
    ]

    optimizer = optim.AdamW(grouped_model_parameters)
    criterion_seq2seq = nn.CrossEntropyLoss(
        ignore_index=0).to(Attack_config.TRAIN_DEVICE)
    criterion_classification = nn.CrossEntropyLoss().to(Attack_config.TRAIN_DEVICE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=3,
                                                     verbose=True, min_lr=3e-8)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: 1e-2 if ep < 4 else 1.0)

    if Attack_config.ONLY_EVAL:
        path = cur_dir + f'/eval_org_seq2seq_attack.log'
        attack_acc, org_predict_acc, attacked_predict_acc = evaluate_attack(
            test_data, seq2seq_model, victim_model, path, tokenizer, 0)
        logging(
            f'attack_acc:{attack_acc}   org_predict_acc:{org_predict_acc}   attacked_predict_acc:{attacked_predict_acc}')
    else:
        best_attack_acc = 0.0
        temp_path = cur_models_dir + f'/{Attack_config.DATASET}_temp_model.pt'
        for ep in range(Attack_config.NUM_EPOCHS):
            logging(f'epoch {ep+1} start train')
            train_loss = adversary_train(
                train_data, seq2seq_model, victim_model, criterion_classification, criterion_seq2seq, optimizer)
            logging(f'epoch {ep+1} start evaluate')
            indices_path = cur_dir + f'/eval_attack_epoch_{ep+1}.log'
            attack_acc, org_predict_acc, attacked_predict_acc = evaluate_attack(
                test_data, seq2seq_model, victim_model, indices_path, tokenizer, ep)
            if attack_acc > best_attack_acc:
                best_attack_acc = attack_acc
                best_path = cur_models_dir + \
                    f'/{Seq2Seq_config.DATASET}_{attack_acc:.5f}_{get_time()}.pt'
                best_state = copy.deepcopy(seq2seq_model.state_dict())

                if ep > 3 and best_state != None:
                    logging(
                        f'saving best seq2seq_model attack acc {best_attack_acc:.5f} in {temp_path}')
                    torch.save(best_state, temp_path)

            logging(
                f'epoch {ep+1} done! train_loss {train_loss:.5f} \n'
                f'attack_acc:{attack_acc}   org_predict_acc:{org_predict_acc}   attacked_predict_acc:{attacked_predict_acc}')

        if best_state != None:
            logging(
                f'saving best seq2seq_model attack acc {best_attack_acc:.5f} in {best_path}')
            torch.save(best_state, best_path)
