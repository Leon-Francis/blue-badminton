import torch 
from torch import nn
from transformers import BertModel


class Victim_Bert(nn.Module):
    def __init__(self, label_num:int, linear_layer_num:int, dropout_rate:float, is_fine_tuning=True):
        super(Victim_Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = 768
        if not is_fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        modules = [nn.Dropout(dropout_rate)]

        for i in range(linear_layer_num-1):
            modules += [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
        modules.append(nn.Linear(self.hidden_size, label_num))
        self.fc = nn.Sequential(*modules)


    def forward(self, x, masks, types):
        # inputs = (x, types, masks)
        encoder, pooled = self.bert_model(x, masks, types)[:]
        logits = self.fc(pooled)
        return logits

    def embedding(self, x, masks, types):
        encoder, pooled = self.bert_model(x, masks, types)[:]
        return pooled

    def classification(self, pooled):
        return self.fc(pooled)