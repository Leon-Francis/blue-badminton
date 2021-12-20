import torch
from torch import nn
from transformers import BertModel

class Attack_Bert(nn.Module):
    def __init__(self, is_fine_tuning=True):
        super(Attack_Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = 768
        if not is_fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def forward(self, x, masks, types):
        encoder, pooled = self.bert_model(x, masks, types)[:]
        return pooled