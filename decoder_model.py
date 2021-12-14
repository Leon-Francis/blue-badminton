import torch
from torch import nn
from transformers import BertModel
from decoder_config import BERT_HIDDEN_SIZE, BERT_VOCAB_SIZE


class Decoder_LSTM(nn.Module):
    def __init__(self, nhidden, num_layers, dropout, fine_tuning=False):
        super(Decoder_LSTM, self).__init__()
        self.nhidden = nhidden
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        if not fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        decoder_input_size = BERT_HIDDEN_SIZE + nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True)

        self.linear_decoder = nn.Sequential(nn.Dropout(0.5),
                                            nn.Linear(nhidden, BERT_VOCAB_SIZE))

         
    def forward(self, x, masks, types, hidden):
        sen_len = x.shape[1]
        batch_size = x.shape[0]
        all_hidden = hidden.unsqueeze(1).repeat(1, sen_len, 1)

        # last_hidden_state [batch_size, sen_len, hidden_size]
        last_hidden_state, pooled_output = self.bert_model(x, masks, types)
        augmented_hidden = torch.cat([last_hidden_state, all_hidden], dim=2)

        output, state = self.decoder(augmented_hidden)

        decoded = self.linear_decoder(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, sen_len, -1)

        return decoded