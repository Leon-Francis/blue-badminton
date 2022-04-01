"""
i)     B  = batch size,
ii)    P  = maximum number of positional embeddings from BERT tokenizer (default: 512),
iii)   H  = hidden size dimension in pretrained BERT layers (default: 768),
iv)    H* = hidden size dimension for the additional recurrent (LSTM) layer,
v)     H' = hidden size dimension when multiple BERT layers are concatenated, H' = H iff K = 1
vi)    L  = number of recurrent layers
vi)    K  = number of pretrained BERT layers utilized out of 12,
viii)  N  = number of heads in multi-head, self-attention mechanism of BERT out of 12
ix)    V  = vocab size (default: 30522)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig, BertModel


class Seq2Seq_bert(nn.Module):
    def __init__(self, pretrained_model_name, num_pretrained_bert_layers, max_tokenization_length,
                 recurrent_hidden_size, num_recurrent_layers, device, use_bidirectional=False,
                 dropout_rate=0.2, fine_tuning=True) -> None:
        super(Seq2Seq_bert, self).__init__()

        self.device = device
        self.num_layers = num_recurrent_layers
        self.recurrent_hidden_size = recurrent_hidden_size
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.max_len = max_tokenization_length

        self.config = BertConfig.from_pretrained(pretrained_model_name)

        # Get customized BERT config
        self.config.max_position_embeddings = max_tokenization_length
        self.config.num_hidden_layers = num_pretrained_bert_layers
        self.config.output_hidden_states = True
        self.config.output_attentions = True

        self.bert = BertModel.from_pretrained(pretrained_model_name,
                                              config=self.config)
        if not fine_tuning:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)

        self.embedding_decoder = nn.Embedding(
            self.tokenizer.vocab_size, recurrent_hidden_size)

        self.lstm = nn.LSTM(input_size=self.config.hidden_size+recurrent_hidden_size,
                            hidden_size=recurrent_hidden_size,
                            num_layers=num_recurrent_layers,
                            bidirectional=use_bidirectional,
                            batch_first=True)

        self.linear_decoder = nn.Linear(
            recurrent_hidden_size*2, self.tokenizer.vocab_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def encode(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask)

        # (B, P, H)
        sequence_output = bert_outputs[0]
        # (B, H)
        pooled_output = bert_outputs[1]

        return sequence_output, pooled_output

    def decode(self, hidden, input_ids):
        # hidden (B, H)
        batch_size = hidden.shape[0]
        # all hidden (B, P, H)
        all_hidden = hidden.unsqueeze(dim=1).repeat(
            1, self.tokenizer.max_len, 1)

        state = (hidden.unsqueeze(dim=0).repeat(self.num_layers*2, 1, 1),
                 torch.zeros(self.num_layers*2, batch_size, self.recurrent_hidden_size).to(self.device))

        embeddings = self.embedding_decoder(input_ids)
        augmented_embeddings = torch.cat([embeddings, all_hidden], dim=2)
        # (B, P, H*)
        output, state = self.lstm(augmented_embeddings, state)

        decoded = self.softmax(self.linear_decoder(self.dropout(
            output.reshape(-1, self.recurrent_hidden_size*2))))
        # (B, P, V)
        decoded = decoded.view(
            batch_size, self.tokenizer.max_len, self.config.vocab_size)

        return decoded

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.encode(
            input_ids, token_type_ids, attention_mask)

        decoded_output = self.decode(pooled_output, input_ids)

        return decoded_output

    def generate(self, hidden, temp=1.0):
        """Generate through decoder; no backprop"""
        batch_size = hidden.shape[0]

        state = (hidden.unsqueeze(0).repeat(self.num_layers*2, 1, 1), torch.zeros(self.num_layers*2,
                 batch_size, self.recurrent_hidden_size).to(self.device))
        start_symbols = torch.ones(batch_size, 1).long().to(self.device)
        start_symbols.data.fill_(101)

        decoder_embedding = self.embedding_decoder(start_symbols)
        augmented_embeddings = torch.cat(
            [decoder_embedding, hidden.unsqueeze(1)], dim=2)

        all_decoded = []
        for i in range(self.tokenizer.max_len):
            # (B, 1, H*)
            output, state = self.lstm(augmented_embeddings, state)
            # (B, V)
            decoded = self.softmax(self.linear_decoder(self.dropout(output.squeeze(dim=1))))

            all_decoded.append(decoded.unsqueeze(1))

            topv, topi = decoded.topk(1)

            decoder_embedding = self.embedding_decoder(topi.squeeze().detach())
            augmented_embeddings = torch.cat(
                [decoder_embedding.unsqueeze(1), hidden.unsqueeze(1)], 2)

        # (B, P, V)
        all_decoded = torch.cat(all_decoded, dim=1)

        return all_decoded
