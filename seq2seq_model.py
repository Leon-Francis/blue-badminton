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


class Seq2Seq_bert(nn.modules):
    def __init__(self, pretrained_model_name, num_pretrained_bert_layers, max_tokenization_length,
                 recurrent_hidden_size, num_recurrent_layers, device, use_bidirectional=False, num_classes=2, 
                 dropout_rate=0.2) -> None:
        super(Seq2Seq_bert, self).__init__()

        self.device = device
        self.num_layers = num_recurrent_layers
        self.recurrent_hidden_size = recurrent_hidden_size
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.max_len = max_tokenization_length

        self.config = BertConfig.from_pretrained(pretrained_model_name)
        
        
        self.config.max_position_embeddings = max_tokenization_length
        self.config.num_hidden_layers = num_pretrained_bert_layers
        self.config.output_hidden_states = True

        self.bert = BertModel.from_pretrained(pretrained_model_name,
                                              config=self.config)

        self.dropout = nn.Dropout(dropout_rate)

        self.embedding_decoder = nn.Embedding(max_tokenization_length, recurrent_hidden_size)

        self.lstm = nn.LSTM(input_size=self.config.hidden_size,
                            hidden_size=recurrent_hidden_size,
                            num_layers=num_recurrent_layers,
                            bidirectional=use_bidirectional,
                            batch_first=True)

        self.linear_decoder = nn.Linear(recurrent_hidden_size, self.config.vocab_size)


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
        all_hidden = hidden.unsqueeze(dim=1).repeat(1, self.tokenizer.max_len, 1)

        state = (hidden.unsqueeze(0), torch.zeros(self.num_layers, batch_size, self.recurrent_hidden_size).to(self.device))

        embeddings = self.embedding_decoder(input_ids)
        augmented_embeddings = torch.cat([embeddings, all_hidden], dim=2)
        # (B, P, H*)
        output, state = self.lstm(augmented_embeddings, state)

        decoded = self.linear_decoder(self.dropout(output.view(-1, self.recurrent_hidden_size)))
        # (B, P, V)
        decoded = decoded.view(batch_size, self.tokenizer.max_len, self.config.vocab_size)

        return decoded

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.encode(input_ids, token_type_ids, attention_mask)

        decoded_output = self.decode(pooled_output, input_ids)

        return decoded_output

    def generate(self, hidden, temp=1.0):
        """Generate through decoder; no backprop"""
        batch_size = hidden.shape[0]
        
        all_hidden = hidden.unsqueeze(dim=1).repeat(1, self.tokenizer.max_len, 1)

        state = (hidden.unsqueeze(0), torch.zeros(self.num_layers, batch_size, self.recurrent_hidden_size).to(self.device))
        start_symbols = torch.ones(batch_size, 1).long().to(self.device)
        start_symbols.data.fill_(101)

        decoder_embedding = self.embedding_decoder(start_symbols)
        augmented_embeddings = torch.cat([decoder_embedding, hidden.unsqueeze(1)], dim=2)

        all_indices = []
        for i in range(self.tokenizer.max_len):
            # (B, 1, H*)
            output, state = self.lstm(augmented_embeddings, state)
            overvocab = self.linear_decoder(output.squeeze(dim=1))

            # sampling
            probs = F.softmax(overvocab / temp)
            indices = torch.multinomial(probs, 1)

            if indices.ndimension() == 1:
                indices = indices.unsqueeze(dim=1)
            all_indices.append(indices)

            decoder_embedding = self.embedding_decoder(indices)
            augmented_embeddings = torch.cat([decoder_embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices

    def get_tokenizer(self):
        return self.tokenizer