import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, bert, src_vocab, device):
        super().__init__()
        
        self.bert = bert
        self.src_vocab = src_vocab
        self.device = device
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        src = src.permute(1, 0)
        
        src_padded_mask = (src == self.src_vocab.stoi["[PAD]"]).to(self.device)
        src_padded_mask = torch.where(src_padded_mask, 0, 1)
        
        enc_outputs, embedded_sentence = self.bert(src, encoder_attention_mask=src_padded_mask)[:2]
        #enc_outputs = [batch size, src len, hid_dim=768]
        #embedded_sentence = [batch size, hid_dim=768]
        enc_outputs = enc_outputs.permute(1, 0, 2)
        embedded_sentence = embedded_sentence.unsqueeze(0)
        #enc_outputs = [src len, batch size, hid_dim=768]
        #embedded_sentence = [1, batch size, hid_dim=768]
        return enc_outputs, embedded_sentence
    

class Attention(nn.Module):
    def __init__(self, dec_hid_dim, enc_output_dim):
        super().__init__()
        self.dec_hid_dim = dec_hid_dim
        self.enc_output_dim = enc_output_dim
        
        self.att = nn.Linear(in_features=enc_output_dim+dec_hid_dim, 
                             out_features=dec_hid_dim)
        
        self.v = nn.Linear(in_features=dec_hid_dim,
                           out_features=1,
                           bias=False)
        
    def forward(self, hidden, enc_outputs):
        # hidden = [1, batch size, hid dim]
        hidden = hidden.repeat(enc_outputs.shape[0], 1, 1)
        output = torch.cat((hidden, enc_outputs), dim=2)
        
        # output = [src sent len, batch_size, hid_dim*2]
        
        attention = torch.tanh(self.att(output))
        attention = self.v(attention).squeeze(dim=2)
        
        # attention = [src sent len, batch_size]
        
        return nn.functional.softmax(attention, dim=0)
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.attention = attention
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.GRU(
            input_size=hid_dim+emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.out = nn.Linear(
            in_features=hid_dim*2+emb_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, enc_outputs):
        
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # enc_outputs = [src sent len, batch_size, hidden_size=768]
        
        input = input.unsqueeze(0)
        
        # input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        a = self.attention(hidden, enc_outputs)
        # a = [src sent len, batch_size]
        
        a = a.unsqueeze(1).permute(2, 1, 0)
        # a = [batch_size, 1, src sent len]
        
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs = [batch_size, src sent len, hidden_size]
        
        weights = torch.bmm(a, enc_outputs)
        # weights = [batch_size, 1, hidden_size]
        
        weights = weights.permute(1, 0, 2)
        # weights = [1, batch_size, hidden_size]
        
        rnn_input = torch.cat((embedded, weights), dim=2)
        # rnn_input = [1, batch_size, hidden_size + emb_size]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output = [1, batch_size, hidden_size]
        # hidden = [1, batch_size, hidden_size]
        
        output = torch.cat((output, weights, embedded), dim=2)
        # output = [1, batch_size, hidden_size + hidden_size + emb_size]
        
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_outputs, hidden = self.encoder(src)
        #enc_outputs = [src len, batch size, hid_dim=768]
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, enc_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs

