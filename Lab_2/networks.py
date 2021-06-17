import torch, torch.nn as nn
import torch.nn.functional as F


class CharRNNCell(nn.Module):
    def __init__(self, num_tokens, embedding_size=64, rnn_num_units=128):
        super(self.__class__,self).__init__()
        self.num_units = rnn_num_units
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(
            embedding_size + rnn_num_units,
            rnn_num_units
        )
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)
        
    def forward(self, x, h_prev):
        x_emb = self.embedding(x)
        x_and_h = torch.cat([x_emb, h_prev], dim=-1)
        h_next = self.rnn_update(x_and_h)
        h_next = torch.tanh(h_next)
        logits = self.rnn_to_logits(h_next)
        
        return h_next, logits
    
    def initial_state(self, batch_size):
        return torch.zeros(batch_size, self.num_units, requires_grad=True)


class CharLSTMLoop(nn.Module):
    def __init__(self, num_tokens, emb_size=64, rnn_num_units=128, num_layers=2):
        super(self.__class__, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.num_units = rnn_num_units
        self.num_layers = num_layers
        self.emb = nn.Embedding(self.num_tokens, self.emb_size)
        self.rnn = nn.LSTM(self.emb_size, self.num_units, self.num_layers, batch_first=True)
        self.hid_to_logits = nn.Linear(self.num_units, self.num_tokens)
        
    def forward(self, x, prev_state):
        assert isinstance(x.data, torch.LongTensor)
        h_seq, state = self.rnn(self.emb(x), prev_state)
        next_logits = self.hid_to_logits(h_seq)
        return next_logits, state
    
    def initial_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.num_units, requires_grad=True),
                torch.zeros(self.num_layers, batch_size, self.num_units, requires_grad=True))
