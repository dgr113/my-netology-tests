# coding: utf-8

from typing import Tuple

from torch import Tensor  # type: ignore
from torch.nn import Module, Linear, RNN, Embedding  # type: ignore





class CustomEncoder(Module):
    def __init__(self, input_size: int, emb_size: int, hidden_size: int = 128):
        super().__init__()

        self.hidden_size = hidden_size
        self.emb = Embedding(input_size, emb_size)
        self.rnn = RNN(input_size, hidden_size, batch_first=True)

    def forward(self, out: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out = self.emb(out)
        out, hidden = self.rnn(out)
        return out, hidden



class CustomDecoder(Module):
    def __init__(self, emb_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()

        self.hidden_size = hidden_size
        self.emb = Embedding(output_size, emb_size)
        self.rnn = RNN(output_size, hidden_size, batch_first=True)
        self.out = Linear(hidden_size, output_size)

    def forward(self, out: 'Tensor', hidden: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out = self.emb(out)
        out, hidden = self.rnn(out, hidden)
        out = self.out(out)
        out = out.view(-1, 28)  # Flatten output
        return out, hidden



class Seq2Seq(Module):
    def __init__(self, encoder: 'CustomEncoder', decoder: 'CustomDecoder'):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X: 'Tensor', y: 'Tensor') -> 'Tensor':
        out, hidden = self.encoder(X)
        out, _ = self.decoder(y, hidden)
        return out
