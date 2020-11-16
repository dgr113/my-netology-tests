# coding: utf-8

import torch
import numpy as np
import pandas as pd
from torch import nn


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")




class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden



class Processing:

    @staticmethod
    def one_hot_encode(sequence, dict_size, seq_len, batch_size):
        features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u, sequence[i][u]] = 1
        return features


    @staticmethod
    def train(model, input_seq, target_seq):
        model = model.to(device)

        n_epochs = 100
        lr = 0.01

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        input_seq = input_seq.to(device)
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            output, hidden = model(input_seq)
            output = output.to(device)
            target_seq = target_seq.to(device)
            loss = criterion(output, target_seq.view(-1).long())
            loss.backward()
            optimizer.step()

            # if epoch % 10 == 0:
            #     print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            #     print("Loss: {:.4f}".format(loss.item()))


    @staticmethod
    def _predict(model, character, char2int, int2char, dict_size):
        character = np.array([[char2int[c] for c in character]])
        character = Processing.one_hot_encode(character, dict_size, character.shape[1], 1)
        character = torch.from_numpy(character)
        character = character.to(device)

        out, hidden = model(character)

        prob = nn.functional.softmax(out[-1], dim=0).data
        char_ind = torch.max(prob, dim=0)[1].item()

        return int2char[char_ind], hidden


    @staticmethod
    def predict_sample(model, out_len, char2int, int2char, dict_size, start='hey'):
        model.eval()
        start = start.lower()

        chars = [ch for ch in start]
        size = out_len - len(chars)

        for _ in range(size):
            char, h = Processing._predict(model, chars, char2int, int2char, dict_size)
            chars.append(char)

        return ''.join(chars)




def main():
    # TEXT = ['hey how are you', 'good i am fine', 'have a nice day']
    TEXT = pd.read_csv('./data.csv')['normalized_text'].fillna('').str[:50].iloc[:100].tolist()
    chars = set('abcdefghijklmnopqrstuvwxyz ')

    int2char = dict( enumerate(chars) )
    char2int = { char: ind for ind, char in int2char.items() }


    maxlen = len( max(TEXT, key=len) )
    for i in range(len(TEXT)):
        while len(TEXT[i]) < maxlen:
            TEXT[i] += ' '


    input_seq = []
    target_seq = []

    for i in range(len(TEXT)):
        input_seq.append(TEXT[i][:-1])
        target_seq.append(TEXT[i][1:])
        # print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

    for i in range(len(TEXT)):
        input_seq[i] = [ char2int.get(character, char2int[' ']) for character in input_seq[i] ]
        target_seq[i] = [ char2int.get(character, char2int[' ']) for character in target_seq[i] ]


    dict_size = len(char2int)
    seq_len = maxlen - 1
    batch_size = len(TEXT)


    input_seq = Processing.one_hot_encode(input_seq, dict_size, seq_len, batch_size)
    # print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)


    model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

    Processing.train(model, input_seq, target_seq)
    result = Processing.predict_sample(model, 15, char2int, int2char, dict_size, 'le')
    print('RESULT: ', result)





if __name__ == '__main__':
    main()
