# coding: utf-8

import torch
import numpy as np
import pandas as pd
from torch import nn


if torch.cuda.is_available():
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

        return torch.from_numpy(features)


    @staticmethod
    def train(model, X, y):
        model = model.to(device)

        # print(X.shape, y.shape)  # [100, 14, 27]

        n_epochs = 10
        lr = 0.01

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X = X.to(device)
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()


            # print(X.shape, y.shape)  # [100, 14, 27] - [100, 14]
            X_predicted, hidden = model(X)
            X_predicted, y = X_predicted.to(device), y.to(device)
            # print(X_predicted.shape, y.shape)  # [1400, 27] - [100, 14]

            loss = criterion(X_predicted, y.view(-1).long())
            loss.backward()
            optimizer.step()

            # if epoch % 10 == 0:
            #     print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            #     print("Loss: {:.4f}".format(loss.item()))


    @staticmethod
    def _predict(model, character, char2int, int2char, dict_size):
        character = np.array([[char2int[c] for c in character]])
        character = Processing.one_hot_encode(character, dict_size, character.shape[1], 1)
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
    TEXT = pd.read_csv('./data/data.csv')['normalized_text'].fillna('').str[:15].iloc[:100].tolist()
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


    input_seq = Processing.one_hot_encode(input_seq, dict_size, seq_len, batch_size)  # Embedding to extend shape
    target_seq = torch.Tensor(target_seq)


    model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

    Processing.train(model, input_seq, target_seq)
    result = Processing.predict_sample(model, 15, char2int, int2char, dict_size, 'le')
    print('RESULT: ', result)





if __name__ == '__main__':
    main()
