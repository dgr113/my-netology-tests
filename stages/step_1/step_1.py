# coding: utf-8

import torch
import numpy as np  # type: ignore
from typing import Tuple
from torch import Tensor
from torch.nn import Sequential
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils import Bunch  # type: ignore




def normalize_values(x: 'np.array') -> 'np.array':
    """ Get normalized values for each feature """
    return (x - x.mean()) / x.std()




def get_train_data(dataset: 'Bunch') -> Tuple['Tensor', 'Tensor', 'Tensor', 'Tensor']:
    """ Separate dataset object into learning and test datasets in tensor forms """

    X, y = dataset['data'], dataset['target']

    X = np.apply_along_axis(normalize_values, 0, X)

    X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X, y, test_size=0.3, random_state=30)

    X_train = torch.tensor(X_train_, dtype=torch.float)
    Y_train = torch.tensor(Y_train_, dtype=torch.float).view(-1, 1)  # type: 'Tensor'

    X_test = torch.tensor(X_test_, dtype=torch.float)
    Y_test = torch.tensor(Y_test_, dtype=torch.float).view(-1, 1)  # type: 'Tensor'

    return X_train, Y_train, X_test, Y_test




def get_model(input_features: int, output_features: int = 1, lr: float = 0.05) -> Tuple['Sequential', '_Loss', 'Optimizer']:
    model = torch.nn.Sequential(torch.nn.Linear(input_features, output_features))
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_func, optimizer




def train_model(
    model: 'Sequential',
    loss_func: '_Loss',
    optimizer: 'Optimizer',
    X_train: 'Tensor',
    y_train: 'Tensor',
    X_test: 'Tensor',
    y_test: 'Tensor',
    batch_size: int = 30,
    epochs: int = 5

) -> 'Sequential':

    train_data = DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(epochs):
        for X_batch, y_batch in train_data:
            output = model(X_batch)
            loss = loss_func(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # noinspection PyUnboundLocalVariable
        print("{} epoch loss: {}".format(epoch + 1, loss.item()))

    result = loss_func( model(X_test), y_test )
    print("Test data loss: ", result.item())
    return model




def main():
    X_train, y_train, X_test, y_test = get_train_data( load_boston() )

    model, loss_func, optimizer = get_model(X_train.shape[1])

    _ = train_model(model, loss_func, optimizer, X_train, y_train, X_test, y_test)





if __name__ == '__main__':
    main()
