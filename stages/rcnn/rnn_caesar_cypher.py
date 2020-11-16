# coding: utf-8

import pandas as pd

from functools import partial
from itertools import starmap
from string import ascii_lowercase
from typing import Union, Tuple, Sequence, Optional, Dict, Callable
from dataclasses import dataclass, field, InitVar

from torch import Tensor, no_grad, device, cuda, zeros, max as torch_max  # type: ignore
from torch.nn import Module, Linear, RNN, Embedding  # type: ignore
from torch.nn.functional import cross_entropy, softmax  # type: ignore
from torch.optim import Optimizer, Adam  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )  # USE CUDA GPU

UNI_CHARS_TYPE = Union[str, Sequence[str]]
_2D_INT_ARRAY = Sequence[Sequence[int]]
DATA_TRANSFORM_FUNC = Callable[[Sequence], 'Tensor']




@dataclass
class DataLoaderParams:
    batch_size: int = 30
    batch_shuffle: bool = False



@dataclass
class TrainParams:
    epochs: int
    lr: float
    accuracy_threshold: float = 0.85



@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float
    train_params: 'TrainParams'




class CustomTextDataset(Dataset):
    def __init__(
        self,
        data: Sequence[str],
        char2int: Dict[str, int],
        doc_len: int = 15,
        *,
        transform_X: Optional[DATA_TRANSFORM_FUNC] = None,
        transform_y: Optional[DATA_TRANSFORM_FUNC] = None
    ):
        self.doc_len = doc_len
        self.transform_X = transform_X
        self.transform_y = transform_y
        self._data = data
        self._to_tensor = partial(CustomTextDataset.get_tensor_data, char2int)

    @staticmethod
    def get_tensor_data(char2int: Dict[str, int], chars: UNI_CHARS_TYPE, *, as_unsqueeze: bool = False) -> 'Tensor':
        t = Tensor( list( char2int.get(ch, 0) for ch in chars ) ).long()
        return t.unsqueeze(0) if as_unsqueeze else t

    @staticmethod
    def get_char_ind_map(vocab: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        char2int = { w: i for i, w in enumerate(vocab, start=1) }
        int2char = { i: w for w, i in char2int.items() }
        return char2int, int2char

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple['Tensor', 'Tensor']:
        s_appendix = [''] * self.doc_len

        s = self._data[idx]
        s = [ *s, *s_appendix ][:self.doc_len]

        X, y = ( self.transform_X(s) if self.transform_X else s ), ( self.transform_y(s) if self.transform_y else s )
        X, y = self._to_tensor(X), self._to_tensor(y)
        return X, y




class CustomRNN(Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = Embedding(input_size, input_size)
        self.rnn = RNN(input_size, hidden_size, batch_first=True)
        self.out = Linear(hidden_size, output_size)

    def _init_hidden(self, batch_size: int) -> 'Tensor':
        return zeros(1, batch_size, self.hidden_size)

    def forward(self, out: 'Tensor', hidden: Optional['Tensor'] = None) -> Tuple['Tensor', 'Tensor']:
        out = self.emb(out)
        out, hidden = self.rnn(out)
        out = self.out(out)
        out = out.view(-1, 28)  # Flatten output
        return out, hidden



@dataclass
class TrainContext:
    dataset: InitVar['Dataset']

    train_params: 'TrainParams'
    loader_params: 'DataLoaderParams'

    model: 'Module'
    optimizer: 'Optimizer'

    _current_epoch: int = 0
    _epoch_loss: float = 0
    _current_accuracy: float = 0
    _correct_pred_count: int = 0
    _data: 'DataLoader' = field(init=False)

    def __post_init__(self, dataset: 'Dataset'):
        self._data = DataLoader(dataset, batch_size=self.loader_params.batch_size, shuffle=self.loader_params.batch_shuffle)


    def __iter__(self):
        return self


    def __next__(self):
        """ Next train epoch """

        if self._current_epoch < self.train_params.epochs:
            self._epoch_start()

            hidden = None
            train_loss = 0
            train_passed = 0

            for X_batch, y_batch in Processing.loader_to_device(self._data):
                # print("Train data shapes: ", X_batch.shape, y_batch.shape)  # >>> [100, 14], [100, 14]
                X_predicted, hidden = self.model(X_batch, hidden)
                # print("Predicted data shapes: ", X_predicted.shape, y_batch.shape)  # >>> [1400, 28], [100, 14]

                loss = cross_entropy(X_predicted, y_batch.flatten())
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_passed += 1

            print( "Train loss: {:.3f}".format(train_loss / train_passed) )
            epoch_stats = self._get_epoch_stats()
            return epoch_stats
        else:
            raise StopIteration


    def _epoch_start(self) -> None:
        self._current_epoch += 1
        self._epoch_loss = 0
        self._correct_pred_count = 0


    def _epoch_update(self, loss: 'Tensor', y_pred: 'Tensor', y_target: 'Tensor') -> None:
        """ Update inner epoch stats

            :param loss: current training loss
            :param y_pred: Predicted labels - 2D Array with probabilities of matching each class for each row
            :param y_target: Array of target labels for each row of current data batch
        """
        self._epoch_loss += loss.item() * self._data.batch_size
        self._correct_pred_count += int( y_pred.argmax(dim=1).eq(y_target).sum().item() )


    def _get_epoch_stats(self) -> 'TrainStats':
        """ Calculate main epoch stats """
        dataset_len = len(self._data.dataset)
        return TrainStats(
            epoch=self._current_epoch,
            loss=self._epoch_loss / dataset_len,
            accuracy=self._correct_pred_count / dataset_len,
            train_params=self.train_params
        )



class Processing:
    @staticmethod
    def _batch_to_device(X: 'Tensor', y: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        return X.to(TORCH_DEVICE), y.to(TORCH_DEVICE)

    @staticmethod
    def loader_to_device(data: 'DataLoader') -> Tuple['Tensor', 'Tensor']:
        yield from starmap(Processing._batch_to_device, data)

    @staticmethod
    def train_model(dataset: 'Dataset', train_params: 'TrainParams', loader_params: 'DataLoaderParams', model: 'Module', optimizer: 'Optimizer') -> None:
        for epoch_stats in TrainContext(dataset, train_params, loader_params, model, optimizer):
            # print( "Epoch train stats: {}".format(epoch_stats) )
            pass

    @staticmethod
    def test_model(dataset: 'Dataset', loader_params: 'DataLoaderParams', model: 'Module') -> float:
        model.train()
        total, correct = 0, 0
        with no_grad():
            for images, labels in Processing.loader_to_device(DataLoader(dataset, batch_size=loader_params.batch_size, shuffle=loader_params.batch_shuffle)):
                outputs = model(images)
                _, predicted = torch_max(outputs.data, 1)
                total += labels.size(0)
                correct += int( outputs.argmax(dim=1).eq(labels).sum().item() )

        result_accuracy = 100 * correct / total
        print( "Accuracy of the network on test data is: {}%".format(result_accuracy) )
        return result_accuracy


    @staticmethod
    def _predict_step(int2char: Dict[int, str], outputs: 'Tensor') -> str:
        outputs = outputs[-1]  # Get last doc, because there's only one of them

        # print(outputs.shape)  # [1, 5, 28] - 1 документ, 5 символов, 28 вероятностей следующег осимвола для каждого класса
        prob = softmax(outputs, dim=0).data
        _, result = torch_max(prob, dim=0)

        predicted_ind = int(result.item())
        predicted_char = int2char.get(predicted_ind, ' ')
        return predicted_char


    @staticmethod
    def predict(model: 'Module', int2char: Dict[int, str], char2int: Dict[str, int], chars: UNI_CHARS_TYPE) -> None:
        model.eval()

        with no_grad():
            for i in range(30):
                X = CustomTextDataset.get_tensor_data(char2int, chars, as_unsqueeze=True)
                outputs, _ = model(X)
                predicted_char = Processing._predict_step(int2char, outputs)
                chars += predicted_char

            result = ''.join(chars)
            print('Predicted: ', result)



@dataclass
class ModelsTests:
    train_params: 'TrainParams' = TrainParams(epochs=10, lr=0.002)

    def test_one(self) -> None:
        """ Test RNN to predict the following letters """

        vocab = 'abcdefghijklmnopqrstuvwxyz '
        char2int, int2char = CustomTextDataset.get_char_ind_map(vocab)
        vocab_size = len(vocab) + 1
        transform_X = ( lambda t: t[:-1] )
        transform_y = ( lambda t: t[1:] )

        data = pd.read_csv('./data/data.csv')['normalized_text'].fillna('').str[:15].iloc[:100].tolist()
        train_dataset = CustomTextDataset(data, char2int, transform_X=transform_X, transform_y=transform_y)

        model = CustomRNN(vocab_size, vocab_size).to(TORCH_DEVICE)
        optimizer = Adam(model.parameters(), lr=0.005)

        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer)

        Processing.predict(model, int2char, char2int, 'le')


    def test_two(self) -> None:
        """ Test RNN with caesar encription """

        vocab = 'abcdefghijklmnopqrstuvwxyz '
        char2int, int2char = CustomTextDataset.get_char_ind_map(vocab)
        vocab_size = len(vocab) + 1

        data = pd.read_csv('./data/data.csv')['normalized_text'].fillna('').str[:15].iloc[:100].tolist()
        train_dataset = CustomTextDataset(data, char2int)

        model = CustomRNN(vocab_size, vocab_size).to(TORCH_DEVICE)
        optimizer = Adam(model.parameters(), lr=0.005)

        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer)

        Processing.predict(model, int2char, char2int, 'le')



def caesar_enc(text: str, shift: int) -> str:
    alphabet = ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    return text.translate(table)



def main():
    tests_context = ModelsTests()
    tests_context.test_one()

    # s = "Hello you"
    # result = caesar_enc(s, 2)
    # print(result)



if __name__ == '__main__':
    main()
