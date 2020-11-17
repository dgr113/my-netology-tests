# coding: utf-8

import pandas as pd  # type: ignore

from functools import partial
from itertools import starmap
from string import ascii_lowercase
from typing import Union, Tuple, Sequence, Optional, Dict, Callable, List, Generator
from dataclasses import dataclass, field, InitVar

from torch import Tensor, no_grad, device, cuda, max as torch_max  # type: ignore
from torch.nn import Module, Linear, RNN, Embedding  # type: ignore
from torch.nn.functional import cross_entropy, softmax  # type: ignore
from torch.optim import Optimizer, Adam  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )  # USE CUDA GPU

UNI_CHARS_TYPE = Union[str, List[str]]
A2D_INT_ARRAY = Sequence[Sequence[int]]
DATA_TRANSFORM_FUNC = Callable[[str], str]




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
        s = self._data[idx]
        s = s.ljust(self.doc_len)[:self.doc_len]

        X_, y_ = ( self.transform_X(s) if self.transform_X else s ), ( self.transform_y(s) if self.transform_y else s )
        X, y = self._to_tensor(X_), self._to_tensor(y_)
        return X, y



class CustomRNN(Module):
    def __init__(self, input_size: int, emb_size: int, hidden_size: int = 128):
        super().__init__()

        self.input_size, self.output_size = [ input_size + 1 ] * 2  # Alignment for case when the vocab length is passed

        self.emb = Embedding(self.input_size, emb_size)
        self.rnn = RNN(emb_size, hidden_size, batch_first=True)
        self.out = Linear(hidden_size, self.output_size)

    def forward(self, out: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out = self.emb(out)
        out, hidden = self.rnn(out)
        out = self.out(out)
        out = out.view(-1, self.output_size)  # Flatten output
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

            train_loss = 0
            train_passed = 0

            for X_batch, y_batch in Processing.loader_to_device(self._data):
                X_predicted, _ = self.model(X_batch)

                loss = cross_entropy(X_predicted, y_batch.flatten(0))
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
    def loader_to_device(data: 'DataLoader') -> Generator[Tuple['Tensor', 'Tensor'], None, None]:
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
                total += labels.size()[0]
                correct += int( outputs.argmax(dim=1).eq(labels).sum().item() )

        result_accuracy = 100 * correct / total
        print( "Accuracy of the network on test data is: {}%".format(result_accuracy) )
        return result_accuracy

    @staticmethod
    def predict(model: 'Module', int2char: Dict[int, str], char2int: Dict[str, int], chars: str) -> str:
        model.eval()

        with no_grad():
            for i in range(30):
                chars = "".join(chars)
                X = CustomTextDataset.get_tensor_data(char2int, chars, as_unsqueeze=True)
                outputs, _ = model(X)
                ###
                outputs = outputs[-1]  # Get last doc, because there's only one of them
                prob = softmax(outputs, dim=0).data
                _, d = torch_max(prob, dim=0)
                predicted_char = int2char.get(int(d.item()), ' ')
                ###
                chars += predicted_char

            result = "".join(chars)
            return result

    @staticmethod
    def predict_caesar(model: 'Module', int2char: Dict[int, str], char2int: Dict[str, int], chars: str) -> str:
        model.eval()

        with no_grad():
            X = CustomTextDataset.get_tensor_data(char2int, chars, as_unsqueeze=True)
            outputs, _ = model(X)
            prob = torch_max(outputs, dim=1)
            predicted_chars = [ int2char.get(int(ind), ' ') for ind in prob.indices ]

            result = "".join( predicted_chars )
            return result



@dataclass
class Tests:
    train_params: 'TrainParams' = TrainParams(epochs=100, lr=0.002)

    def test_phrase(self, data: Sequence[str], vocab: str, test_str: str) -> None:
        """ Test RNN to predict the following letters """

        char2int, int2char = CustomTextDataset.get_char_ind_map(vocab)
        transform_X = ( lambda s: s[:-1] )
        transform_y = ( lambda s: s[1:] )

        train_dataset = CustomTextDataset(data, char2int, transform_X=transform_X, transform_y=transform_y)

        model = CustomRNN(len(vocab), 10).to(TORCH_DEVICE)
        optimizer = Adam(model.parameters(), lr=0.005)

        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer)

        result = Processing.predict(model, int2char, char2int, test_str)
        print("PREDICTED PHRASE: '{}'". format(result))


    def test_caesar(self, data: Sequence[str], vocab: str, test_str: str) -> None:
        """ Test RNN with caesar encription """

        def _caesar_enc(alphabet: str, shift: int, s: str) -> str:
            shifted_alphabet = alphabet[shift:] + alphabet[:shift]
            return "".join( s.translate( str.maketrans(alphabet, shifted_alphabet) ) )

        char2int, int2char = CustomTextDataset.get_char_ind_map(vocab)
        transform_X = ( lambda s: _caesar_enc(vocab, 3, s) )

        train_dataset = CustomTextDataset(data, char2int, transform_X=transform_X)

        model = CustomRNN(len(vocab), 10).to(TORCH_DEVICE)
        optimizer = Adam(model.parameters(), lr=0.005)

        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=True)
        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer)

        result = Processing.predict_caesar(model, int2char, char2int, _caesar_enc(vocab, 3, test_str))  # maggie look wha
        print("PREDICTED PHRASE: '{}'". format(result))




def main():
    data = pd.read_csv('./data/data.csv')['normalized_text'].fillna('').str[:15].iloc[:100].tolist()
    vocab = ascii_lowercase

    tests_context = Tests()
    # tests_context.test_phrase(data, vocab, 'le')
    tests_context.test_caesar(data, vocab, 'maggie look wha')




if __name__ == '__main__':
    main()
