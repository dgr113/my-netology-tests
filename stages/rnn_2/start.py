# coding: utf-8

import re
import pandas as pd  # type: ignore

from pathlib import Path
from pprint import pprint
from functools import partial
from itertools import starmap
from dataclasses import dataclass, InitVar, field
from string import ascii_lowercase
from urllib.request import urlretrieve
from random import randint, choice as random_choice
from typing import Generator, Optional, Dict, Sequence, Union, List, Callable, Tuple, Type

from torch import Tensor, no_grad, device, cuda, softmax, distributions, max as torch_max  # type: ignore
from torch.nn import Module, Embedding, Linear, RNN, GRU, LSTM, RNNBase
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader  # type: ignore

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )


UNI_NUM_TYPE = Union[int, float]
UNI_CHARS_TYPE = Union[str, List[str]]
UNI_PATH_TYPE = Union[str, Path]
CHARS_DATA_TRANSFORM_FUNC = Callable[[str], str]
NUM_DATA_TRANSFORM_FUNC = Callable[[Sequence[UNI_NUM_TYPE]], Sequence[UNI_NUM_TYPE]]
COMMON_RNN_TYPE = Type[RNNBase]




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
        X_doc_len: int,
        y_doc_len: int,
        *,
        transform_X: Optional[CHARS_DATA_TRANSFORM_FUNC] = None,
        transform_y: Optional[CHARS_DATA_TRANSFORM_FUNC] = None,
        X_len_align: Optional[int] = None,
        y_len_align: Optional[int] = None
    ):
        self.X_doc_len = X_doc_len
        self.y_doc_len = y_doc_len
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.X_len_align = X_len_align
        self.y_len_align = y_len_align
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

    @staticmethod
    def batch_generator(
        filename: UNI_PATH_TYPE,
        batch_size: int,
        batch_shift: int,
        transform_func: Optional[CHARS_DATA_TRANSFORM_FUNC] = None

    ) -> Generator[str, None, None]:

        current_row = 0
        buffered_chars = ''
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.isspace():
                    full_line = ( buffered_chars + line.rstrip('\n') )
                    if transform_func:
                        full_line = transform_func(full_line)

                    _curr_start_pos = 0
                    while True:
                        _curr_end_pos = batch_size+_curr_start_pos
                        part_of_line = full_line[_curr_start_pos:_curr_end_pos]
                        if len(part_of_line) < batch_size:
                            buffered_chars = "{} ".format(part_of_line)
                            break
                        else:
                            _curr_start_pos += batch_shift
                            current_row += 1
                            yield part_of_line

    @staticmethod
    def align_doc_len(s: str, doc_len: int) -> str:
        return s.ljust(doc_len)[:doc_len]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple['Tensor', 'Tensor']:
        s = self._data[idx]
        X = self._to_tensor( self.align_doc_len(( self.transform_X(s) if self.transform_X else s ), self.X_doc_len) )
        y = self._to_tensor( self.align_doc_len(( self.transform_y(s) if self.transform_y else s ), self.y_doc_len) )
        return X, y



class CustomNumDataset(Dataset):
    def __init__(
        self,
        data: Sequence[Sequence[UNI_NUM_TYPE]],
        *,
        transform_X: Optional[NUM_DATA_TRANSFORM_FUNC] = None,
        transform_y: Optional[NUM_DATA_TRANSFORM_FUNC] = None
    ):
        self.transform_X = transform_X
        self.transform_y = transform_y
        self._data = data
        self._to_tensor = CustomNumDataset.get_tensor_data

    @staticmethod
    def get_tensor_data(chars: Sequence[UNI_NUM_TYPE], *, as_unsqueeze: bool = False) -> 'Tensor':
        t = Tensor( chars ).long()
        return t.unsqueeze(0) if as_unsqueeze else t

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple['Tensor', 'Tensor']:
        s = self._data[idx]
        X = self._to_tensor( self.transform_X(s) if self.transform_X else s )
        y = self._to_tensor( self.transform_y(s) if self.transform_y else s )
        return X, y.flatten(0)



@dataclass
class TrainContext:
    dataset: InitVar['Dataset']
    train_params: 'TrainParams'
    loader_params: 'DataLoaderParams'
    model: 'Module'
    optimizer: 'Optimizer'

    hidden_state_predict: bool = False  # use hiddent_state as predicted values

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
                outputs, hidden_state = self.model(X_batch)
                X_predicted = hidden_state if self.hidden_state_predict else outputs

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
        """ Update inner epoch stats """
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
    def batch_to_device(X: 'Tensor', y: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        return X.to(TORCH_DEVICE), y.to(TORCH_DEVICE)

    @staticmethod
    def loader_to_device(data: 'DataLoader') -> Generator[Tuple['Tensor', 'Tensor'], None, None]:
        yield from starmap(Processing.batch_to_device, data)

    @staticmethod
    def train_model(
        dataset: 'Dataset',
        train_params: 'TrainParams',
        loader_params: 'DataLoaderParams',
        model: 'Module',
        optimizer: 'Optimizer',
        hidden_state_predict: bool = False
    ) -> None:
        for epoch_stats in TrainContext(dataset, train_params, loader_params, model, optimizer, hidden_state_predict):
            # print( "Epoch train stats: {}".format(epoch_stats) )
            pass

    @staticmethod
    def predict_sample(preds: 'Tensor') -> 'Tensor':
        softmaxed = softmax(preds, 0)
        probas = distributions.multinomial.Multinomial(1, softmaxed).sample()
        return probas.argmax()

    @staticmethod
    def predict_consecutively(model: 'Module', int2char: Dict[int, str], char2int: Dict[str, int], chars: str, default_char: str = ' ') -> str:
        MAX_LEN = len(chars)
        model.to(TORCH_DEVICE)
        model.eval()

        with no_grad():
            generated = chars
            for i in range(MAX_LEN):
                X = CustomTextDataset.get_tensor_data(char2int, generated[-MAX_LEN:], as_unsqueeze=True).to(TORCH_DEVICE)
                preds = model( X )  # Predicting model
                next_ind = int( Processing.predict_sample( preds[0] ) )  # Первый элемент, так как он всего один
                next_char = int2char.get(next_ind, default_char)
                generated += next_char

            return generated[:MAX_LEN] + ' | ' + generated[MAX_LEN:]

    @staticmethod
    def predict_at_once(model: 'Module', seq: Sequence[int]):
        model.to(TORCH_DEVICE)
        model.eval()

        with no_grad():
            X = CustomNumDataset.get_tensor_data(seq, as_unsqueeze=True).to(TORCH_DEVICE)
            preds, _ = model( X )  # Predicting model
            softmaxed = torch_max(preds, dim=1)
            return softmaxed



class CustomCommonRNN(Module):
    def __init__(self, rnn_class: COMMON_RNN_TYPE, input_size: int, emb_size: int, hidden_size: int = 128):
        super().__init__()
        rnn_class_name = rnn_class.__name__

        self.input_size, self.output_size = [ input_size ] * 2  # Alignment for case when the vocab length is passed
        self.emb = Embedding(self.input_size, emb_size)
        self.rnn = rnn_class(rnn_class_name, emb_size, hidden_size, batch_first=True)
        self.out = Linear(hidden_size, self.output_size)

    def forward(self, out: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        out = self.emb(out)
        out, hidden = self.rnn(out)
        out, hidden = self.out(out), self.out(hidden[0])
        out = out.view(-1, self.output_size)  # Flatten output
        return out, hidden



@dataclass
class Tests:
    train_params: 'TrainParams' = TrainParams(epochs=100, lr=0.005)

    @staticmethod
    def _get_test_values(row: Sequence[UNI_NUM_TYPE]) -> Sequence[UNI_NUM_TYPE]:
        gen = iter(row)
        first_v = next(gen)
        return [
            first_v,
            *(
                y - 10 if y >= 10 else y for y in
                ( (x + first_v) for x in gen )
            )
        ]


    def test_practice(self) -> None:
        urlretrieve('https://s3.amazonaws.com/text-datasets/nietzsche.txt', './nietzsche.txt')

        TEXT_CLEAR_PATT = re.compile(r'[^a-zA-Z]')
        TEXT_TRANSFROM = ( lambda s: ' '.join(TEXT_CLEAR_PATT.sub(' ', s).split()).lower() )
        X_TRANSFORM = ( lambda s: s[:-1] )
        Y_TRANSFORM = ( lambda s: s[-1:] )

        df = pd.DataFrame( CustomTextDataset.batch_generator('./nietzsche.txt', 41, 3, TEXT_TRANSFROM) )
        data = df[0].tolist()

        vocab = ascii_lowercase
        char2int, int2char = CustomTextDataset.get_char_ind_map(vocab)
        train_dataset = CustomTextDataset(data, char2int, 40, 1, transform_X=X_TRANSFORM, transform_y=Y_TRANSFORM)

        model = CustomCommonRNN(LSTM, len(vocab), 128).to(TORCH_DEVICE)
        optimizer = Adam(model.parameters(), lr=0.005)
        train_loader_params = DataLoaderParams(batch_size=512, batch_shuffle=True)
        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer, True)

        sample_phrase = df.sample(1).iloc[0, 0]
        predicted_phrase = Processing.predict_consecutively(model, int2char, char2int, sample_phrase)
        pprint(
            predicted_phrase,
            compact=True
        )


    def test_homework(self) -> None:
        PROBES_COUNT = 1000
        SEQ_LEN = 40
        X_MIN = 0
        X_MAX = 9

        data = [ [ randint(X_MIN, X_MAX) for _ in range(SEQ_LEN) ] for _ in range(PROBES_COUNT) ]
        train_dataset = CustomNumDataset(data, transform_y=Tests._get_test_values)

        for rnn_class in [RNN, GRU, LSTM]:
            model = CustomCommonRNN(rnn_class, 10, 128).to(TORCH_DEVICE)
            optimizer = Adam(model.parameters(), lr=0.005)
            train_loader_params = DataLoaderParams(batch_size=200, batch_shuffle=True)

            Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer, False)

            test_samples = random_choice(data)
            target_samples = Tests._get_test_values(test_samples)
            predicted = Processing.predict_at_once(model, test_samples)
            pprint({
                rnn_class.__name__: {
                    "TARG": target_samples,
                    "PRED": predicted.indices.tolist()
                }
            }, compact=True)





def main():
    tests_context = Tests()
    tests_context.test_homework()




if __name__ == '__main__':
    main()
