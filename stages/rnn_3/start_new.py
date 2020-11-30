# coding: utf-8

import re
import unicodedata

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

from time import time
from math import floor
from pathlib import Path
from functools import partial
from operator import attrgetter
from urllib import request
from zipfile import ZipFile
from itertools import starmap, islice
from random import random
from dataclasses import dataclass, InitVar, field
from typing import Tuple, Union, List, Type, Generator, Sequence, Iterator, Optional

from torch import Tensor, device, cuda, tensor, zeros as torch_zeros, relu, no_grad
from torch.nn import Module, Embedding, Linear, LogSoftmax, NLLLoss, RNN, GRU, LSTM
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torchtext.data import Dataset, TabularDataset, Field, BucketIterator, Example  # type: ignore

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )

UNI_PATH_TYPE = Union[Path, str]
UNI_NUM_TYPE = Union[int, float]
COMMON_RNN_TYPE = Type[Union[RNN, GRU, LSTM]]
HIDDEN_TYPE = Union[Tensor, Tuple[Tensor, Tensor]]




class TimeMeasure:
    @staticmethod
    def as_minutes(s: float) -> str:
        m = floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)

    @classmethod
    def time_since(cls, since: UNI_NUM_TYPE, percent: UNI_NUM_TYPE) -> str:
        now = time()
        s = now - since
        es = s / ( percent or 1 )
        rs = es - s
        return "{} (- {})".format(cls.as_minutes(s), cls.as_minutes(rs))



class Visualize:
    @staticmethod
    def show_plot(points):
        plt.figure()
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)



class FileUtils:
    @staticmethod
    def _download_archive(url: str, tmp_file_path: UNI_PATH_TYPE) -> None:
        opener = request.build_opener()
        opener.addheaders = []
        request.install_opener(opener)
        request.urlretrieve(url, tmp_file_path)

    @staticmethod
    def load_archive(url: str, save_dir: UNI_PATH_TYPE, tmp_file_path: UNI_PATH_TYPE = './tmp.archive') -> None:
        tmp_file_path = Path(tmp_file_path)
        save_dir = Path(save_dir)

        FileUtils._download_archive(url, tmp_file_path)
        with ZipFile(tmp_file_path, 'r') as zf:
            zf.extractall(save_dir)

        tmp_file_path.unlink()



class PrepareData:
    @staticmethod
    def _unicode_to_ascii(s: str) -> str:
        """ Turn a Unicode string to plain ASCII ( http://stackoverflow.com/a/518232/2809427 ) """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @classmethod
    def normalize_string(cls, s: str, *, as_tokens: bool = True, splitter: str = ' ') -> Union[str, Sequence[str]]:
        s = cls._unicode_to_ascii( s.lower().strip() )
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^0-9a-zA-Zа-яА-Я.!?]+", r" ", s)
        return s.split(splitter) if as_tokens else s

    @staticmethod
    def filter_dataset(first_lang_name: str, second_lang_name: str, data_example: 'Example', *, max_length: Optional[int] = None) -> bool:
        p_0, p_1 = getattr(data_example, first_lang_name), getattr(data_example, second_lang_name)
        len_check_result = all( ln < max_length for ln in map(len, (p_0, p_1)) ) if max_length else True
        return len_check_result



class EncoderRNN(Module):
    def __init__(self, rnn_class: COMMON_RNN_TYPE, hidden_size: int, input_size: int, num_layers: int = 1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = Embedding(input_size, hidden_size)
        self.rnn = rnn_class(hidden_size, hidden_size, num_layers)

    def forward(self, inp: 'Tensor', hidden: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        output = self.embedding(inp).view(1, 1, -1)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def init_hidden(self) -> HIDDEN_TYPE:
        t = torch_zeros(self.num_layers, 1, self.hidden_size, device=TORCH_DEVICE)
        return (t, t) if ( self.rnn.__class__.__name__ == 'LSTM' ) else t



class DecoderRNN(Module):
    def __init__(self, rnn_class: COMMON_RNN_TYPE, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = Embedding(output_size, hidden_size)
        self.rnn = rnn_class(hidden_size, hidden_size, num_layers)
        self.out = Linear(hidden_size, output_size)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, inp: 'Tensor', hidden: Tensor) -> Tuple['Tensor', 'Tensor']:
        output = inp.view(1, -1)
        output = self.embedding(output)
        output = relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self) -> HIDDEN_TYPE:
        t = torch_zeros(self.num_layers, 1, self.hidden_size, device=TORCH_DEVICE)
        return (t, t) if ( self.rnn.__class__.__name__ == 'LSTM' ) else t



class Seq2Seq(Module):
    def __init__(self, encoder: 'EncoderRNN', decoder: 'DecoderRNN', loss, SOS_token: int, EOS_token: int, teacher_forcing_ratio: float = 0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss

        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.teacher_forcing_ratio = teacher_forcing_ratio


    def _train_apply(
        self,
        train_loss: 'Tensor',
        target_length: int,
        target_tensor: 'Tensor',
        decoder_hidden: HIDDEN_TYPE,
        use_teacher_forcing: bool

    ) -> 'Tensor':

        decoder_input = tensor([[self.SOS_token]], device=TORCH_DEVICE)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                train_loss += self.loss(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                train_loss += self.loss(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break
        return train_loss


    def train_start(self, loss, optimizer, input_tensor: 'Tensor', target_tensor: 'Tensor', max_length: int) -> float:
        encoder_hidden = self.encoder.init_hidden()

        optimizer.zero_grad()

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]

        # Vector that stores embedded outputs of all words from <input_tensor>
        encoder_outputs = torch_zeros(max_length, self.encoder.hidden_size, device=TORCH_DEVICE)

        # Encoding of each word vector from <input_tensor>
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        use_teacher_forcing = True if random() < self.teacher_forcing_ratio else False

        train_loss = self._train_apply(loss, target_length, target_tensor, encoder_hidden, use_teacher_forcing)
        train_loss.backward()
        optimizer.step()

        return train_loss.item() / target_length



@dataclass
class TrainContext:
    dataset: InitVar['Dataset']
    x_name: str
    y_name: str
    max_length: int
    epochs: int
    max_iters: int  # Max iters in every epoch
    batch_shuffle: bool
    model: 'Seq2Seq'
    optimizer: 'Optimizer'

    hidden_state_predict: bool = False  # use hiddent_state as predicted values
    print_every: int = 5000
    plot_every: int = 100

    _current_epoch: int = 0
    _epoch_loss: float = 0
    _current_accuracy: float = 0
    _correct_pred_count: int = 0
    _data: 'Iterator' = field(init=False)

    def __post_init__(self, dataset: 'Dataset'):
        self._data = map(
            attrgetter(self.x_name, self.y_name),
            BucketIterator(dataset, batch_size=1, shuffle=self.batch_shuffle)
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_epoch < self.epochs:
            start = time()
            plot_losses = []
            print_loss_total, plot_loss_total = 0.0, 0.0
            self._epoch_start()

            train_loss_value = 0.0
            for i, (X_batch, y_batch) in enumerate(islice(self._data, 0, self.max_iters), 1):
                train_loss_value = self.model.train_start(train_loss_value, self.optimizer, X_batch.to(TORCH_DEVICE), y_batch.to(TORCH_DEVICE), self.max_length)
                print_loss_total += train_loss_value
                plot_loss_total += train_loss_value

                if i % self.print_every == 0.0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0.0
                    time_diff = i / self.max_iters
                    time_diff_percent = time_diff * 100
                    print('%s (%d %d%%) %.4f' % (TimeMeasure.time_since(start, time_diff), i, time_diff_percent, print_loss_avg))

                if i % self.plot_every == 0:
                    plot_loss_avg = plot_loss_total / self.plot_every
                    plot_losses.append( plot_loss_avg )
                    plot_loss_total = 0.0

            Visualize.show_plot(plot_losses)
            return True
        else:
            raise StopIteration

    def _epoch_start(self) -> None:
        self._current_epoch += 1
        self._epoch_loss = 0
        self._correct_pred_count = 0



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
        x_name: str,
        y_name: str,
        max_length: int,
        epochs: int,
        max_iters: int,
        batch_shuffle: bool,
        model: 'Seq2Seq',
        optimizer: 'Optimizer',
        hidden_state_predict: bool = False
    ) -> None:
        for _ in TrainContext(dataset, x_name, y_name, max_length, epochs, max_iters, batch_shuffle, model, optimizer, hidden_state_predict):
            pass



@dataclass
class EvalContext:
    @staticmethod
    def _get_decoded_words(decoder: 'DecoderRNN', output_lang_field: 'Field', decoder_hidden: HIDDEN_TYPE, max_length: int) -> List[str]:
        SOS_token = output_lang_field.vocab.stoi[output_lang_field.init_token]
        EOS_token = output_lang_field.vocab.stoi[output_lang_field.eos_token]

        decoder_input = tensor([[SOS_token]], device=TORCH_DEVICE)
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            topi_value = topi.item()  # Top index of word value

            if topi_value == EOS_token:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append( output_lang_field.vocab.itos[topi_value] )

            decoder_input = topi.squeeze().detach()
        return decoded_words

    @staticmethod
    def evaluate_apply(
        encoder: 'EncoderRNN',
        decoder: 'DecoderRNN',
        input_lang: Tuple[str, 'Field'],
        output_lang: Tuple[str, 'Field'],
        sentence: str,
        max_length: int

    ) -> List[str]:

        input_lang_name, input_lang_field = input_lang
        output_lang_name, output_lang_field = output_lang

        with no_grad():
            input_tensor = input_lang_field.process([sentence, ]).to(TORCH_DEVICE)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.init_hidden()
            encoder_outputs = torch_zeros(max_length, encoder.hidden_size, device=TORCH_DEVICE)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoded_words = EvalContext._get_decoded_words(decoder, output_lang_field, encoder_hidden, max_length)
            return decoded_words

    @staticmethod
    def evaluate_randomly(
        max_length: int,
        dataset: 'Dataset',
        input_lang: Tuple[str, 'Field'],
        output_lang: Tuple[str, 'Field'],
        encoder: 'EncoderRNN',
        decoder: 'DecoderRNN',

    ) -> None:

        input_lang_name, input_lang_field = input_lang
        output_lang_name, output_lang_field = output_lang

        for src, trg in map(attrgetter(input_lang_name, output_lang_name), dataset):
            src_str, trg_str = map(' '.join, [src, trg])
            print('>', src_str)
            print('=', trg_str)
            output_words = EvalContext.evaluate_apply(encoder, decoder, input_lang, output_lang, src, max_length)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')




def main():
    SOS_INDEX, EOS_INDEX = 0, 1
    LEARNING_RATE = 0.01
    EPOCHS = 1
    TRAIN_ITERS = 75000

    RNN_TYPE = GRU
    MAX_LENGTH = 10
    HIDDEN_SIZE = 256
    HIDDEN_LAYERS_COUNT = 1

    # # RUS => ENG
    FileUtils.load_archive('https://www.manythings.org/anki/rus-eng.zip', './')
    lang_file_path = './rus.txt'
    lang_first_name = 'rus'
    lang_second_name = 'eng'


    dataset_filter_func = partial(PrepareData.filter_dataset, lang_first_name, lang_second_name, max_length=MAX_LENGTH)

    first_lang = Field(lang_first_name, init_token='SOS', eos_token='EOS', fix_length=MAX_LENGTH, lower=True, tokenize=PrepareData.normalize_string)
    second_lang = Field(lang_second_name, init_token='SOS', eos_token='EOS', fix_length=MAX_LENGTH, lower=True, tokenize=PrepareData.normalize_string)
    first_lang_attr = (lang_first_name, first_lang)
    second_lang_attr = (lang_second_name, second_lang)

    dataset = TabularDataset(path=lang_file_path, format='tsv', fields=[first_lang_attr, second_lang_attr], filter_pred=dataset_filter_func)
    first_lang.build_vocab(dataset, max_size=400)
    second_lang.build_vocab(dataset, max_size=800)

    test_dataset, train_dataset = dataset.split(0.0001)

    encoder = EncoderRNN(RNN_TYPE, HIDDEN_SIZE, len( first_lang.vocab ), HIDDEN_LAYERS_COUNT).to(TORCH_DEVICE)
    decoder = DecoderRNN(RNN_TYPE, HIDDEN_SIZE, len( second_lang.vocab ), HIDDEN_LAYERS_COUNT).to(TORCH_DEVICE)
    loss = NLLLoss()

    seq2seq = Seq2Seq(encoder, decoder, loss, SOS_INDEX, EOS_INDEX)
    optimizer = SGD(seq2seq.parameters(), lr=LEARNING_RATE)


    Processing.train_model(train_dataset, lang_first_name, lang_second_name, MAX_LENGTH, EPOCHS, TRAIN_ITERS, True, seq2seq, optimizer)
    EvalContext.evaluate_randomly(MAX_LENGTH, test_dataset, first_lang_attr, second_lang_attr, encoder, decoder)



if __name__ == '__main__':
    main()
