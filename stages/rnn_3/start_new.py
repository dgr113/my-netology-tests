# coding: utf-8

import re
import unicodedata

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

from io import open
from time import time
from math import floor
from pathlib import Path
from collections import Counter
from urllib import request
from zipfile import ZipFile
from functools import partial
from random import random, choice
from itertools import starmap
from dataclasses import dataclass, InitVar, field
from typing import Sequence, Tuple, Union, List, Dict, Type, Callable, Optional, Generator

from torch import Tensor, device, cuda, tensor, long as torch_long, zeros as torch_zeros, no_grad, relu
from torch.nn import Module, Embedding, Linear, RNNBase, LogSoftmax, NLLLoss, RNN, GRU, LSTM
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Optimizer
from torch.utils.data import Dataset, DataLoader

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )

UNI_PATH_TYPE = Union[Path, str]
UNI_NUM_TYPE = Union[int, float]
LANG_PAIR = Tuple[str, str]
COMMON_RNN_TYPE = Type[Union[RNN, GRU, LSTM]]
HIDDEN_TYPE = Union[Tensor, Tuple[Tensor, Tensor]]
LANG_WORDS_PAIR = Tuple[List[str], List[str]]




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
        es = s / percent
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
    def unicode_to_ascii(s: str) -> str:
        """ Turn a Unicode string to plain ASCII ( http://stackoverflow.com/a/518232/2809427 ) """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @classmethod
    def normalize_string(cls, s: str) -> str:
        s = cls.unicode_to_ascii( s.lower().strip() )
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^0-9a-zA-Zа-яА-Я.!?]+", r" ", s)
        return s



class Lang:
    def __init__(self, name: str, sos_ch: str = "SOS", eof_ch: str = "EOS"):
        self.name = name
        self.word_counter = Counter([sos_ch, eof_ch])
        self.word2index: Dict[str, int] = {}
        self.index2word = { 0: sos_ch, 1: eof_ch }
        self.n_words = len(self.index2word)  # Count SOS and EOS

    def _add_word(self, word: str) -> None:
        if word not in self.word_counter:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word_counter.update([word, ])

    def add_sentence(self, sentence: Sequence[str]) -> None:
        for word in sentence:
            self._add_word(word)



class CustomDataset(Dataset):
    def __init__(
            self,
            src_lang_name: str,
            target_lang_name: str,
            eos_token: int,
            data: Sequence[LANG_WORDS_PAIR],
            *,
            transform_X: Optional[Callable] = None,
            transform_y: Optional[Callable] = None
    ):
        self.transform_X = transform_X
        self.transform_y = transform_y

        self.input_lang: 'Lang' = Lang(src_lang_name)
        self.output_lang: 'Lang' = Lang(target_lang_name)
        self.eos_token = eos_token
        self.words2int: Dict[str, int] = {}

        self._data = data
        self._X_to_tensor = partial(CustomDataset.get_tensor_data, eos_token, self.input_lang)
        self._y_to_tensor = partial(CustomDataset.get_tensor_data, eos_token, self.output_lang)


    @staticmethod
    def pairs_generator(
            filename: UNI_PATH_TYPE,
            transform_func: Optional[Callable] = None,
            filter_func: Optional[Callable] = None,
            words_splitter: str = ' '

    ) -> Generator[LANG_WORDS_PAIR, None, None]:

        buffered_chars = ''
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.isspace():
                    full_line = ( buffered_chars + line.rstrip('\n') )
                    words_blocks = full_line.split('\t')
                    if transform_func:
                        words_blocks = list( map(transform_func, words_blocks) )

                    # if not filter_func or filter_func(words_blocks):
                    #     print("!!!", words_blocks)
                    yield words_blocks[0].split(words_splitter), words_blocks[1].split(words_splitter)


    @staticmethod
    def indexes_from_words(lang: 'Lang', words: Sequence[str]) -> List[int]:
        return [ lang.word2index[word] for word in words ]


    @staticmethod
    def get_tensor_data(eos_token: int, lang: 'Lang', words: Sequence[str], *, as_unsqueeze: bool = False) -> 'Tensor':
        indexes = CustomDataset.indexes_from_words(lang, words)
        indexes.append(eos_token)
        # t = tensor(indexes, dtype=torch_long, device=TORCH_DEVICE).view(-1, 1)
        t = tensor(indexes, dtype=torch_long, device=TORCH_DEVICE)
        return t.unsqueeze(0) if as_unsqueeze else t


    def __len__(self) -> int:
        return len(self._data)


    def __getitem__(self, idx: int) -> Tuple['Tensor', 'Tensor']:
        s = self._data[idx]

        X_ = self.transform_X(s) if self.transform_X else s
        y_ = self.transform_y(s) if self.transform_y else s


        print("!!!", X_, y_)


        self.input_lang.add_sentence( X_ )
        self.output_lang.add_sentence( y_ )

        X = self._X_to_tensor(X_)
        y = self._y_to_tensor(y_)
        return X, y



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
            decoder_input: 'Tensor',
            decoder_hidden: HIDDEN_TYPE,
            use_teacher_forcing: bool

    ) -> 'Tensor':

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                train_loss += self.loss(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                train_loss += self.loss(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break
        return train_loss


    def train_(self, loss, optimizer, input_tensor: 'Tensor', target_tensor: 'Tensor', max_length: int) -> float:
        encoder_hidden = self.encoder.init_hidden()

        optimizer.zero_grad()

        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_outputs = torch_zeros(max_length, self.encoder.hidden_size, device=TORCH_DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = tensor([[self.SOS_token]], device=TORCH_DEVICE)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random() < self.teacher_forcing_ratio else False

        train_loss = self._train_apply(loss, target_length, target_tensor, decoder_input, decoder_hidden, use_teacher_forcing)
        train_loss.backward()
        optimizer.step()

        return train_loss.item() / target_length



@dataclass
class TrainContext:
    dataset: InitVar['Dataset']
    max_length: int
    epochs: int
    batch_size: int
    batch_shuffle: bool
    model: 'Seq2Seq'
    optimizer: 'Optimizer'

    hidden_state_predict: bool = False  # use hiddent_state as predicted values

    _current_epoch: int = 0
    _epoch_loss: float = 0
    _current_accuracy: float = 0
    _correct_pred_count: int = 0
    _data: 'DataLoader' = field(init=False)

    def __post_init__(self, dataset: 'Dataset'):
        self._data = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.batch_shuffle)

    def __iter__(self):
        return self

    def __next__(self):
        start = time()
        plot_losses = []
        print_loss_total, plot_loss_total = 0.0, 0.0

        if self._current_epoch < self.epochs:
            self._epoch_start()

            train_loss = 0.0
            train_passed = 0
            for X_batch, y_batch in Processing.loader_to_device(self._data):
                pass
                # train_loss_value = self.model.train_(train_loss, self.optimizer, X_batch, y_batch, self.max_length)
                # print_loss_total += train_loss_value
                # plot_loss_total += train_loss_value
                # #     if itr % print_every == 0.0:
                # #         print_loss_avg = print_loss_total / print_every
                # #         print_loss_total = 0.0
                # #         time_diff = itr / n_iters
                # #         print('%s (%d %d%%) %.4f' % (TimeMeasure.time_since(start, time_diff), itr, itr / n_iters * 100, print_loss_avg))
                # #
                # #     if itr % plot_every == 0:
                # #         plot_loss_avg = plot_loss_total / plot_every
                # #         plot_losses.append(plot_loss_avg)
                # #         plot_loss_total = 0.0
                # #
                # # Visualize.show_plot(plot_losses)
                # train_passed += 1

            print( "Train loss: {:.3f}".format(train_loss / train_passed or 1) )
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
            max_length: int,
            epochs: int,
            batch_size: int,
            batch_shuffle: bool,
            model: 'Seq2Seq',
            optimizer: 'Optimizer',
            hidden_state_predict: bool = False
    ) -> None:
        for _ in TrainContext(dataset, max_length, epochs, batch_size, batch_shuffle, model, optimizer, hidden_state_predict):
            pass

    @staticmethod
    def predict():
        pass



# @dataclass
# class EvalContext:
#     SOS_token: int
#     EOS_token: int
#
#     def get_decoded_words(self, output_lang: 'Lang', decoder: 'DecoderRNN', decoder_hidden: HIDDEN_TYPE, max_length: int) -> List[str]:
#         decoder_input = tensor([[self.SOS_token]], device=TORCH_DEVICE)  # SOS
#         decoded_words = []
#         for di in range(max_length):
#             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == self.EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             else:
#                 decoded_words.append(output_lang.index2word[topi.item()])
#
#             decoder_input = topi.squeeze().detach()
#         return decoded_words
#
#
#     def evaluate(
#         self,
#         input_lang: 'Lang',
#         output_lang: 'Lang',
#         encoder: 'EncoderRNN',
#         decoder: 'DecoderRNN',
#         sentence: str,
#         max_length: int
#
#     ) -> List[str]:
#
#         with no_grad():
#             input_tensor = ConvertFrom.tensor_from_sentence(input_lang, sentence, self.EOS_token)
#             input_length = input_tensor.size()[0]
#             encoder_hidden = encoder.init_hidden()
#
#             encoder_outputs = torch_zeros(max_length, encoder.hidden_size, device=TORCH_DEVICE)
#
#             for ei in range(input_length):
#                 encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
#                 encoder_outputs[ei] += encoder_output[0, 0]
#
#             decoded_words = self.get_decoded_words(output_lang, decoder, encoder_hidden, max_length)
#             return decoded_words
#
#
#     def evaluate_randomly(
#         self,
#         max_length: int,
#         pairs: Sequence[LANG_PAIR],
#         input_lang: 'Lang',
#         output_lang: 'Lang',
#         encoder: 'EncoderRNN',
#         decoder: 'DecoderRNN',
#         n: int = 10
#
#     ) -> None:
#
#         for i in range(n):
#             pair = choice(pairs)
#             print('>', pair[0])
#             print('=', pair[1])
#             output_words = self.evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length)
#             output_sentence = ' '.join(output_words)
#             print('<', output_sentence)
#             print('')





def main():
    SOS_INDEX, EOS_INDEX = 0, 1
    LEARNING_RATE = 0.01
    EPOCHS = 10
    BATCH_SIZE = 256

    RNN_TYPE = GRU
    MAX_LENGTH = 10
    HIDDEN_SIZE = 256
    HIDDEN_LAYERS_COUNT = 1

    ENG_PREFIXES = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    # # ENG => FRA
    # FileUtils.load_archive('https://download.pytorch.org/tutorial/data.zip', './')
    # lang_file_path = './data/eng-fra.txt'
    # lang_first_name = 'eng'
    # lang_second_name = 'fra'

    # # RUS => ENG
    FileUtils.load_archive('https://www.manythings.org/anki/rus-eng.zip', './')
    lang_file_path = './rus.txt'
    lang_first_name = 'rus'
    lang_second_name = 'eng'


    TEXT_TRANSFROM = ( lambda s: PrepareData.normalize_string( s.lower() ) )
    TEXT_FILTER = ( lambda words: words[0].startswith(ENG_PREFIXES) )
    X_TRANSFORM = ( lambda pair: pair[0] )
    Y_TRANSFORM = ( lambda pair: pair[1] )

    data = CustomDataset.pairs_generator(lang_file_path, transform_func=TEXT_TRANSFROM, filter_func=TEXT_FILTER)
    # print(len(list(data)))

    train_dataset = CustomDataset(lang_first_name, lang_second_name, EOS_INDEX, list(data), transform_X=X_TRANSFORM, transform_y=Y_TRANSFORM)


    input_lang_words_count = len( train_dataset.input_lang.word_counter )
    output_lang_words_count = len( train_dataset.output_lang.word_counter )

    encoder = EncoderRNN(RNN_TYPE, HIDDEN_SIZE, input_lang_words_count, HIDDEN_LAYERS_COUNT).to(TORCH_DEVICE)
    decoder = DecoderRNN(RNN_TYPE, HIDDEN_SIZE, output_lang_words_count, HIDDEN_LAYERS_COUNT).to(TORCH_DEVICE)
    loss = NLLLoss()

    seq2seq = Seq2Seq(encoder, decoder, loss, SOS_INDEX, EOS_INDEX)
    optimizer = SGD(seq2seq.parameters(), lr=LEARNING_RATE)



    for X, y in train_dataset:
        if X.shape[0] > 2 or y.shape[0] > 2:
            print(X.shape, y.shape)



    # Processing.train_model(train_dataset, MAX_LENGTH, EPOCHS, BATCH_SIZE, True, seq2seq, optimizer, False)



    # eval_context = EvalContext(SOS_INDEX, EOS_INDEX)
    # eval_context.evaluate_randomly(MAX_LENGTH, pairs, input_lang, output_lang,  encoder, decoder)



if __name__ == '__main__':
    main()
