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
from dataclasses import dataclass
from typing import Sequence, Tuple, Union, List, Dict, Type

from torch import Tensor, device, cuda, tensor, long as torch_long, zeros as torch_zeros, no_grad, relu
from torch.nn import Module, Embedding, Linear, RNNBase, LogSoftmax, NLLLoss, RNN, GRU, LSTM
from torch.optim import SGD, Optimizer

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )

UNI_PATH_TYPE = Union[Path, str]
UNI_NUM_TYPE = Union[int, float]
LANG_PAIR = Tuple[str, str]
COMMON_RNN_TYPE = Type[RNNBase]
HIDDEN_TYPE = Union[Tensor, Tuple[Tensor, Tensor]]




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

    def add_sentence(self, sentence: str, splitter: str = ' ') -> None:
        for word in sentence.split(splitter):
            self._add_word(word)

    @staticmethod
    def read_langs(lang_file_path: UNI_PATH_TYPE, lang1: str, lang2: str, reverse: bool = False, lang_split_ch: str = '\t') -> Tuple['Lang', 'Lang', list]:
        lines = open(Path(lang_file_path), encoding='utf-8').read().strip().split('\n')  # Read the file and split into lines

        pairs = [
            [ PrepareData.normalize_string(s) for s in l.split(lang_split_ch)[:2] ]
            for l in lines
        ]

        if reverse:
            pairs = [ list(reversed(p)) for p in pairs ]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs



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

    @staticmethod
    def filter_pair(max_length: int, eng_prefixes: Tuple[str], pair: Tuple[str, str]) -> bool:
        p_0, p_1 = pair[0], pair[1]
        len_check = all( ln < max_length for ln in map(len, map(str.split, pair)) )
        return len_check and p_1.startswith(eng_prefixes)

    @classmethod
    def filter_pairs(cls, pairs: Sequence[Tuple[str, str]], max_length: int, eng_prefixes: Tuple[str, ...]) -> Sequence[Tuple[str, str]]:
        filter_func = partial(cls.filter_pair, max_length, eng_prefixes)
        return list( filter(filter_func, pairs) )

    @classmethod
    def prepare_data(
        cls,
        input_lang: 'Lang',
        output_lang: 'Lang',
        pairs: Sequence[Tuple[str, str]],
        max_length: int,
        eng_prefixes: Tuple[str, ...]

    ) -> Tuple['Lang', 'Lang', Sequence[Tuple[str, str]]]:

        pairs = cls.filter_pairs(pairs, max_length, eng_prefixes)
        for pair in pairs:
            input_lang.add_sentence( pair[0] )
            output_lang.add_sentence( pair[1] )

        return input_lang, output_lang, pairs



class ConvertFrom:
    @staticmethod
    def indexes_from_sentence(lang: 'Lang', sentence) -> list:
        return [ lang.word2index[word] for word in sentence.split(' ') ]

    @classmethod
    def tensor_from_sentence(cls, lang: 'Lang', sentence, eos_token: int) -> 'Tensor':
        indexes = cls.indexes_from_sentence(lang, sentence)
        indexes.append(eos_token)
        return tensor(indexes, dtype=torch_long, device=TORCH_DEVICE).view(-1, 1)

    @classmethod
    def tensors_from_pair(cls, pair: LANG_PAIR, input_lang: 'Lang', output_lang: 'Lang', eos_token: int) -> Tuple['Tensor', 'Tensor']:
        input_tensor = cls.tensor_from_sentence(input_lang, pair[0], eos_token)
        target_tensor = cls.tensor_from_sentence(output_lang, pair[1], eos_token)
        return input_tensor, target_tensor



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



@dataclass
class TrainContext:
    model: Seq2Seq
    optimizer: 'Optimizer'

    def train_model(
        self,
        max_length: int,
        pairs: Sequence[LANG_PAIR],
        input_lang: 'Lang',
        output_lang: 'Lang',
        n_iters: int,
        print_every: int = 1000,
        plot_every: int = 100

    ) -> None:

        start = time()
        plot_losses = []
        print_loss_total, plot_loss_total = 0.0, 0.0

        training_pairs = [
            ConvertFrom.tensors_from_pair(choice(pairs), input_lang, output_lang, self.model.EOS_token)
            for _ in range(n_iters)
        ]

        train_loss_value = 0.0
        for itr in range(1, n_iters + 1):
            training_pair = training_pairs[itr - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            train_loss_value = self.model.train_(train_loss_value, self.optimizer, input_tensor, target_tensor, max_length)
            print_loss_total += train_loss_value
            plot_loss_total += train_loss_value

            if itr % print_every == 0.0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0.0
                time_diff = itr / n_iters
                print('%s (%d %d%%) %.4f' % (TimeMeasure.time_since(start, time_diff), itr, itr / n_iters * 100, print_loss_avg))

            if itr % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0.0

        Visualize.show_plot(plot_losses)



@dataclass
class EvalContext:
    SOS_token: int
    EOS_token: int

    def get_decoded_words(self, output_lang: 'Lang', decoder: 'DecoderRNN', decoder_hidden: HIDDEN_TYPE, max_length: int) -> List[str]:
        decoder_input = tensor([[self.SOS_token]], device=TORCH_DEVICE)  # SOS
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == self.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoded_words


    def evaluate(
        self,
        input_lang: 'Lang',
        output_lang: 'Lang',
        encoder: 'EncoderRNN',
        decoder: 'DecoderRNN',
        sentence: str,
        max_length: int

    ) -> List[str]:

        with no_grad():
            input_tensor = ConvertFrom.tensor_from_sentence(input_lang, sentence, self.EOS_token)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.init_hidden()

            encoder_outputs = torch_zeros(max_length, encoder.hidden_size, device=TORCH_DEVICE)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoded_words = self.get_decoded_words(output_lang, decoder, encoder_hidden, max_length)
            return decoded_words


    def evaluate_randomly(
        self,
        max_length: int,
        pairs: Sequence[LANG_PAIR],
        input_lang: 'Lang',
        output_lang: 'Lang',
        encoder: 'EncoderRNN',
        decoder: 'DecoderRNN',
        n: int = 10

    ) -> None:

        for i in range(n):
            pair = choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words = self.evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')





def main():
    SOS_INDEX, EOS_INDEX = 0, 1
    LEARNING_RATE = 0.01

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
    # lang_one_name = 'eng'
    # lang_two_name = 'fra'

    # # RUS => ENG
    FileUtils.load_archive('https://www.manythings.org/anki/rus-eng.zip', './')
    lang_file_path = './rus.txt'
    lang_one_name = 'rus'
    lang_two_name = 'eng'


    input_lang, output_lang, pairs = Lang.read_langs(lang_file_path, lang_one_name, lang_two_name, True)
    input_lang, output_lang, pairs = PrepareData.prepare_data(input_lang, output_lang, pairs, MAX_LENGTH, ENG_PREFIXES)

    input_lang_words_count = len(input_lang.word_counter)
    output_lang_words_count = len(output_lang.word_counter)

    encoder = EncoderRNN(RNN_TYPE, HIDDEN_SIZE, input_lang_words_count, HIDDEN_LAYERS_COUNT).to(TORCH_DEVICE)
    decoder = DecoderRNN(RNN_TYPE, HIDDEN_SIZE, output_lang_words_count, HIDDEN_LAYERS_COUNT).to(TORCH_DEVICE)
    loss = NLLLoss()

    seq2seq = Seq2Seq(encoder, decoder, loss, SOS_INDEX, EOS_INDEX)
    optimizer = SGD(seq2seq.parameters(), lr=LEARNING_RATE)

    train_context = TrainContext(seq2seq, optimizer)
    train_context.train_model(MAX_LENGTH, pairs, input_lang, output_lang, 75000, print_every=5000)

    eval_context = EvalContext(SOS_INDEX, EOS_INDEX)
    eval_context.evaluate_randomly(MAX_LENGTH, pairs, input_lang, output_lang,  encoder, decoder)




if __name__ == '__main__':
    main()
