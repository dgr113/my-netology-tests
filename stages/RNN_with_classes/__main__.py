# coding: utf-8

import pandas as pd  # type: ignore

from string import ascii_lowercase
from typing import Sequence, Iterable
from dataclasses import dataclass

from torch import device, cuda  # type: ignore
from torch.optim import Adam  # type: ignore

from stages.RNN_with_classes.helpers import TrainParams, CustomTextDataset, DataLoaderParams
from stages.RNN_with_classes.models import CustomEncoder, CustomDecoder, Seq2Seq
from stages.RNN_with_classes.processing import Processing

TORCH_DEVICE = device( 'cuda' if cuda.is_available() else 'cpu' )  # USE CUDA GPU




@dataclass
class ModelsTests:
    train_params: 'TrainParams' = TrainParams(epochs=100, lr=0.002)

    def test_two(self) -> None:
        """ Test RNN with caesar encription """
        def caesar_enc(s: Iterable[str], shift: int = 3) -> Sequence[str]:
            if not isinstance(s, str):
                s = "".join( x or ' ' for x in s )  # ПРОВЕРИТЬ ВОЗМОЖНОСЬ БОЛЕЕ ГИБКОГО РЕШЕНИЯ!
            alphabet = ascii_lowercase
            shifted_alphabet = alphabet[shift:] + alphabet[:shift]
            table = str.maketrans(alphabet, shifted_alphabet)
            return list( s.translate(table) )

        vocab = 'abcdefghijklmnopqrstuvwxyz '
        char2int, int2char = CustomTextDataset.get_char_ind_map(vocab)
        vocab_size = len(vocab) + 1
        transform_X = ( lambda s: caesar_enc(s)[:-1] )
        transform_y = ( lambda s: s[1:] )

        data = pd.read_csv('data/data.csv')['normalized_text'].str[:15].iloc[:1].tolist()
        train_dataset = CustomTextDataset(data, char2int, transform_X=transform_X, transform_y=transform_y)

        encoder = CustomEncoder(vocab_size, vocab_size).to(TORCH_DEVICE)
        decoder = CustomDecoder(vocab_size, vocab_size).to(TORCH_DEVICE)
        model = Seq2Seq(encoder, decoder).to(TORCH_DEVICE)

        optimizer = Adam(model.parameters(), lr=0.005)

        train_loader_params = DataLoaderParams(batch_size=256, batch_shuffle=False)
        Processing.train_model(train_dataset, self.train_params, train_loader_params, model, optimizer)

        ### PREDICT TEST
        Processing.predict_caesar_test(model, int2char, char2int, 'pdjjlh orrn zkd')  # maggie look wha




def main():
    tests_context = ModelsTests()
    tests_context.test_two()




if __name__ == '__main__':
    main()
