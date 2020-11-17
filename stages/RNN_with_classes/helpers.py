# coding: utf-8

from functools import partial
from typing import Tuple, Sequence, Optional, Dict
from dataclasses import dataclass

from torch import Tensor  # type: ignore
from torch.utils.data import Dataset  # type: ignore

from .types import DATA_TRANSFORM_FUNC, UNI_CHARS_TYPE




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

        s_ = self._data[idx]
        s = [ *s_, *s_appendix ][:self.doc_len]

        X_, y_ = ( self.transform_X(s) if self.transform_X else s ), ( self.transform_y(s) if self.transform_y else s )
        X, y = self._to_tensor(X_), self._to_tensor(y_)
        return X, y
