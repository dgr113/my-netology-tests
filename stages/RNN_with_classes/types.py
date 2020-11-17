# coding: utf-8

from typing import Callable, Iterable, Sequence, Union, List


UNI_CHARS_TYPE = Union[str, List[str]]
_2D_INT_ARRAY = Sequence[Sequence[int]]
DATA_TRANSFORM_FUNC = Callable[[Iterable[str]], Sequence[str]]
