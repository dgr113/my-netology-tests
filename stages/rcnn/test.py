# coding: utf-8

import numpy as np
import pandas as pd
from torch import Tensor




def main():
    arr = np.array([
        [1,2,3,4,5],
        [10,20,30,40,50]
    ])


    t = Tensor(arr)
    print(t)
    print(t.view(-1))





if __name__ == '__main__':
    main()
