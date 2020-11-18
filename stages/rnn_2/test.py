# coding: utf-8

import pandas as pd  # type: ignore

from typing import Generator
from urllib.request import urlretrieve




def ext_file_save(url: str = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt', save_path: str = './nietzsche.txt') -> None:
    urlretrieve(url, save_path)



def batch_generator(filename: str, batch_size: int, batch_shift: int) -> Generator[str, None, None]:
    current_row = 0
    buffered_chars = ''

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.isspace():
                full_line = ( buffered_chars + line.rstrip('\n') )

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




def func():
    ext_file_save('https://s3.amazonaws.com/text-datasets/nietzsche.txt', './nietzsche.txt')

    df = pd.DataFrame( batch_generator('./nietzsche.txt', 41, 3) )
    df = pd.DataFrame( df[0].str.lower().str.replace(r'[^a-z\s]]', '').apply( lambda s: (s[:-1], s[-1:]) ).tolist() )
    print( df.head() )





def main():
    func()




if __name__ == '__main__':
    main()
