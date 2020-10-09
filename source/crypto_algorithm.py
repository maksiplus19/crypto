import os
import random as rnd
from functools import partial, lru_cache
from typing import Optional, Callable

import numpy as np

from source.data import *

symbols = [chr(i) for i in range(ord('a'), ord('z') + 1)]
symbols += [chr(i) for i in range(ord('A'), ord('Z') + 1)]
symbols = ''.join(symbols)


class BaseAlgorithm:
    @staticmethod
    @lru_cache(maxsize=4)
    def extend_key(key: str, batch_size: int):
        return key * int(np.ceil(batch_size / len(key)))

    @staticmethod
    def vernam_cipher(input_file: str, key: str, update: Callable[[int], None], batch_size: int = 1024,
                      *, generator: bool = False, decryption: bool = False) -> Optional[str]:
        """Алгоритм Вернама"""
        output_file = input_file.rsplit('.', maxsplit=1)
        output_file[-1] = DECRYPTED_FILE_EXTENSION if decryption else ENCRYPTED_FILE_EXTENSION
        output_file = '.'.join(output_file)
        if not os.path.exists(input_file):
            return None

        if generator:
            rnd.seed(key)
        file_size = os.path.getsize(input_file)
        progress = 0
        update(0)
        with open(input_file, mode='rb') as in_file:
            with open(output_file, mode='wb') as out_file:
                for chunk in iter(partial(in_file.read, batch_size), b''):
                    if generator:
                        key_arr = np.array(bytearray(''.join(rnd.choices(symbols, k=len(chunk))), encoding='utf8'))
                    else:
                        key_arr = Algorithm.extend_key(key, batch_size)[:len(chunk)]
                    chunk = np.array(bytearray(chunk))
                    key_arr = np.array(bytearray(key_arr, encoding='utf8'))
                    out_file.write(chunk ^ key_arr)
                    progress += len(key_arr)
                    update(int(np.ceil(100 * progress / file_size)))
                    # print(int(np.ceil(progress / file_size)))

        return output_file


class Algorithm(BaseAlgorithm):
    reverse_connect = {
        BaseAlgorithm.vernam_cipher.__doc__: BaseAlgorithm.vernam_cipher
    }
