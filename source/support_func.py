import random as rnd
from collections import Iterable
from typing import Union, List, cast, Any

import numpy as np

from source.part2 import prime_gen

_prime_list = []


def bits_to_int(*bits: int) -> int:
    res = 0
    for i in bits:
        res |= i
        res <<= 1
    return res >> 1


def to_bit_arr(some: Any) -> List[int]:
    if isinstance(some, bytes):
        some = cast(bytes, some)
        some = int.from_bytes(some, 'big', signed=True)
        b = bin(some)[2:]
        return list(map(int, b))
    elif isinstance(some, str):
        some = cast(str, some)
        return list(map(int, some))
    elif isinstance(some, bytearray):
        some = cast(bytearray, some)
        return to_bit_arr(b''.join(some))
    elif isinstance(some, int):
        return list(map(int, bin(some)[2:]))
    elif isinstance(some, Iterable):
        some = ''.join(map(lambda x: bin(x)[2:], some))
        return list(map(int, some))
    else:
        raise TypeError(f'Not implemented for this type ({type(some)})')


def bit_arr_to_int(arr: Iterable) -> int:
    return int(''.join(map(str, arr)), 2)


def bit_not(num: int, size: int = None) -> int:
    if size is None:
        return num ^ ((1 << num.bit_length()) - 1)
    else:
        return num ^ ((1 << size) - 1)


def rounded_bit_move_left(n: int, step: int) -> int:
    begin = n >> (n.bit_length() - step)
    mask = (1 << step) - 1
    mask <<= n.bit_length() - step
    n &= bit_not(mask)
    n <<= step
    n |= begin
    return n


def rounded_bit_move_right(n: int, step: int) -> int:
    begin = (n & (~(1 << (step + 1)))) << (n.bit_length() - step)
    n >>= step
    n |= begin
    return n


def permutation(num: int, perm: Union[List, np.array]) -> int:
    result = 0
    for k in perm:
        result <<= 1
        result |= (num >> (k - 1)) & 1
    return result


def arr_permutation(arr: Union[List, np.array], perm: List) -> Union[List, np.array]:
    res = []
    for i in perm:
        res.append(arr[i - 1])
    if isinstance(arr, np.ndarray):
        res = np.array(res)
    return res


def r_solid_index(s: Union[str, bytes], char: Union[str, bytes]) -> int:
    i = len(s) - 1
    while i and s[i] == char:
        i -= 1
    return i + 1


def compute_prime_list():
    gen = prime_gen()
    for _ in range(1000):
        next(gen)
    for _ in range(1000):
        _prime_list.append(next(gen))


def get_rnd_prime(seed=None) -> int:
    if not len(_prime_list):
        compute_prime_list()
    if seed is not None:
        rnd.seed(seed)
    return rnd.choice(_prime_list)
