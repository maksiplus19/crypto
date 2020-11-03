from typing import List
import re

import numpy as np

from source.exceptions import *

param_pattern = re.compile(r'[^\d]')


def split_params(params: str) -> List[int]:
    try:
        return list(map(int, filter(None, param_pattern.split(params))))
    except ValueError:
        raise ParamParseException(params)


def bin_dots(num: int) -> str:
    num = bin(num)[2:]
    if len(num) % 4 != 0:
        num = f'{"0" * (4 - (len(num) % 4))}{num}'
    return '.'.join(num[4 * i:4 * (i + 1)] for i in range(len(num) // 4))


def bit_not(num: int, size: int = None) -> int:
    if size is None:
        return num ^ ((1 << num.bit_length()) - 1)
    return num ^ ((1 << size) - 1)


def task1(a: int, params: str) -> List:
    """
    С клавиатуры вводится 32-х разрядное целое число a в двоичной системе
    счисления.
        1. Вывести k −ый бит числа a. Номер бита предварительно запросить у пользователя.
        2. Установить/снять k −ый бит числа a.
        3. Поменять местами i −ый и j −ый биты в числе a. Числа i и j предварительно запросить у пользователя.
        4. Обнулить младшие m бит.
    """
    params = split_params(params)
    if len(params) < 4:
        raise ParamParseException(params)
    k, i, j, m = params[0], params[1], params[2], params[3]

    result = [f'k-ый бит {(a >> k) & 1}', f'Установлен k-ый бит {bin_dots(a & ~(1 << k))}',
              f'Снят k-ый бит {bin_dots(a | (1 << k))}']

    b1, b2 = (a >> i) & 1, (a >> j) & 1
    x = b1 ^ b2
    x = (x << i) | (x << j)
    result.append(f'Обмен битов {i=} {j=} {bin_dots(a ^ x)}')
    result.append(f'Обнуление младших {m=} битов {bin_dots(a >> m << m)}')
    return result


def task2(a: int, params: str) -> List:
    """
    A) «Склеить» первые i битов с последними i битами из целого числа длиной len битов.
    Пример.
    Пусть есть 12-ти разрядное целое число, представленное в двоичной системе счисления 100011101101.
    «Склеим» первые 3 и последние 3 бита, получим 100101.
    B) Получить биты из целого числа длиной len битов, находящиеся между первыми i битами и последними i битами.
    Пример.
    Пусть есть 12-ти разрядное целое число, представленное в двоичной системе счисления 100011101101.
    Получим биты находящиеся между первыми 3 и последними 3 битами: 011101.
    """
    params = split_params(params)
    if len(params) < 1:
        raise NotEnoughParamsException(len(params), 1)
    i = params[0]
    result = [f'Склеиные биты {bin_dots((a >> (a.bit_length() - i) << i) | (a & ((1 << i) - 1)))}']

    mask = ((1 << (a.bit_length() - 2 * i)) - 1) << i
    result.append(bin_dots((a & mask) >> i))
    return result


def task3(a: int, params: str) -> List:
    """Поменять местами байты в заданном 32-х разрядном целом числе. Перестановка задается пользователем."""
    # 11111111000000001010101010011001
    params = split_params(params)
    if len(params) != 2:
        raise ParamParseException(params)
    i, j = params[0], params[1]
    if i > 3 or j > 3 or i < 0 or j < 0:
        raise ParamValueException(params)

    result = []
    a = bytearray(a.to_bytes(4, byteorder='little'))
    t = a[i]
    a[i] = a[j]
    a[j] = t
    res = int.from_bytes(a, byteorder='little', signed=False)
    result.append(f'Result {bin_dots(res)}')
    return result


def task4(a: int, params: str) -> List:
    if a & 1:
        return ['Нечетное число']
    s = bin(a)[2:]
    return [f'Максимальная степень 2 = {len(s) - s.rindex("1") - 1}']


def task5(a: int, params: str) -> List:
    p = a.bit_length() - 1
    return [f'{p=}', f'{2 ** p} <= {a} <= {2 ** (p + 1)}']


def task6(x: int, params: str) -> List:
    """ «Поксорить» все биты этого числа друг с другом"""
    while (bl := x.bit_length()) != 1 and x != 0:
        bl //= 2
        t = x & ((1 << bl) - 1)
        x = t ^ (x >> bl)
    return [f'Result {x}']


def task7(a: int, params: str) -> List:
    """Написать макросы циклического сдвига в 2**p разрядном целом числе на n бит влево и вправо"""

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

    params = split_params(params)
    right = None
    if len(params) == 1:
        left = params[0]
    elif len(params) == 2:
        left, right = params[0], params[1]
    else:
        raise ParamParseException(params)
    result = [f'Сдвиг влево {bin_dots(rounded_bit_move_left(a, left))}']
    if right is not None:
        result.append(f'Сдвиг вправо {bin_dots(rounded_bit_move_right(a, right))}')
    return result


def task8(a: int, params: str) -> List:
    """Дано *n* битовое данное. Задана перестановка бит (1, 8, 23, 0, 16, … ). Написать функцию, выполняющую эту
    перестановку """

    def get_bit(num: int, bit: int):
        return (num >> bit) & 1

    params = split_params(params)
    result = 0
    params = np.array(params)
    is_incorrect = params > a.bit_length() - 1
    if np.any(is_incorrect):
        raise ParamValueException(params)

    for i in params:
        result |= get_bit(a, i)
        result <<= 1
    result >>= 1
    return [f'Пререстановка {bin_dots(result)}', str(list(params))]
