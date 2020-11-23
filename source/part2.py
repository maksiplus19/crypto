from itertools import count
from typing import Iterable, List, Optional


def prime_gen():
    numbers = count(2)
    while True:
        p = next(numbers)
        yield p
        numbers = filter(p.__rmod__, numbers)


# gen = prime()
# i = 1
# while (i := next(gen)) < 1000:
#     print(i)

def gcd_ext(first: int, second: int) -> (int, int, int):
    """Iterative  Greatest Common Divisor for number"""
    x, xt, y, yt = 1, 0, 0, 1
    while second:
        q = first // second
        first, second = second, first % second
        x, xt = xt, x - xt * q
        y, yt = yt, y - yt * q
    return first, x, y


def prime_system(base: int) -> Iterable:
    return filter(lambda x: gcd_binary(x, base) == 1, range(1, base))


def phi(m: int) -> int:
    return len(list(prime_system(m)))


def prime_decomposition(num: int) -> List[int]:
    res = []
    gen = prime_gen()
    while num != 1:
        p = next(gen)
        while num % p == 0:
            res.append(p)
            num //= p
    return res


def inverse_mod(a: int, n: int) -> Optional[int]:
    d, x, y = gcd_ext(a, n)
    return None if d != 1 else x % n


def binary_power_mod(a: int, deg: int, m: int):
    if m == 0:
        raise ValueError("n = 0")

    a %= m
    if a == 0:
        return 0 if deg else None

    if deg < 0:
        a = inverse_mod(a, m)
        if a is None:
            return None
        deg *= -1

    res = 1
    while deg:
        if deg & 1:
            res *= a
            res %= m
        a *= a
        a %= m
        deg >>= 1
    return res


def gcd_binary(first: int, second: int) -> int:
    k = 1
    while first != 0 and second != 0:
        while first % 2 == 0 and second % 2 == 0:
            first >>= 1
            second >>= 1
            k <<= 1
        while first % 2 == 0:
            first >>= 1
        while second % 2 == 0:
            second >>= 1
        if first >= second:
            first -= second
        else:
            second -= first
    return second * k
