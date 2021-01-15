from source.support_func import to_bit_arr


def poly_to_str(n: int) -> str:
    """Напишите функцию, представляющую элемент из GF(256) в полиномиальной форме."""
    arr = to_bit_arr(n)
    res = []
    for i, c in enumerate(reversed(arr)):
        if c:
            if i > 1:
                res.append(f'x^{i}')
            elif i == 1:
                res.append('x')
            else:
                res.append('1')
    return '+'.join(reversed(res))


def poly_mul(a: int, b: int) -> int:
    """Напишите функцию, умножения двух двоичных многочленов; умножения,двух элементов из GF(256)."""
    if max(a.bit_length(), b.bit_length()) > 8:
        raise ValueError(f'{a.bit_length()=} {b.bit_length()=}')
    p = 0
    while a and b:
        if b & 1:
            p ^= a

        if a >= 128:
            a = ((a << 1) ^ 0x11b) & 0b11111111
        else:
            a <<= 1
        b >>= 1
    return p & 0b11111111


def poly_inv(a: int) -> int:
    """Напишите функцию, для поиска мультипликативного обратного для элемента из GF(256)."""
    res = 1
    deg = 254
    a = poly_mul(a, a)
    deg >>= 1
    for i in range(7):
        res = poly_mul(res, a)
        a = poly_mul(a, a)
        deg >>= 1
    return res


if __name__ == '__main__':
    poly = 6
    print(f'{poly_inv(poly) = }')
    print(f'{poly_mul(poly, poly_inv(poly)) = }')
