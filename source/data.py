from functools import reduce

from source.part3 import poly_inv, poly_mul

ENCRYPTED_FILE_EXTENSION = 'crypto'
DECRYPTED_FILE_EXTENSION = 'decrypto'

_waterfall = [[1, 0, 0, 0, 1, 1, 1, 1],
              [1, 1, 0, 0, 0, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 1, 1],
              [1, 1, 1, 1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 1, 1, 1, 1, 1]]


def byte_transform(b: int) -> int:
    b = poly_inv(b)
    b = list(map(int, bin(b)[2:]))
    if len(b) != 8:
        b = [0] * (8 - len(b)) + b
    b.reverse()

    res = (not b[0]) ^ b[4] ^ b[5] ^ b[6] ^ b[7]
    res |= ((not b[0]) ^ b[1] ^ b[5] ^ b[6] ^ b[7]) << 1
    res |= (b[0] ^ b[1] ^ b[2] ^ b[6] ^ b[7]) << 2
    res |= (b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[7]) << 3
    res |= (b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4]) << 4
    res |= ((not b[1]) ^ b[2] ^ b[3] ^ b[4] ^ b[5]) << 5
    res |= ((not b[2]) ^ b[3] ^ b[4] ^ b[5] ^ b[6]) << 6
    res |= (b[3] ^ b[4] ^ b[5] ^ b[6] ^ b[7]) << 7

    return res
    # for k, line in enumerate(_waterfall):
    #     t = list(map(lambda a: a[0] and a[1], zip(b, line)))
    #
    #     res |= reduce(lambda a1, a2: a1 ^ a2, t) << k


Rijndael_Sub_Bytes = {}
Rijndael_Inv_Sub_Bytes = {}

Rijndael_Mix_Table = [[2, 3, 1, 1],
                      [1, 2, 3, 1],
                      [1, 1, 2, 3],
                      [3, 1, 1, 2]]

Rijndael_Inv_Mix_Table = [[0x0e, 0x0b, 0x0d, 0x09],
                          [0x09, 0x0e, 0x0b, 0x0d],
                          [0x0d, 0x09, 0x0e, 0x0b],
                          [0x0b, 0x0d, 0x09, 0x0e]]


Rijndael_Rcon = [[0x1, 0, 0, 0],
                 [0x2, 0, 0, 0]]

for i in range(256):
    b = byte_transform(i)
    Rijndael_Sub_Bytes[i] = b
    Rijndael_Inv_Sub_Bytes[b] = i

for i in range(253):
    Rijndael_Rcon.append([poly_mul(Rijndael_Rcon[-1][0], 2), 0, 0, 0])

# if __name__ == '__main__':
#     a = 143
#     print(Rijndael_Inv_Sub_Bytes[Rijndael_Sub_Bytes[a]])
    # print(hex(byte_transform(0x5a)))
    # print(hex(0b10111110))
    # print(Rijndael_Rcon)
