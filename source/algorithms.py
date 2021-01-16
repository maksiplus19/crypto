import os
from copy import deepcopy
from functools import partial, lru_cache, reduce
from typing import Optional, Callable, Dict

from source import des_data, data
from source.exceptions import CryptoKeyError
from source.part2 import inverse_mod, binary_power_mod
from source.part3 import poly_mul
from source.signals import UpdateSignal
from source.support_func import *

symbols = [chr(i) for i in range(ord('a'), ord('z') + 1)]
symbols += [chr(i) for i in range(ord('A'), ord('Z') + 1)]
symbols += [str(i) for i in range(10)]
symbols = ''.join(symbols)

Block = Union[bytes]


class Algo:
    @staticmethod
    def batch_size(decrypt: bool = False) -> int:
        raise NotImplementedError

    @staticmethod
    def execute(block: Block, key: Union[str, np.array], decrypt: bool = False) -> Block:
        raise NotImplementedError

    @staticmethod
    def setup(param: Dict):
        pass

    @staticmethod
    def teardown():
        pass


Mode = Callable[[Algo, bytes, Optional[bytes], Optional[bytes], str, bool], bytes]


class CryptoMode:
    @staticmethod
    def electronic_codebook(alg: Algo, prev_chunk: Optional[bytes], chunk: Optional[bytes],
                            prev_encrypted: Optional[bytes], key: str, decryption: bool) -> bytes:
        """Электронная кодовая книга"""
        trash = alg.batch_size() - len(chunk)
        if trash:
            chunk += b'\x00' * trash
        res = alg.execute(chunk, key, decryption)
        if decryption:
            trash = r_solid_index(res, b'\x00')
            if trash:
                res = res[:trash]
        return res

    @staticmethod
    def block_chain(alg: Algo, prev_chunk: Optional[bytes], chunk: Optional[bytes], prev_encrypted: Optional[bytes],
                    key: str, decryption: bool) -> bytes:
        """Сцепление блоков шифротекста"""
        second_chunk = prev_chunk if decryption else prev_encrypted
        if second_chunk is None:
            rnd.seed(key)
            second_chunk = bytes(''.join(rnd.choices(symbols, k=alg.batch_size())), encoding='utf8')
        trash = alg.batch_size() - len(chunk)
        if trash:
            chunk += b'\x00' * trash
        if decryption:
            chunk = (np.array(bytearray(second_chunk), dtype=np.int8) ^ np.array(bytearray(chunk),
                                                                                 dtype=np.int8)).tobytes()
            res = alg.execute(chunk, key, decryption)
            trash = r_solid_index(res, b'\x00')
            if trash:
                res = res[:trash]
        else:
            chunk = alg.execute(chunk, key, decryption)
            res = (np.array(bytearray(second_chunk), dtype=np.uint8) ^ np.array(bytearray(chunk),
                                                                                dtype=np.uint8)).tobytes()
        if decryption:
            trash = r_solid_index(res, b'\x00')
            if trash:
                res = res[:trash]
        return res

    @staticmethod
    def feedback(alg: Algo, prev_chunk: Optional[bytes], chunk: Optional[bytes], prev_encrypted: Optional[bytes],
                 key: str, decryption: bool) -> bytes:
        """Обратная связь по шифротексту"""
        second_chunk = prev_chunk if decryption else prev_encrypted
        if second_chunk is None:
            rnd.seed(key)
            second_chunk = bytes(''.join(rnd.choices(symbols, k=alg.batch_size())), encoding='utf8')
        trash = alg.batch_size() - len(chunk)
        if trash:
            chunk += b'\x00' * trash
        second_chunk = alg.execute(second_chunk, key)
        res = (np.array(bytearray(second_chunk), dtype=np.uint8) ^ np.array(bytearray(chunk),
                                                                            dtype=np.uint8)).tobytes()
        if decryption:
            trash = r_solid_index(res, b'\x00')
            if trash:
                res = res[:trash]
        return res


class DES(Algo):
    """DES"""

    @staticmethod
    def batch_size(decrypt: bool = False) -> int:
        return 8

    @staticmethod
    def setup(param: Dict):
        pass

    @staticmethod
    def teardown():
        pass

    @staticmethod
    def gen_key(key: str) -> List[np.array]:
        rnd.seed(key)
        arr = []
        key = int.from_bytes(bytes(''.join(rnd.choices(symbols, k=8)), encoding='utf8'), 'big')
        key = permutation(key, des_data.G)
        left, right = key >> 28, key & ((1 << 28) - 1)
        for i in range(16):
            left = rounded_bit_move_left(left, des_data.KEY_MOVE[i])
            right = rounded_bit_move_left(right, des_data.KEY_MOVE[i])
            ll = list(map(int, bin(left)[2:]))
            rr = list(map(int, bin(right)[2:]))
            ll = [0] * (28 - len(ll)) + ll
            rr = [0] * (28 - len(rr)) + rr
            arr.append(np.array(ll + rr))
        arr = list(map(partial(arr_permutation, perm=des_data.H), arr))
        return arr

    @staticmethod
    def execute(chunk: Block, key: Union[str, np.array], decryption: bool = False) -> Block:
        p_chunk = permutation(int.from_bytes(chunk, 'big', signed=False),
                              des_data.DES_IP_INV if decryption else des_data.DES_IP)
        p_chunk = np.array(bytearray(p_chunk.to_bytes(8, 'big', signed=False)))

        key = DES.gen_key(key)[::-1] if decryption else DES.gen_key(key)
        for i in range(16):
            if decryption:
                left, right = p_chunk[4:] ^ DES._crypt(p_chunk[:4], key[i]), p_chunk[:4]
            else:
                left, right = p_chunk[4:], p_chunk[:4] ^ DES._crypt(p_chunk[4:], key[i])
            p_chunk = np.concatenate([left, right])
        res = permutation(int.from_bytes(p_chunk, 'big', signed=False),
                          des_data.DES_IP_INV if decryption else des_data.DES_IP)
        return res.to_bytes(8, 'big', signed=False)

    @staticmethod
    def _crypt(chunk: np.array, key: np.array) -> np.array:
        if len(key) != 48:
            raise CryptoKeyError(len(key), 48)
        chunk = int.from_bytes(b''.join([int(i).to_bytes(1, 'big') for i in chunk]), 'big')
        chunk = permutation(chunk, des_data.DES_EXT)
        chunk = to_bit_arr(chunk)
        chunk = np.array([0] * (48 - len(chunk)) + chunk)
        chunk ^= key
        six_bits = [chunk[i:i + 5] for i in range(0, 48, 6)]
        new_chunk = []
        for i, six in enumerate(six_bits):
            row = bits_to_int(six[0], six[-1])
            column = bits_to_int(*six[1:-1])
            new_chunk.append(des_data.SUB_BOX[i][row][column])
        completed = []
        for i in map(lambda x: bin(x)[2:], new_chunk):
            completed.append('0' * (4 - len(i)) + i)
        num = permutation(int(''.join(completed), 2), des_data.P)
        res = np.array(bytearray(num.to_bytes(4, 'big', signed=False)))
        return res


class RC4(Algo):
    """RC4"""
    _n = 16
    _S = None

    @staticmethod
    def batch_size(decrypt: bool = False) -> int:
        return 1024

    @staticmethod
    def gen_key():
        def gen():
            i, j = 0, 0
            while True:
                i = (i + 1) % 2 ** RC4._n
                j = (j + RC4._S[i]) % 2 ** RC4._n
                RC4._S[i], RC4._S[j] = RC4._S[j], RC4._S[i]
                yield int(RC4._S[(RC4._S[i] + RC4._S[j]) % 2 ** RC4._n])

        return gen()

    _gen = None

    @staticmethod
    def execute(block: Block, key: Union[str, np.array], decrypt: bool = False) -> Block:
        k = []
        for i in range(len(block) // 2 + 1):
            word = int(next(RC4._gen)).to_bytes(2, 'big')
            k.extend([word[:1][0], word[1:][0]])
        k = k[:len(block)]
        res = np.array(k, dtype=np.int8) ^ np.array(bytearray(block), dtype=np.int8)
        return res.tobytes()

    @staticmethod
    def setup(param: Dict):
        key = param['key']
        cast(str, key)
        j = 0
        RC4._S = np.arange(0, 2 ** RC4._n, 1)
        k_len = len(key)
        RC4._S = list(RC4._S)
        for i in range(2 ** RC4._n):
            j = (j + RC4._S[i] + ord(key[i % k_len])) % 2 ** RC4._n
            RC4._S[i], RC4._S[j] = RC4._S[j], RC4._S[i]
        RC4._S = np.array(RC4._S)
        RC4._gen = RC4.gen_key()


class RSA(Algo):
    """RSA"""

    @staticmethod
    def batch_size(decrypt: bool = False) -> int:
        if decrypt:
            return 4
        return 2

    @staticmethod
    def execute(block: Block, key: Union[str, np.array], decrypt: bool = False) -> Block:
        p, q = get_rnd_prime(key), get_rnd_prime()
        n = p * q
        phi_num = (p - 1) * (q - 1)
        e = 65537
        while (d := inverse_mod(e, phi_num)) is None:
            e -= 1
        m = int.from_bytes(block, 'big', signed=False)
        res = binary_power_mod(m, d if decrypt else e, n)
        return res.to_bytes(RSA.batch_size(not decrypt), 'big', signed=False)

    @staticmethod
    def setup(param: Dict):
        pass

    @staticmethod
    def teardown():
        pass


class Rijndael(Algo):
    """Rijndael"""
    BLOCK_SIZE = 128
    KEY_SIZE = 128
    Nb = None
    Nk = None
    SHIFT_ROW_NUMBERS = {4: [1, 2, 3], 6: [1, 2, 3], 8: [1, 3, 4]}

    # INV_SHIFT_ROW_NUMBERS = {4: [3, 2, 1], 6: [5, 4, 3], 8: [7, 5, 4]}

    @staticmethod
    def batch_size(decrypt: bool = False) -> int:
        return Rijndael.BLOCK_SIZE // 8

    @staticmethod
    def execute(block: Block, key: Union[str, np.array], decrypt: bool = False) -> Block:
        Rijndael.Nb = Rijndael.BLOCK_SIZE // 32
        Rijndael.Nk = Rijndael.KEY_SIZE // 32
        block = np.array(bytearray(block), dtype=np.ubyte)
        block = np.reshape(block, (4, Rijndael.Nb), order='F')
        block = block.tolist()
        block = cast(List[List[int]], block)
        # key = np.a
        ext_key = Rijndael.key_expansion(key)
        ext_key = np.array(ext_key, dtype=np.ubyte)
        # ext_key = np.reshape(ext_key, ext_key.shape[::-1])

        nr = Rijndael.get_round_count()
        if not decrypt:
            block = np.bitwise_xor(block, np.reshape(ext_key[0:Rijndael.Nb], (4, Rijndael.Nb))).tolist()
            for i in range(1, nr):
                Rijndael.sub_bytes(block)
                Rijndael.shift_rows(block)
                Rijndael.mix_columns(block)
                block = np.bitwise_xor(block, np.reshape(ext_key[Rijndael.Nb * i:Rijndael.Nb * (i + 1)], (4, Rijndael.Nb))).tolist()

            Rijndael.sub_bytes(block)
            Rijndael.shift_rows(block)
            block = np.bitwise_xor(block, np.reshape(ext_key[Rijndael.Nb * nr:Rijndael.Nb * (nr + 1)], (4, Rijndael.Nb))).tolist()
        else:
            block = np.bitwise_xor(block, np.reshape(ext_key[Rijndael.Nb * nr:Rijndael.Nb * (nr + 1)], (4, Rijndael.Nb))).tolist()
            for i in range(nr - 1, 0, -1):
                Rijndael.inv_sub_bytes(block)
                Rijndael.inv_shift_rows(block)
                block = np.bitwise_xor(block, np.reshape(ext_key[Rijndael.Nb * i:Rijndael.Nb * (i + 1)], (4, Rijndael.Nb))).tolist()
                Rijndael.inv_mix_columns(block)

            Rijndael.inv_shift_rows(block)
            Rijndael.inv_sub_bytes(block)
            block = np.bitwise_xor(block, np.reshape(ext_key[0:Rijndael.Nb], (4, Rijndael.Nb))).tolist()
        return np.reshape(np.array(block, dtype=np.ubyte), (Rijndael.Nb * 4), order='F').tobytes()

    @staticmethod
    def get_round_count() -> int:
        if Rijndael.Nb == 8 or Rijndael.Nk == 8:
            return 14
        elif Rijndael.Nb == 6 or Rijndael.Nk == 6:
            return 12
        return 10

    @staticmethod
    def key_expansion(key) -> List[List[int]]:
        key = list(bytearray(key, encoding='utf8'))
        if len(key) != Rijndael.KEY_SIZE // 8:
            raise CryptoKeyError(len(key), Rijndael.KEY_SIZE // 8)

        w = []
        # if Rijndael.Nk <= 6:
        for i in range(Rijndael.Nk):
            w.append([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]])
        for i in range(Rijndael.Nk, Rijndael.Nb * (Rijndael.get_round_count() + 1)):
            temp = w[i - 1]
            if i % Rijndael.Nk == 0:
                temp = Rijndael.sub_byte(Rijndael.rot_byte(temp))
                for k, c in enumerate(data.Rijndael_Rcon[i // Rijndael.Nk]):
                    temp[k] ^= c
            elif Rijndael.Nk > 6 and i % Rijndael.Nk == 4:
                temp = Rijndael.sub_byte(temp)
            w.append(list(map(lambda x: x[0] ^ x[1], zip(temp, w[i - Rijndael.Nk]))))
        return w

    @staticmethod
    def sub_bytes(arr: List[List[int]]):
        for i in range(4):
            for j in range(Rijndael.Nb):
                arr[i][j] = data.Rijndael_Sub_Bytes[arr[i][j]]

    @staticmethod
    def shift_rows(arr: List[List[int]]):
        for i, n in enumerate(Rijndael.SHIFT_ROW_NUMBERS[Rijndael.Nb]):
            k = i + 1
            left, right = arr[k][:n], arr[k][n:]
            arr[k] = right + left

    @staticmethod
    def mix_columns(arr: List[List[int]]):
        for i in range(Rijndael.Nb):
            t = [arr[0][i], arr[1][i], arr[2][i], arr[3][i]]
            for j in range(4):
                arr[j][i] = reduce(lambda a, b: a ^ b, map(lambda x: poly_mul(x[0], x[1]),
                                                           zip(t, data.Rijndael_Mix_Table[j])))

    @staticmethod
    def rot_byte(arr: List[int]) -> List[int]:
        return arr[1:] + arr[:1]

    @staticmethod
    def sub_byte(arr: List[int]) -> List[int]:
        res = []
        for i in arr:
            res.append(data.Rijndael_Sub_Bytes[i])
        return res

    @staticmethod
    def inv_sub_bytes(arr: List[List[int]]):
        for i in range(4):
            for j in range(Rijndael.Nb):
                arr[i][j] = data.Rijndael_Inv_Sub_Bytes[arr[i][j]]

    @staticmethod
    def inv_shift_rows(arr: List[List[int]]):
        for i, n in enumerate(Rijndael.SHIFT_ROW_NUMBERS[Rijndael.Nb]):
            k = i + 1
            left, right = arr[k][:Rijndael.Nb - n], arr[k][Rijndael.Nb - n:]
            arr[k] = right + left

    @staticmethod
    def inv_mix_columns(arr: List[List[int]]):
        for i in range(Rijndael.Nb):
            t = [arr[0][i], arr[1][i], arr[2][i], arr[3][i]]
            for j in range(4):
                arr[j][i] = reduce(lambda a, b: a ^ b, map(lambda x: poly_mul(x[0], x[1]),
                                                           zip(t, data.Rijndael_Inv_Mix_Table[j])))


class BaseAlgorithm:
    @staticmethod
    def gen_key():
        pass

    @staticmethod
    @lru_cache(maxsize=4)
    def extend_key(key: str, batch_size: int):
        return key * int(np.ceil(batch_size / len(key)))

    @staticmethod
    def vernam_cipher(input_file: str, key: str, update: UpdateSignal, *, batch_size: int = None,
                      generator: bool = False, decryption: bool = False, ext: str) -> Optional[str]:
        """Алгоритм Вернама"""
        output_file = input_file.rsplit('.', maxsplit=1)
        output_file[-1] = ext if decryption else data.ENCRYPTED_FILE_EXTENSION
        output_file = '.'.join(output_file)
        if not os.path.exists(input_file):
            return None

        if generator:
            rnd.seed(key)
        file_size = os.path.getsize(input_file)
        if batch_size is None:
            batch_size = file_size // 100  # 0 if file_size > 2**20 else 2**20
        progress = 0
        update.update.emit(0)
        with open(input_file, mode='rb') as in_file:
            with open(output_file, mode='wb') as out_file:
                for chunk in iter(partial(in_file.read, batch_size), b''):
                    if generator:
                        key_arr = np.array(bytearray(''.join(rnd.choices(symbols, k=len(chunk))), encoding='utf8'))
                    else:
                        key_arr = BaseAlgorithm.extend_key(key, batch_size)[:len(chunk)]
                    chunk = np.array(bytearray(chunk))
                    key_arr = np.array(bytearray(key_arr, encoding='utf8'))
                    out_file.write(chunk ^ key_arr)
                    progress += len(key_arr)
                    update.update.emit(int(np.ceil(100 * progress / file_size)))
        return output_file

    @staticmethod
    def encrypt_decrypt(input_file: str, mode: Mode, alg: Algo, signal: UpdateSignal,
                        key: str, decryption: bool = False, ext: str = None) -> Optional[str]:
        if decryption and ext is None:
            raise ValueError('Decryption need original file extension')

        output_file = input_file.rsplit('.', maxsplit=1)
        output_file[-1] = f'{ext}' if decryption else data.ENCRYPTED_FILE_EXTENSION
        output_file = '_decrypted.'.join(output_file) if decryption else '.'.join(output_file)
        if not os.path.exists(input_file):
            return None

        file_size = os.path.getsize(input_file)
        progress = 0
        batch_size = alg.batch_size(decryption)
        signal.update.emit(0)
        alg.setup({'key': key})
        with open(input_file, mode='rb') as in_file:
            with open(output_file, mode='wb') as out_file:
                prev_chunk = None
                prev_encrypted = None
                for chunk in iter(partial(in_file.read, batch_size), b''):
                    encrypted = mode(alg, prev_chunk, chunk, prev_encrypted, key, decryption)
                    prev_chunk = chunk
                    prev_encrypted = encrypted
                    out_file.write(encrypted)
                    progress += batch_size
                    signal.update.emit(int(progress / file_size * 100))
        alg.teardown()
        return output_file


class AlgorithmConnect:
    reverse_connect = {
        # BaseAlgorithm.vernam_cipher.__doc__: BaseAlgorithm.vernam_cipher
        RSA.__doc__: RSA,
        DES.__doc__: DES,
        RC4.__doc__: RC4,
        Rijndael.__doc__: Rijndael
    }

    mode_connect = {
        CryptoMode.electronic_codebook.__doc__: CryptoMode.electronic_codebook,
        CryptoMode.block_chain.__doc__: CryptoMode.block_chain,
        CryptoMode.feedback.__doc__: CryptoMode.feedback
    }


if __name__ == '__main__':
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    Rijndael.Nb = 4
    Rijndael.Nk = 4
    print(a)
    b = a.copy()
    Rijndael.mix_columns(a)
    print(a)
    Rijndael.inv_mix_columns(a)
    print(a)
    print(a == b)
