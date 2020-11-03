import os
import random as rnd
from functools import partial, lru_cache
from typing import Optional, Callable, Dict

from source import des_data
from source.data import *
from source.exceptions import CryptoKeyError
from source.signals import UpdateSignal
from source.support_func import *

symbols = [chr(i) for i in range(ord('a'), ord('z') + 1)]
symbols += [chr(i) for i in range(ord('A'), ord('Z') + 1)]
symbols += [str(i) for i in range(10)]
symbols = ''.join(symbols)

Block = Union[bytes]


class Algo:
    @staticmethod
    def batch_size() -> int:
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
        return alg.execute(chunk, key, decryption)

    @staticmethod
    def block_chain(alg: Algo, prev_chunk: Optional[bytes], chunk: Optional[bytes], prev_encrypted: Optional[bytes],
                    key: str, decryption: bool) -> bytes:
        """Сцепление блоков шифротекста"""
        if prev_encrypted is None:
            rnd.seed(key)
            prev_chunk = bytes(''.join(rnd.choices(symbols, k=len(chunk))), encoding='utf8')
        xor = []
        for i, j in zip(chunk, prev_chunk):
            xor.append(int(i ^ j).to_bytes(1, 'big', signed=False))
        chunk = b''.join(xor)
        return alg.execute(chunk, key, decryption)


class DES(Algo):
    """DES"""

    @staticmethod
    def batch_size() -> int:
        return 8

    @staticmethod
    def setup(param: Dict):
        pass

    @staticmethod
    def teardown():
        pass

    # @staticmethod
    # def __enter__():
    #     pass

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
    _batch_size_degree = 13
    _size = 2**_batch_size_degree
    _S = None

    @staticmethod
    def batch_size() -> int:
        return 2**(RC4._batch_size_degree - 3)

    @staticmethod
    def set_batch_size(bit_size: int):
        """n_size is power of 2\nbatch size wil be 2**n_size"""
        if bit_size < 3:
            raise ValueError(f'Too low {bit_size = }')
        RC4._batch_size_degree = bit_size
        RC4._size = 2 ** RC4._batch_size_degree

    @staticmethod
    def gen_key():
        def gen():
            i, j = 0, 0
            while True:
                i = (i + 1) % RC4._size
                j = (j + RC4._S[i]) % RC4._size
                RC4._S[i], RC4._S[j] = RC4._S[j], RC4._S[i]
                yield RC4._S[(RC4._S[i] + RC4._S[j]) % RC4._size]
        return gen()

    _gen = None

    @staticmethod
    def execute(block: Block, key: Union[str, np.array], decrypt: bool = False) -> Block:
        k = next(RC4._gen)
        k = cast(int, k)
        bts = np.array(bytearray(k.to_bytes(RC4.batch_size(), 'big')))
        block = np.array(bytearray(block))
        res = bts ^ block
        return res.tobytes()


    @staticmethod
    def setup(param: Dict):
        key = param['key']
        cast(str, key)
        j = 0
        RC4._S = np.arange(0, RC4._size, 1)
        k_len = len(key)
        RC4._S = list(RC4._S)
        for i in range(RC4._size):
            j = (j + RC4._S[i] + ord(key[i % k_len])) % RC4._size
            RC4._S[i], RC4._S[j] = RC4._S[j], RC4._S[i]
        RC4._S = np.array(RC4._S)
        RC4._gen = RC4.gen_key()


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
        output_file[-1] = ext if decryption else ENCRYPTED_FILE_EXTENSION
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
        output_file[-1] = f'{ext}' if decryption else ENCRYPTED_FILE_EXTENSION
        output_file = '_decrypted.'.join(output_file) if decryption else '.'.join(output_file)
        if not os.path.exists(input_file):
            return None

        file_size = os.path.getsize(input_file)
        progress = 0
        batch_size = alg.batch_size()
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

    #
    # @staticmethod
    # def __des_gen_keys(k: str) -> List:
    #     pass
    #
    # @staticmethod
    # def __des_f_crypt(block, r_key) -> np.array:
    #     pass
    #
    # @staticmethod
    # def DES(input_file: str, key: str, update: UpdateSignal, *, ext: str, decryption: bool = False,
    #         mode: Callable[[np.array, np.array], np.array]):
    #     """Алгоритм DES"""
    #     output_file = input_file.rsplit('.', maxsplit=1)
    #     output_file[-1] = ext if decryption else ENCRYPTED_FILE_EXTENSION
    #     output_file = '.'.join(output_file)
    #     if not os.path.exists(input_file):
    #         return None
    #
    #     keys = gen_keys(key)
    #     # keys = [bytearray(''.join(rnd.choices(symbols, k=len(chunk))), encoding='utf8') for i in range(16)]
    #     prev_res = None
    #     with open(input_file, mode='rb') as in_file:
    #         with open(output_file, mode='wb') as out_file:
    #             for chunk in iter(partial(in_file.read, 8), b''):
    #                 pass
    #
    # @staticmethod
    # def __des_round(chunk: bytes, decryption: bool, key: np.array):
    #     p_chunk = BaseAlgorithm.__des_permutation(int.from_bytes(chunk, 'big', signed=False),
    #                                               des_data.DES_IP_INV if decryption else des_data.DES_IP)
    #     p_chunk = np.array(bytearray(p_chunk.to_bytes(8, 'big', signed=False)))
    #     left, right = None, None
    #     for i in range(16):
    #         if decryption:
    #             left, right = p_chunk[4:] ^ BaseAlgorithm.__des_f_crypt(p_chunk[:4], key), p_chunk[:4]
    #         else:
    #             left, right = p_chunk[4:], p_chunk[:4] ^ BaseAlgorithm.__des_f_crypt(p_chunk[4:], key)
    #         p_chunk = np.concatenate([left, right])
    #     res = BaseAlgorithm.__des_permutation(int.from_bytes(p_chunk, 'big', signed=False),
    #                                           des_data.DES_IP_INV if decryption else des_data.DES_IP)


class AlgorithmConnect:
    reverse_connect = {
        # BaseAlgorithm.vernam_cipher.__doc__: BaseAlgorithm.vernam_cipher
        DES.__doc__: DES,
        RC4.__doc__: RC4
    }

    mode_connect = {
        CryptoMode.electronic_codebook.__doc__: CryptoMode.electronic_codebook,
        CryptoMode.block_chain.__doc__: CryptoMode.block_chain
    }
