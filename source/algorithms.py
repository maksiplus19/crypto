import os
from functools import partial, lru_cache
from typing import Optional, Callable, Dict

from source import des_data
from source.data import *
from source.exceptions import CryptoKeyError
from source.part2 import inverse_mod, binary_power_mod
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
        return alg.execute(chunk, key, decryption)

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
        print(f'{m=}')
        res = binary_power_mod(m, d if decrypt else e, n)
        print(f'r={res}')
        print(f'{p=} {q=} {e=} {d=}')
        return res.to_bytes(RSA.batch_size(not decrypt), 'big', signed=False)

    @staticmethod
    def setup(param: Dict):
        pass

    @staticmethod
    def teardown():
        pass


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
        RC4.__doc__: RC4
    }

    mode_connect = {
        CryptoMode.electronic_codebook.__doc__: CryptoMode.electronic_codebook,
        CryptoMode.block_chain.__doc__: CryptoMode.block_chain,
        CryptoMode.feedback.__doc__: CryptoMode.feedback
    }
