class ParamException(Exception):
    def __init__(self):
        super(ParamException, self).__init__()


class ParamParseException(ParamException):
    def __init__(self, params):
        super(ParamParseException, self).__init__()
        self.params = params

    def __str__(self):
        return f'Cannot parse params: {self.params}'


class ParamValueException(ParamException):
    def __init__(self, value):
        super(ParamValueException, self).__init__()
        self.value = value

    def __str__(self):
        return f'Wrong value of param {self.value}'


class NotEnoughParamsException(ParamException):
    def __init__(self, get: int, need: int):
        super(NotEnoughParamsException, self).__init__()
        self.get = get
        self.need = need

    def __str__(self):
        return f'Not enough params got {self.get} instead of {self.need}'


class CryptoException(Exception):
    def __init__(self):
        super(CryptoException, self).__init__()


class CryptoKeyError(CryptoException):
    def __init__(self, key_len: int, expected_len: int):
        super(CryptoKeyError, self).__init__()
        self.k_len = key_len
        self.e_len = expected_len

    def __str__(self):
        return f'Key has wrong len. Expected len {self.e_len} got {self.k_len}'
