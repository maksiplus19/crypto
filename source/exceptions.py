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
