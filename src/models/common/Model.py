from .. import backends

class Model:
    def __init__(self):
        self.backend: backends.Backend = None
        self.net: backends.Net = None