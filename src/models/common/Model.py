# SPDX-License-Identifier: AGPL-3.0-or-later
from .. import backends

class Model:
    def __init__(self):
        self.backend: backends.Backend = None
        self.net: backends.Net = None