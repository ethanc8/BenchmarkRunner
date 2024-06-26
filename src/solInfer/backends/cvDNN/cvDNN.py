# SPDX-License-Identifier: AGPL-3.0-or-later
from solInfer import backends
import cv2

class Backend_class(backends.Backend):
    def __init__(self):
        pass

class Net(backends.Net):
    def __init__(self, net: cv2.dnn.Net):
        self.net = net

    @classmethod
    def loadONNX(self, filename: str) -> cv2.dnn.Net:
        return Net(cv2.dnn.readNetFromONNX(filename))

    def forwardPass(self, inputTensor: backends.Tensor) -> backends.Tensor:
        self.net.setInput(inputTensor.get_ndarray())
        return backends.NPTensor(self.net.forward())

Backend = Backend_class()