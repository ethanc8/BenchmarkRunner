from ..common import *
import cv2

class cvDNN_class(Backend):
    def __init__(self):
        pass

class cvDNN_Net:
    def __init__(self, net: cv2.dnn.Net):
        self.net = net

    @staticmethod
    def loadONNX(self, filename: str) -> cv2.dnn.Net:
        return cvDNN_Net(cv2.dnn.readNetFromONNX(filename))

    def forwardPass(self, inputTensor: Tensor) -> Tensor:
        self.net.setInput(inputTensor.to_ndarray())
        return NPTensor(self.net.forward())

cvDNN = cvDNN_class()