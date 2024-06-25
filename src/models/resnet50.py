from common import *
from torchvision import models
from .. import backends
import torch
import os

class ResNet50(Model):
    @staticmethod
    def getPretrainedTorchModelNet() -> backends.pytorch.Net:
        models.resnet50(pretrained=True)

    @staticmethod
    def convertTo(self, format: Format, filename: str = "resnet50.onnx"):
        if format == formats.ONNX:
            # generate model input
            generated_input = torch.autograd.Variable(
                torch.randn(1, 3, 224, 224)
            )

            # model export into ONNX format
            torch.onnx.export(
                self.torchModel,
                generated_input,
                filename,
                verbose=True,
                input_names=["input"],
                output_names=["output"],
                opset_version=11
            )

            return filename
        else:
            return None
    
    def loadImagenetLabelsFromFile(self, labels_path) -> list[str]:
        with open(labels_path) as f:
            self.imagenet_labels = [line.strip() for line in f.readlines()]

    def infer(self, net: backends.Net) -> dict:
        out = net.forwardPass()



        
