from common import *
from torchvision import models
import torch
import os

class ResNet50(Model):
    def __init__(self):
        self.torchModel = models.resnet50(pretrained=True)

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
        
    def getTorchModel(self):
        return self.torchModel
        
