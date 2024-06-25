from common import *
from torchvision import models
from .. import backends
import torch
import os

class ResNet50(Model):
    @staticmethod
    def getPretrained() -> backends.Net:
        return models.resnet50(pretrained=True)

    @staticmethod
    def convertToDiskFormat(diskFormat: DiskFormat, torchModel: backends.pytorch.Net, filename: str = "resnet50.onnx"):
        if diskFormat == diskFormats.ONNX:
            # generate model input
            generated_input = torch.autograd.Variable(
                torch.randn(1, 3, 224, 224)
            )

            # model export into ONNX format
            torch.onnx.export(
                torchModel,
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
    
    @staticmethod
    def loadImagenetLabelsFromFile(labels_path) -> list[str]:
        with open(labels_path) as f:
            imagenet_labels = [line.strip() for line in f.readlines()]
        return imagenet_labels

    @staticmethod
    def infer(net: backends.Net) -> dict:
        out = net.forwardPass()




        
