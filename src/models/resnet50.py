# SPDX-License-Identifier: AGPL-3.0-or-later
from src.models.common import *
from torchvision import models
from src import backends
import torch
import os

class ResNet50(Model):
    @classmethod
    def getPretrained(self) -> backends.Net:
        return models.resnet50(pretrained=True)

    @classmethod
    def convertToDiskFormat(self, diskFormat: DiskFormat, torchModel: backends.pytorch.Net, filename: str = "resnet50.onnx"):
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
    
    @classmethod
    def loadImagenetLabelsFromFile(self, labels_path) -> list[str]:
        with open(labels_path) as f:
            imagenet_labels = [line.strip() for line in f.readlines()]
        return imagenet_labels

    @classmethod
    def infer(self, net: backends.Net, imagenet_labels: list[str] = None) -> dict:
        out = net.forwardPass()

        imagenet_class_id = out.argmax()
        confidence = out[0][imagenet_class_id]

        retval = {
            "imagenet_class_id": imagenet_class_id,
            "confidence": confidence
        }
        if imagenet_labels is not None:
            retval["imagenet_class_label"] = imagenet_labels[imagenet_class_id]
        
        return retval




        
