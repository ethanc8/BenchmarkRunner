# SPDX-License-Identifier: AGPL-3.0-or-later
from solInfer.models.common import *
from torchvision import models
from solInfer import backends
import torch
import os

class ResNet50(Model):
    @classmethod
    def get_pretrained(self) -> backends.Net:
        return backends.pytorch.Net(models.resnet50(pretrained=True))

    @classmethod
    def convert_to_disk_format(self, disk_format: DiskFormat, pytorch_net: backends.pytorch.Net, filename: str = "resnet50.onnx", image_size=(256, 256)):
        if disk_format == disk_formats.ONNX:
            # generate model input
            generated_input = torch.autograd.Variable(
                torch.randn(1, 3, image_size[0], image_size[1])
            )

            # model export into ONNX format
            torch.onnx.export(
                pytorch_net.net,
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
    def load_imagenet_labels_from_file(self, labels_path) -> list[str]:
        with open(labels_path) as f:
            imagenet_labels = [line.strip() for line in f.readlines()]
        return imagenet_labels

    @classmethod
    def infer(self, net: backends.Net, input_img: backends.Tensor, imagenet_labels: list[str] = None) -> dict:
        out = net.forwardPass(input_img)

        imagenet_class_id = out.argmax()
        confidence = out[0][imagenet_class_id]

        retval = {
            "imagenet_class_id": imagenet_class_id,
            "confidence": confidence
        }
        if imagenet_labels is not None:
            retval["imagenet_class_label"] = imagenet_labels[imagenet_class_id]
        
        return retval




        
