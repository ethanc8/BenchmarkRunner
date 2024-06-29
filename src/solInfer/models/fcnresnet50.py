# SPDX-License-Identifier: AGPL-3.0-or-later
from solInfer.models.common import *
from torchvision import models
from solInfer import backends
import torch
import os

class FCNResNet50(Model):
    @classmethod
    def get_pretrained(self) -> backends.Net:
        return backends.pytorch.Net(models.segmentation.fcn_resnet50(pretrained=True))

    @classmethod
    def convert_to_disk_format(self, disk_format: DiskFormat, pytorch_net: backends.pytorch.Net, filename: str = "resnet50.onnx", image_size = (256, 256)):
        if disk_format == disk_formats.ONNX:
            # generate model input
            # TODO: Figure out how to deal with images that are not 366x500px, without having to pass in image size
            # before generating the ONNX file
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
    def infer(self, net: backends.Net, input_img: backends.Tensor) -> dict:
        out = net.forwardPass(input_img)
        # Postprocess the output
        # (batch_size == 1, num_classes, height, width) -> (1, height, width, num_classes)
        out = out.permute((0, 2, 3, 1))
        # (1, height, width, num_classes) -> (height, width)
        seg_map = out.argmax(axis=3)[0]
        return seg_map




        
