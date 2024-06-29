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
    def convert_to_disk_format(self, disk_format: DiskFormat, pytorch_net: backends.pytorch.Net, filename: str = "resnet50.onnx"):
        if disk_format == disk_formats.ONNX:
            # generate model input
            generated_input = torch.autograd.Variable(
                torch.randn(1, 3, 500, 500)
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
        # print(f"out: {out.data}")
        print(f"shape: {out.data.shape}")
        predictions = out.argmax(axis=0)
        print(f"predictions shape: {predictions.data.shape}")
        return predictions




        
